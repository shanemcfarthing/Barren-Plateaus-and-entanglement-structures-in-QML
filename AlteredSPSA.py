# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Altered Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

This implementation allows both, standard first-order as well as second-order SPSA,
with the added functionality of being able to track the optimization gradients and
loss for the parameters of neural network at each iteration. These changes are
implemented in the minimize function.

"""

from typing import Iterator, Optional, Union, Callable, Tuple, Dict
import logging
import warnings
from time import time

from collections import deque
import scipy
import numpy as np

from qiskit.utils import algorithm_globals

from qiskit.algorithms.optimizers.optimizer import Optimizer, OptimizerSupportLevel
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms.objective_functions import BinaryObjectiveFunction

# number of function evaluations, parameters, loss, stepsize, accepted
CALLBACK = Callable[[int, np.ndarray, float, float, bool], None]

logger = logging.getLogger(__name__)

class AlteredSPSA(SPSA):

    def _minimize(self, loss, initial_point):

        # ensure learning rate and perturbation are correctly set: either none or both
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            get_eta, get_eps = self.calibrate(loss, initial_point)
        else:
            get_eta, get_eps = self._validate_pert_and_learningrate(
                self.perturbation, self.learning_rate
            )
        eta, eps = get_eta(), get_eps()

        if self.lse_solver is None:
            lse_solver = np.linalg.solve
        else:
            lse_solver = self.lse_solver

        # prepare some initials
        x = np.asarray(initial_point)
        if self.initial_hessian is None:
            self._smoothed_hessian = np.identity(x.size)
        else:
            self._smoothed_hessian = self.initial_hessian

        self._nfev = 0

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = loss(x)

            self._nfev += 1
            if self.allowed_increase is None:
                self.allowed_increase = 2 * self.estimate_stddev(loss, x)

        logger.info("=" * 30)
        logger.info("Starting SPSA optimization")
        start = time()

        # keep track of the last few steps to return their average
        last_steps = deque([x])

        # this list will be used to store the gradients computed at each iteration
        # of the optimizer
        grads = []

        # this list will be used to store the loss computed for each set of parameters
        loss_values = []

        for k in range(1, self.maxiter + 1):
            #print('Optimizer iteration: ', k)
            iteration_start = time()
            # compute update
            update = self._compute_update(loss, x, k, next(eps), lse_solver)

            # store the gradient estimate in the record of calculated gradients
            grads.append(update)

            # store the loss for the current set of parameters
            loss_values.append(loss(x))

            # trust region
            if self.trust_region:
                norm = np.linalg.norm(update)
                if norm > 1:  # stop from dividing by 0
                    update = update / norm

            # compute next parameter value
            update = update * next(eta)
            x_next = x - update

            # blocking
            if self.blocking:
                self._nfev += 1
                fx_next = loss(x_next)

                if fx + self.allowed_increase <= fx_next:  # accept only if loss improved
                    if self.callback is not None:
                        self.callback(
                            self._nfev,  # number of function evals
                            x_next,  # next parameters
                            fx_next,  # loss at next parameters
                            np.linalg.norm(update),  # size of the update step
                            False,
                        )  # not accepted

                    logger.info(
                        "Iteration %s/%s rejected in %s.",
                        k,
                        self.maxiter + 1,
                        time() - iteration_start,
                    )
                    continue
                fx = fx_next

            logger.info(
                "Iteration %s/%s done in %s.", k, self.maxiter + 1, time() - iteration_start
            )

            if self.callback is not None:
                # if we didn't evaluate the function yet, do it now
                if not self.blocking:
                    self._nfev += 1
                    fx_next = loss(x_next)

                self.callback(
                    self._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(update),  # size of the update step
                    True,
                )  # accepted

            # update parameters
            x = x_next

            # update the list of the last ``last_avg`` parameters
            if self.last_avg > 1:
                last_steps.append(x_next)
                if len(last_steps) > self.last_avg:
                    last_steps.popleft()

        logger.info("SPSA finished in %s", time() - start)
        logger.info("=" * 30)

        if self.last_avg > 1:
            x = np.mean(last_steps, axis=0)

        return x, loss(x), self._nfev, grads, loss_values

    def _validate_pert_and_learningrate(self, perturbation, learning_rate):
        if learning_rate is None or perturbation is None:
            raise ValueError("If one of learning rate or perturbation is set, both must be set.")

        if isinstance(perturbation, float):

            def get_eps():
                return self.constant(perturbation)

        elif isinstance(perturbation, (list, np.ndarray)):

            def get_eps():
                return iter(perturbation)

        else:
            get_eps = perturbation

        if isinstance(learning_rate, float):

            def get_eta():
                return self.constant(learning_rate)

        elif isinstance(learning_rate, (list, np.ndarray)):

            def get_eta():
                return iter(learning_rate)

        else:
            get_eta = learning_rate

        return get_eta, get_eps

    def constant(self, eta=0.01):
        """Yield a constant series."""

        while True:
            yield eta
