from qiskit.algorithms import VQE, NumPyEigensolver, NumPyMinimumEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP, ADAM
from qiskit import IBMQ, BasicAer, Aer
from qiskit.utils import algorithm_globals
from joblib import *

#from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo

from qiskit_nature.drivers import UnitsType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter

from ModifiedAdam import *
from ModifiedVQE import *
from Configuration import *
from Graphing import *

import timeit

# ___________________________________________________________________________________________________________________
user_input = input("Would you like to run the VQE (y/n)?\n>")

if user_input == "y":

    model_input = input("Which model would you like to run:\nA - Hydrogen Molecule\nB - Maxcut\nC - Travelling Salesman\n>")
    if model_input == "A":

        '''path_extension = "H2"

        # necessary to form the fermionic operator
        driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.7', unit=UnitsType.ANGSTROM,
                                                     charge=0, spin=0, basis='sto3g')
        problem = ElectronicStructureProblem(driver)

        # generate the second-quantized operators
        second_q_ops = problem.second_q_ops()
        main_op = second_q_ops[0]

        particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")

        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals

        # setup the mapper and qubit converter
        mapper = ParityMapper()
        converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

        # map to qubit operators
        qubit_op = converter.convert(main_op, num_particles=num_particles)

        print('Done generating the qubit operator')'''

    elif model_input == "B":

        path_extension = "Maxcut"

        weight_matrix = np.array([[0., 1., 1., 1.],
                                  [1., 0., 1., 0.],
                                  [1., 1., 0., 1.],
                                  [1., 0., 1., 0.]])

        # create instance of maxcut problem
        maxcut_instance = Maxcut(weight_matrix)

        # convert the maxcut problem to a quadratic problem
        quad_p = maxcut_instance.to_quadratic_program()

        # obtain the ising hamiltonian for the problem
        qubit_op, offset = quad_p.to_ising()

    elif model_input == "C":

        path_extension = "Tsp"

        weight_matrix = np.array([[0., 48., 91.],
                                  [48., 0., 63.],
                                  [91., 63., 0.]])

        # create instance of tsp
        tsp_instance = Tsp(weight_matrix)

        # convert the tsp to a quadratic problem
        quad_p = tsp_instance.to_quadratic_program()

        # convert quadratic problem with linear constraints to minimization problem in QUBO format
        converter = QuadraticProgramToQubo()
        qubo_p = converter.convert(quad_p)

        # obtain the ising hamiltonian for the problem
        qubit_op, offset = qubo_p.to_ising()

    start = timeit.default_timer()
    # create the backend to simulate the circuit
    backend = BasicAer.get_backend("statevector_simulator")

    # the options that will be used to create the different variational forms
    num_qubits = (2, 4, 6, 8, 10, 12)
    depths = (3, 6, 9, 12, 15, 18, 20)

    if model_input == "A":
        # instantiate a classical eigensolver to get result for benchmarking purposes
        classical_solver = NumPyEigensolver()
        result = classical_solver.compute_eigenvalues(qubit_op)
        exact_energy = np.real(result.eigenvalues)
        print("Minimum eigenvalue using classical eigensolver: ", exact_energy[0])

    elif model_input == "B" or model_input == "C":
        classical_solver = NumPyMinimumEigensolver()
        result = classical_solver.compute_minimum_eigenvalue(qubit_op)
        exact_energy = np.real(result.eigenvalue)
        print("Minimum eigenvalue using classical eigensolver: ", exact_energy)

    # create the optimizer and set the random number generator seed so that the results are reproducible
    algorithm_globals.random_seed = 10
    optimizer = modifiedADAM(maxiter=500)

# ____________________________________________________________________________________________________________________
    parallel_pool = Parallel(n_jobs=-1)

    # create the configuration objects for the variational forms that are going to be used in the VQE
    # 4 different forms of entanglement will be used
    print('\nCreating linear configs')
    linear_config = createConfigurations('linear', num_qubits, depths)
    print('Creating full configs')
    full_config = createConfigurations('full', num_qubits, depths)
    print('Creating circular configs')
    circular_config = createConfigurations('circular', num_qubits, depths)
    print('Creating sca configs\n')
    sca_config = createConfigurations('sca', num_qubits, depths)

    # run the VQE for each of the linearly entangled variational forms
    linear_results = parallel_pool(delayed(getResult)(l, optimizer, backend, qubit_op) for l in linear_config)
    lResults_file = dump(linear_results, str("./Results/" + path_extension + "/linear_results.txt"))
    print('Done with the linear entangled variational forms')

    # run the VQE for each of the fully entangled variational forms
    full_results = parallel_pool(delayed(getResult)(f, optimizer, backend, qubit_op) for f in full_config)
    fResults_file = dump(full_results, str("./Results/" + path_extension + "/full_results.txt"))
    print('Done with the full entangled variational forms')

    # run the VQE for each of the circular entangled variational forms
    circular_results = parallel_pool(delayed(getResult)(c, optimizer, backend, qubit_op) for c in circular_config)
    cResults_file = dump(circular_results, str("./Results/" + path_extension + "/circular_results.txt"))
    print('Done with the circular entangled variational forms')

    # run the VQE for each of the sca entangled variational forms
    sca_results = parallel_pool(delayed(getResult)(s, optimizer, backend, qubit_op) for s in sca_config)
    sResults_file = dump(sca_results, str("./Results/" + path_extension + "/sca_results.txt"))
    print('Done with the sca entangled variational forms\n')

    elapsed = timeit.default_timer() - start
    print('\nTime taken is ', elapsed, ' seconds')

# ___________________________________________________________________________________________________________________

elif user_input == "n":
    file_input = input("Which model would you like to graph the results of:\nA - Hydrogen Molecule\nB - Maxcut\nC - Travelling Salesman\n>")

    if file_input == "A":
        path_extension = "H2"

    elif file_input == "B":
        path_extension = "Maxcut"

    elif file_input == "C":
        path_extension = "Tsp"

    l_results = load(str("./Results/" + path_extension + "/linear_results.txt"))
    f_results = load(str("./Results/" + path_extension + "/full_results.txt"))
    c_results = load(str("./Results/" + path_extension + "/circular_results.txt"))
    s_results = load(str("./Results/" + path_extension + "/sca_results.txt"))
    plotResults(l_results)
    plotResults(f_results)
    plotResults(c_results)
    plotResults(s_results)



