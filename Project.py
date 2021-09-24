from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from qiskit import Aer, BasicAer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit.opflow import Gradient, StateFn, CircuitSampler, PauliExpectation, Z, I

from joblib import *
import numpy as np

import timeit
import itertools

from Plot2 import *
from Tracking import *
from AlteredSPSA import *

import telegram_send

# _________________________________________________________________________________________________________________________________________________

# Samples the gradients of a specific ansatz using 100 different randomly
# generated parameter sets, and then return a list of the gradients.

def sample_gradients(num_qubits, reps, ent_scheme, local=False):

    index = num_qubits - 1

    if local:
        operator = Z ^ Z ^ (I ^ (num_qubits - 2))
    else:
        operator = Z ^ num_qubits

    ansatz = EfficientSU2(num_qubits, entanglement=ent_scheme, reps=reps)

    expectation = StateFn(operator, is_measurement=True).compose(StateFn(ansatz))
    grad = Gradient().convert(expectation, params=ansatz.parameters[index])

    num_points = 100
    grads = []

    for _ in range(num_points):
        point = np.random.uniform(0, np.pi, ansatz.num_parameters)
        value_dict = dict(zip(ansatz.parameters, point))
        grads.append(sampler.convert(grad, value_dict).eval())

    return grads

# _________________________________________________________________________________________________________________________________________________

# Takes a list of (entanglement mode, number of qubits, number of reps) configurations
# and splits it into sublists of configurations which are grouped by the number
# of qubits. A list of sublists is then returned.

def split_configs(config_list, max_qubits):

    upper = int(len(config_list) / max_qubits)

    split_configs = []
    counter = 0

    while counter < max_qubits:
        lower_index = upper * (counter)
        upper_index = upper * (counter + 1)
        split = linear_configs[lower_index:upper_index]
        split_configs.append(split)
        counter += 1

    return split_configs


# _________________________________________________________________________________________________________________________________________________
# create and store the datasets which will be used for the training and testing
# of the Variational Quantum Classifier

num_qubits = list(range(2, 16))

# if the user wishes to create new datasets, these are created using the sklearn
# make_blobs method which generates isotropic Gaussian blobs. These datasets are
# then stored for future use. In that case, the existing datasets are read from
# text files and then used.

datasets_input = input("Would you like to create new datasets or use existing datasets? (new/existing):\n")

if datasets_input == 'new':

    # create 5 new datasets
    datasets = []

    for i in range(5):

        # create the datasets
        training_data, testing_data = createDatasets(15, 2, num_qubits)
        d = [training_data, testing_data]

        datasets.append([training_data, testing_data])

        # save the datasets to individual files
        dFile = dump(d, './Datasets/Dataset ' + str(i) + '.txt')


elif datasets_input == 'existing':

    datasets = []

    for i in range(5):

        # load the datasets
        d = load('./Datasets/Dataset ' + str(i) + '.txt')
        datasets.append(d)
# _________________________________________________________________________________________________________________________________________________

backend = Aer.get_backend("qasm_simulator")
q_instance = QuantumInstance(backend, shots=8192, seed_simulator=2178, seed_transpiler=2178)

sampler = CircuitSampler(q_instance)

# the optimizer that will be used to train qnn built with the generated circuits
optimizer = AlteredSPSA(maxiter=100, learning_rate=0.01, perturbation=0.01)

depth = list(range(1, 21))

parallel_pool = Parallel(n_jobs=-1)

# these lists will hold the TrackingResult objects returned by the createAndFitClassifier
# method calls. Once they have been initialised with the results, they are dumped
# to text files for storage and future use.

linear_results = []
full_results = []
circular_results = []
sca_results = []

# There is the option of either creating a variational quantum classifier and training
# this model with the given training data, tracking the gradients calculated during
# the optimization of the parameters, or the sample_gradients method can be used
# to track the gradients for different parameters of an ansatz, without using datasets.

user_input = input("Would you like to get gradient results for models trained with datasets, or for models without datasets? (with/without)\n>")

# _________________________________________________________________________________________________________________________________________________

# In this case, a number of VQC's are created (1120 in total), and trained using
# the training data. For each instance, a TrackingResult object is returned, containing
# the gradients and loss values calculated for each iteration of the optimizer.

if user_input == 'with':

    # pick the dataset to train with
    data = datasets[0]

    training_data = data[0]
    testing_data = data[1]

    # the lists of configurations for the circuits that will be tested. Each config
    # details the entanglement mode for the ansatz, the number of qubits, and the
    # number of repititions of the ansatz in the circuit

    linear_configs = list(itertools.product(['linear', ], num_qubits, depth))
    full_configs = list(itertools.product(['full', ], num_qubits, depth))
    circular_configs = list(itertools.product(['circular', ], num_qubits, depth))
    sca_configs = list(itertools.product(['sca', ], num_qubits, depth))

    # split the configuration lists into sublists according to the number of qubits.
    # Each of these sublists will then sequentially be used to generate the neural
    # network classifiers. This is to enable more frequent dumping of results
    # to save files

    linear_split = split_configs(linear_configs, len(num_qubits))
    full_split = split_configs(full_configs, len(num_qubits))
    circular_split = split_configs(circular_configs, len(num_qubits))
    sca_split = split_configs(sca_configs, len(num_qubits))

    counter = 2
    for i in linear_split:

        print("Creating and Fitting Classifiers with Linear Entangled Ansatz")

        message = 'Starting Linear Configs with ' + str(counter) + ' qubits'
        telegram_send.send(messages=[message])

        start = timeit.default_timer()

        trained_linear_classifiers = parallel_pool(delayed(createAndFitClassifier)(c, training_data[counter], optimizer, q_instance) for c in i)
        file = './Results/With Datasets/linear_classifiers/' + str(counter) + 'qubits.txt'
        lFile = dump(trained_linear_classifiers, file)

        elapsed = timeit.default_timer() - start
        print("Finished: ", elapsed, " seconds\n")

        message = 'Linear Configs with ' + str(counter) + ' qubits done in ' + str(elapsed) + ' seconds'
        telegram_send.send(messages=[message])

        counter += 1

    counter = 2

    for i in full_split:

        print("Creating and Fitting Classifiers with Full Entangled Ansatz")

        message = 'Starting Full Configs with ' + str(counter) + ' qubits'
        telegram_send.send(messages=[message])

        start = timeit.default_timer()

        trained_full_classifiers = parallel_pool(delayed(createAndFitClassifier)(c, training_data[counter], optimizer, q_instance) for c in i)
        file = './Results/With Datasets/full_classifiers/' + str(counter) + 'qubits.txt'
        lFile = dump(trained_full_classifiers, file)

        elapsed = timeit.default_timer() - start
        print("Finished: ", elapsed, " seconds\n")

        message = 'Full Configs with ' + str(counter) + ' qubits done in ' + str(elapsed) + ' seconds'
        telegram_send.send(messages=[message])

        counter += 1

    counter = 2

    for i in circular_split:

        print("Creating and Fitting Classifiers with Circular Entangled Ansatz")

        message = 'Starting Circular Configs with ' + str(counter) + ' qubits'
        telegram_send.send(messages=[message])

        start = timeit.default_timer()

        trained_circular_classifiers = parallel_pool(delayed(createAndFitClassifier)(c, training_data[counter], optimizer, q_instance) for c in i)
        file = './Results/With Datasets/circular_classifiers/' + str(counter) + 'qubits.txt'
        lFile = dump(trained_circular_classifiers, file)

        elapsed = timeit.default_timer() - start
        print("Finished: ", elapsed, " seconds\n")

        message = 'Circular Configs with ' + str(counter) + ' qubits done in ' + str(elapsed) + ' seconds'
        telegram_send.send(messages=[message])

        counter += 1

    counter = 2

    for i in sca_split:

        print("Creating and Fitting Classifiers with SCA Entangled Ansatz")

        message = 'Starting SCA Configs with ' + str(counter) + ' qubits'
        telegram_send.send(messages=[message])

        start = timeit.default_timer()

        trained_sca_classifiers = parallel_pool(delayed(createAndFitClassifier)(c, training_data[counter], optimizer, q_instance) for c in i)
        file = './Results/With Datasets/sca_classifiers/' + str(counter) + 'qubits.txt'
        lFile = dump(trained_sca_classifiers, file)

        elapsed = timeit.default_timer() - start
        print("Finished: ", elapsed, " seconds\n")

        message = 'SCA Configs with ' + str(counter) + ' qubits done in ' + str(elapsed) + ' seconds'
        telegram_send.send(messages=[message])

        counter += 1


# _________________________________________________________________________________________________________________________________________________

# Without datasets, the sample_gradients method is called and the gradients for
# 100 different parameter sets are computed and returned. This is done for 1120
# different ansatz.

elif user_input == "without":
    user_input = input("Which attribute would you like to have on x-axis? (reps/qubits)\n>")
    if user_input == 'qubits':
        print('Working')
        start = timeit.default_timer()
        for d in depth:
            linear_entanglement_gradients = parallel_pool(delayed(sample_gradients)(n, d, 'linear', local=True) for n in num_qubits)
            linear_results.append(linear_entanglement_gradients)

        elapsed1 = timeit.default_timer() - start
        print('Linear Done in ', elapsed1, ' seconds')

        start2 = timeit.default_timer()

        for d in depth:
            full_entanglement_gradients = parallel_pool(delayed(sample_gradients)(n, d, 'full', local=True) for n in num_qubits)
            full_results.append(full_entanglement_gradients)

        elapsed2 = timeit.default_timer() - start2

        print('Full done in ', elapsed2, ' seconds')

        start3 = timeit.default_timer()

        for d in depth:
            circular_entanglement_gradients = parallel_pool(delayed(sample_gradients)(n, d, 'circular', local=True) for n in num_qubits)
            circular_results.append(circular_entanglement_gradients)

        elapsed3 = timeit.default_timer() - start3
        print('Circular done in ', elapsed3, ' seconds')

        start4 = timeit.default_timer()

        for d in depth:
            sca_entanglement_gradients = parallel_pool(delayed(sample_gradients)(n, d, 'sca', local=True) for n in num_qubits)
            sca_results.append(sca_entanglement_gradients)

        elapsed4 = timeit.default_timer() - start4
        print('SCA done in ', elapsed4, ' seconds')

        lResults_file = dump(linear_results, str("./Results/Without Datasets/linear_results.txt"))
        fResults_file = dump(full_results, str("./Results/Without Datasets/full_results.txt"))
        cResults_file = dump(circular_results, str("./Results/Without Datasets/circular_results.txt"))
        sResults_file = dump(sca_results, str("./Results/Without Datasets/sca_results.txt"))

    elif user_input == "reps":

        print('Working')
        start = timeit.default_timer()
        for q in num_qubits:
            linear_entanglement_gradients = parallel_pool(delayed(sample_gradients)(q, d, 'linear', local=True) for d in depth)
            linear_results.append(linear_entanglement_gradients)

        elapsed1 = timeit.default_timer() - start
        print('Linear Done in ', elapsed1, ' seconds')

        start2 = timeit.default_timer()

        for q in num_qubits:
            full_entanglement_gradients = parallel_pool(delayed(sample_gradients)(q, d, 'full', local=True) for d in depth)
            full_results.append(full_entanglement_gradients)

        elapsed2 = timeit.default_timer() - start2

        print('Full done in ', elapsed2, ' seconds')

        start3 = timeit.default_timer()

        for q in num_qubits:
            circular_entanglement_gradients = parallel_pool(delayed(sample_gradients)(q, d, 'circular', local=True) for d in depth)
            circular_results.append(circular_entanglement_gradients)

        elapsed3 = timeit.default_timer() - start3
        print('Circular done in ', elapsed3, ' seconds')

        start4 = timeit.default_timer()

        for q in num_qubits:
            sca_entanglement_gradients = parallel_pool(delayed(sample_gradients)(q, d, 'sca', local=True) for d in depth)
            sca_results.append(sca_entanglement_gradients)

        elapsed4 = timeit.default_timer() - start4
        print('SCA done in ', elapsed4, ' seconds')

        lResults_file = dump(linear_results, str("./Results/Without Datasets/linear_results2.txt"))
        fResults_file = dump(full_results, str("./Results/Without Datasets/full_results2.txt"))
        cResults_file = dump(circular_results, str("./Results/Without Datasets/circular_results2.txt"))
        sResults_file = dump(sca_results, str("./Results/Without Datasets/sca_results2.txt"))

