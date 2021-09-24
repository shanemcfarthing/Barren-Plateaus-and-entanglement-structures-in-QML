from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit.opflow import Gradient, StateFn, CircuitSampler, PauliExpectation, Z, I

from qiskit_machine_learning.neural_networks import OpflowQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Objects from this class are returned by the createAndFitClassifier method. This
# class provides a means to easily track all of the relevant characteristics and
# results for the classifier that is trained.

class TrackingResult:

    def __init__(self, num_qubits, num_reps, entanglement, gradients=None, loss=None):
        self.num_qubits = num_qubits
        self.num_reps = num_reps
        self.entanglement = entanglement
        self.gradients = gradients
        self.loss = loss


# Isotropic Gaussian blobs are generated, and in this case the instances in the datasets
# have 15 features and belong to 2 classes. Principal Component Analysis is used
# to create versions of the same dataset which have different dimensions. This is done
# so that the same dataset can be used to train classifiers which have different
# input dimensions for the Feature Maps and Ansatz that are used.

def createDatasets(num_features, centers, num_qubits):

    # create the dataset
    X, y = make_blobs(n_features=15, centers=2)

    # standardise the data since PCA is affected by scale
    x_new = StandardScaler().fit_transform(X)

    # this holds the different versions of the same dataset after performing
    # PCA with varying numbers of components to fit the different neural networks
    datasets = {}

    for n in num_qubits:

        pca = PCA(n_components=n)

        # perform the PCA
        pca_x = pca.fit_transform(x_new)

        # store the new dataset with reduced dimensionality
        datasets[n] = [pca_x, y]

    training_datasets = {}
    testing_datasets = {}

    for i in datasets:

        d = datasets[i]

        # split the dataset into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(d[0], d[1])

        train = [X_train, y_train]
        test = [X_test, y_test]

        training_datasets[i] = train
        training_datasets[i] = test

    return training_datasets, testing_datasets


# A quantum circuit is created from a feature map and ansatz, which in turn were
# created using the configuration list passed to the method. Then a Quantum
# Neural Network is created using the circuit as well as a local operator for
# the expectation values. A Neural Network classifier is then created using this
# QNN and then trained using the training data provided, and the dimension of the
# training data will match the number of qubits in the circuit that the QNN is built
# from. A TrackingResult object is then created and returned.

def createAndFitClassifier(config, dataset, optimizer, quantum_instance):

    entanglement_mode = config[0]
    num_qubits = config[1]
    num_reps_ansatz = config[2]

    training_data = dataset[0]
    training_labels = dataset[1]

    # feature map used to encode the classical data
    feature_map = ZZFeatureMap(num_qubits, reps=1)

    # ansatz used to create a trainable model
    ansatz = EfficientSU2(num_qubits=num_qubits, entanglement=entanglement_mode,
                          reps=num_reps_ansatz)

    # create the entire circuit comprised of the two components above
    circuit = feature_map.compose(ansatz)

    # the local operator for the expectation values
    hamiltonian = Z ^ Z ^ (I ^ (num_qubits - 2))

    expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(circuit)

    # create the quantum neural network
    qnn = OpflowQNN(expectation,
                    input_params=list(feature_map.parameters),
                    weight_params=list(ansatz.parameters),
                    exp_val=PauliExpectation(),
                    gradient=Gradient(),
                    quantum_instance=quantum_instance)

    # the classifier which will be trained using the dataset
    classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer)

    # train the classifier using the features and labels of the training data
    classifier.fit(training_data, training_labels)

    print('Done: ', num_qubits, ' qubits and ', num_reps_ansatz, ' reps')

    return TrackingResult(num_qubits, num_reps_ansatz, entanglement_mode, classifier._fit_result[3], classifier._fit_result[4])


