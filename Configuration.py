from qiskit.circuit.library import EfficientSU2
from ModifiedVQE import *
import timeit

# objects of this class represent configurations of the information required
# to create the different variational forms for the VQE
class Configuration:

    def __init__(self, entanglement_type, depth, num_qubits):
        self.entanglement = entanglement_type
        self.depth = depth
        self.num_qubits = num_qubits

# objects of this class will be returned as the results of the main program, and
# hold all of the information required for the final processing of results
class Result:

    def __init__(self, entanglement_type, num_qubits, depth, eigenvalues, gradients):
        self.entanglement = entanglement_type
        self.num_qubits = num_qubits
        self.depth = depth
        self.eigenvalues = eigenvalues
        self.gradients = gradients

# creates a list of configurations to be used when creating the
# variational forms in parallel


def createConfigurations(entanglement_type, num_qubits, depths):
    configurations = []
    for d in depths:
        for n in num_qubits:
            configurations.append(Configuration(entanglement_type, d, n))

    return configurations

# this method handles all the creation of the variational forms and VQE
# instances. It runs the VQE and returns a Result object. This method is
# used in parallelization of the code in Project.py


def getResult(configuration, optimizer, backend, qubit_op):
    #result = vqe.compute_minimum_eigenvalue(qubit_op)
    # return np.real(result.eigenvalue)
    var_form = EfficientSU2(num_qubits=configuration.num_qubits,
                            entanglement=configuration.entanglement,
                            reps=configuration.depth)
    start = timeit.default_timer()
    vqe = modifiedVQE(var_form, optimizer, quantum_instance=backend)
    result, gradients, solutions = vqe.compute_minimum_eigenvalue(qubit_op)
    elapsed = timeit.default_timer() - start
    print("	Done with ",configuration.entanglement,": ",configuration.num_qubits," qubits and ",configuration.depth," layers. Time taken = ",elapsed,"seconds")

    return Result(configuration.entanglement, configuration.num_qubits, configuration.depth, solutions, gradients)
