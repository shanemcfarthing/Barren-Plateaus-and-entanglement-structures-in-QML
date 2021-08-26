from qiskit.algorithms import VQE, NumPyEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP, ADAM
from qiskit import IBMQ, BasicAer, Aer
from joblib import *
from qiskit_nature.drivers.second_quantization.pyscfd import PySCFDriver
from qiskit_nature.drivers import UnitsType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from ModifiedAdam import *
from ModifiedVQE import *
from Configuration import *

#___________________________________________________________________________________________________________________
#use PySCF to compute the one- and two-body integrals in molecular-orbital basis,
#necessary to form the fermionic operator
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.7', unit=UnitsType.ANGSTROM, 
                     charge=0, spin=0, basis='sto3g')
problem = ElectronicStructureProblem(driver)

#generate the second-quantized operators
second_q_ops = problem.second_q_ops()
main_op = second_q_ops[0]

particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")

num_particles = (particle_number.num_alpha, particle_number.num_beta)
num_spin_orbitals = particle_number.num_spin_orbitals

#setup the mapper and qubit converter
mapper = ParityMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

#map to qubit operators
qubit_op = converter.convert(main_op, num_particles=num_particles)

print('Done generating the qubit operator')

#___________________________________________________________________________________________________________________

#create the backend to simulate the circuit
backend = BasicAer.get_backend("statevector_simulator")

#the options that will be used to create the different variational forms
num_qubits = (2,4,6,8,10,12)
depths = (3,6,9,12,15,18,20)

#instantiate a classical eigensolver to get result for benchmarking purposes
classical_solver = NumPyEigensolver()
result = classical_solver.compute_eigenvalues(qubit_op)

exact_energy = np.real(result.eigenvalues)
vqe_energies = []
optimizer = modifiedADAM(maxiter=500)

print("Minimum eigenvalue using classical eigensolver: ",exact_energy[0])

#____________________________________________________________________________________________________________________
parallel_pool = Parallel(n_jobs=16)

linear_config = createConfigurations('linear',num_qubits,depths)
#full_config = createConfigurations('full',num_qubits,depths)
#circular_config = createConfigurations('circular',num_qubits,depths)
#sca_config = createConfigurations('sca',num_qubits,depths)

'''this call currently results in an exception which I have not yet fixed. The getResult call works when called on 
   individual Configuration objects sequentially, but throws an unexpected exception when run in parallel.
   The exception relates to the minimize method in the modifiedAdam optimizer class.'''
linear_results = parallel_pool(delayed(getResult)(l,optimizer,backend,qubit_op) for l in linear_config)

#full_results = parallel_pool(delayed(getResult)(f,optimizer,backend,qubit_op) for f in full_config)
#circular_results = parallel_pool(delayed(getResult)(c,optimizer,backend,qubit_op) for c in circular_config)
#sca_results = parallel_pool(delayed(getResult)(s,optimizer,backend,qubit_op) for s in sca_config)
