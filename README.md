# Honours Project
 Contains the code for investigating the entanglement threshhold in variational forms.
 
 Required Packages are qiskit, qiskit_nature, pyscf, and numpy.
 
 Breakdown of Files:
 Project.py - contains the main program of the project and is the file that should be run.
 Configuration.py - contains helper classes and functions.
 ModifiedAdam.py - contains an altered ADAM optimizer class which has gradient tracking functionality.
 ModifiedVQE.py - contains an altered VQE which makes use of the modified ADAM optimizer to utilize the gradients from the optimization procedure.
