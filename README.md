# Honours Project
 In this investigation, the linear, full, circular, and shifted-circular-alternating entanglement schemes offered by qiskit are utilised to construct
 Variational Quantum Circuits, with the purpose of observing how the choice of entanglement scheme affects the occurrence of barren plateaus when training
 these circuits. 
 
 For each circuit, the von Neumann Entropy of Entanglement is also calculated at each training iteration, and this is compared with the variance of the gradients 
 calculated during the optimisation so that the relationship between entanglement in the system and the occurrence of barren plateaus can be investigated, with 
 respect to the different entanglement schemes used.
 
 
 Required Packages: 
 
    qiskit
    
    qiskit_machine_learning
    
    pylatexenc
    
    sklearn
    
    matplotlib
    
    numpy
    
    itertools
    
    timeit
    
    pandas
 
 Breakdown of Files:
 
    Barren Plateaus in Quantum Machine Learning.ipynb -  This is the jupyter notebook containing the project and project code
 
    AlteredADAM.py - This python file contains a version of the SPSA optimizer, using the Qiskit implementation, but with the added functionality 
                     of saving the parameters to a csv file called 'spsa_params.csv'. This file also contains the TrackingResult
                     class, instances of which hold all of the relevant information and results for each quantum circuit that is run.
                     
    Datasets - This folder contains the datasets.txt file, which is holds the pickled dictionary of datasets used in the project.
    
    Parameters - Within its subfolders are contained the adam_params.csv files for each circuit that is run. These files hold the parameters which are
                 generated and used at each iteration of the optimization process for the variational quantum circuits.
    
    Results - Within its subfolders are the text files which contain the pickled TrackingResult objects for each trained VQC.
