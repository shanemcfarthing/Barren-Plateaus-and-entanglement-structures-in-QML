# Honours Project
 Contains the code for investigating the entanglement threshhold in variational forms.
 
 Required Packages: 
 
    qiskit
    
    qiskit_machine_learning
    
    sklearn
    
    matplotlib
    
    numpy
    
    itertools
    
    timeit
    
    pandas
 
 Breakdown of Files:
 
    Barren Plateaus in Quantum Machine Learning.ipynb -  This is the jupyter notebook containing the project and project code
 
    AlteredADAM.py - This python file contains a version of the Adam optimizer, using the Qiskit implementation, but with the added functionality 
                     of tracking and returning the gradients which are calculated at each training iteration. This file also contains the TrackingResult
                     class, instances of which hold all of the relevant information and results for each quantum circuit that is run.
                     
    Datasets - This folder contains the datasets.txt file, which is holds the pickled dictionary of datasets used in the project.
    
    Parameters - Within its subfolders are contained the adam_params.csv files for each circuit that is run. These files hold the parameters which are
                 generated and used at each iteration of the optimization process for the variational quantum circuits.
    
    Results - Within its subfolders are the text files which contain the pickled TrackingResult objects for each trained VQC.
