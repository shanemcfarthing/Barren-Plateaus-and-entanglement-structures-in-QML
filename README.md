# Honours Project
 Contains the code for investigating the entanglement threshhold in variational forms.
 
 Required Packages: 
 
    qiskit
    
    qiskit_machine_learning
    
    sklearn
    
    matplotlib
    
    numpy
 
 Breakdown of Files:
 
    Project.py - contains the main program of the project and is the file that should be run.
 
    AlteredSPSA.py - contains a subclass of the SPSA optimizer. The functionality is the same as SPSA, with the one change being that the minimize
                     method tracks and returns the gradients and loss values computed at each iteration.
 
    Plot.py - contains two functions for plotting the results on graphs
 
    PlottingDataResults.py - contains the code to process and plot the results of the main program using the methods from Plot.py
    
    Tracking.py - contains the TrackingResult class, as well as the methods for generating new datasets as well as creating and training the VQC's.
