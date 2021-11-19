# A comparison of ansatz entanglement structures for mitigating barren plateaus in Quantum Machine Learning
 In this investigation, the linear, full, circular, and shifted-circular-alternating entanglement schemes offered by qiskit are utilised to construct
 Variational Quantum Circuits, with the purpose of observing how the choice of entanglement scheme affects the occurrence of barren plateaus when training
 these circuits. 
 
 For each circuit, the von Neumann Entropy of Entanglement is also calculated at each training iteration, and this is compared with the variance of the gradients 
 calculated during the optimisation so that the relationship between entanglement in the system and the occurrence of barren plateaus can be investigated, with 
 respect to the different entanglement schemes used.
 
 Below are the software and library requirements to run the code in the Jupyter Notebook for the project. All of the versions with which this code was developed
 and tested with are included for completeness. 
 
 Important Note: 
 
   Please ensure that the correct versions of python and joblib are used, as the reading of the precomputed results relies on these two. This has been confirmed to
   work with the versions listed below, but has not been tested with any other versions and so you may encounter issues if other versions are used.
 
 Software requirements:
    
    python=3.8.8
    jupyter notebook
 
 Library requirements:
 
    qiskit=0.31.0
    qiskit_machine_learning=0.2.1
    joblib=1.0.1
    qutip=4.6.2
    pylatexenc=2.10
    matplotlib=3.3.4
    pandas=1.2.4
    numpy=1.20.1 (should automatically be installed as a dependency of qiskit)
    sklearn=1.0.1 (should automatically be installed as a dependency of qiskit_machine_learning)
    
 Recommendation for installing dependencies:
  
    1.) Download and install the Anaconda individual edition, which will include python and Jupyter Notebook.
    2.) Add the 'conda-forge' channel to conda sources using the 'conda config --add channels conda-forge' command, thereby allowing the installation
        of libraries without Visual Studio.
    3.) Run the command 'conda install x' command for each of the dependencies listed above (substituting the libary name and version for x,
        in the format 'name=version'), except for jupyter notebook as this was installed with anaconda.
        
 Running instructions:
 
    Open Jupyter Notebook using the command 'jupyter notebook' in a terminal. This opens jupter notebook in a browser interface, and then you can use
    the accompanying file explorer to open the 'Barren Plateaus in Quantum Machine Learning.ipynb' file.
 
 Breakdown of Files:
 
    Barren Plateaus in Quantum Machine Learning.ipynb -  This is the jupyter notebook containing the project and project code
 
    AlteredSPSA.py - This python file contains a version of the SPSA optimizer, using the Qiskit implementation, but with the added functionality 
                     of saving the parameters to a csv file called 'spsa_params.csv'. This file also contains the TrackingResult
                     class, instances of which hold all of the relevant information and results for each quantum circuit that is run.
                     
    Datasets - This folder contains the datasets.txt file, which is holds the pickled dictionary of datasets used in the project.
    
    Parameters - Within the subfolders are contained the 'spsa_params.csv' files for each circuit that is run. These files hold the parameters which are
                 generated and used at each iteration of the optimization process for the variational quantum circuits. If the program is rerun, the parameters
                 will be appended to the existing file, the file will not be overwritten.
    
    Results - Within the subfolders are the text files which contain the pickled TrackingResult objects for each trained VQC.
