from Plot2 import *
from joblib import load
import numpy as np

# sort the results so that for each Number of Repetitions, a graph of 'Variance in
# gradient over the number of qubits' can be created.


def sortAndNormalizeByReps(results, rep_number):
    rep = []

    for j in results:
        for i in j:
            if i.num_reps == rep_number:
                rep.append(i)

    rep_grads = []

    for i in rep:
        rep_grads.append(np.var(i.gradients, axis=1))

    return rep_grads


num_qubits = list(range(2, 16))
num_reps = list(range(1, 21))

user_input = input('Would you like to plot the results from the models trained with a dataset or without a dataset? (with/without)?\n>')
if user_input == 'with':
    linearResults = []
    for i in range(2, 16):
        string = str(i) + 'qubits'
        linearResults.append(load('./Results/With Datasets/linear_classifiers/' + string + '.txt'))

    fullResults = []
    for i in range(2, 16):
        string = str(i) + 'qubits'
        fullResults.append(load('./Results/With Datasets/full_classifiers/' + string + '.txt'))

    circularResults = []
    for i in range(2, 16):
        string = str(i) + 'qubits'
        circularResults.append(load('./Results/With Datasets/circular_classifiers/' + string + '.txt'))

    scaResults = []
    for i in range(2, 16):
        string = str(i) + 'qubits'
        scaResults.append(load('./Results/With Datasets/sca_classifiers/' + string + '.txt'))

    linear_filtered_results = []
    for i in range(1, 21):
        linear_filtered_results.append(sortAndNormalizeByReps(linearResults, i))

    full_filtered_results = []
    for i in range(1, 21):
        full_filtered_results.append(sortAndNormalizeByReps(fullResults, i))

    circular_filtered_results = []
    for i in range(1, 21):
        circular_filtered_results.append(sortAndNormalizeByReps(circularResults, i))

    sca_filtered_results = []
    for i in range(1, 21):
        sca_filtered_results.append(sortAndNormalizeByReps(scaResults, i))

    #plotResultObjects(linear_filtered_results, num_qubits, num_reps, 'qubits', 'Linear')
    #plotResultObjects(full_filtered_results, num_qubits, num_reps, 'qubits', 'Full')
    #plotResultObjects(circular_filtered_results, num_qubits, num_reps, 'qubits', 'Circular')
    plotResultObjects(sca_filtered_results, num_qubits, num_reps, 'qubits', 'Shifted-Circular-Alternating')

elif user_input == 'without':

    graph_plot = input("With Reps or Qubits on the x-axis? (reps/qubits)\n>")

    if graph_plot == 'qubits':

        linear_results = load('./Results/Without Datasets/linear_results.txt')
        full_results = load('./Results/Without Datasets/full_results.txt')
        circular_results = load('./Results/Without Datasets/circular_results.txt')
        sca_results = load('./Results/Without Datasets/sca_results.txt')

        #plotResults(linear_results, num_qubits, num_reps, 'qubits', 'Linear')
        #plotResults(full_results, num_qubits, num_reps, 'qubits', 'Full')
        #plotResults(circular_results, num_qubits, num_reps, 'qubits', 'Circular')
        plotResults(sca_results, num_qubits, num_reps, 'qubits', 'Shifted-Circular-Alternating')

    elif graph_plot == "reps":

        linear_results = load('./Results/Without Datasets/linear_results2.txt')
        full_results = load('./Results/Without Datasets/full_results2.txt')
        circular_results = load('./Results/Without Datasets/circular_results2.txt')
        sca_results = load('./Results/Without Datasets/sca_results2.txt')

        plotResults(linear_results, num_qubits, num_reps, 'reps', 'Linear')
        plotResults(full_results, num_qubits, num_reps, 'reps', 'Full')
        plotResults(circular_results, num_qubits, num_reps, 'reps', 'Circular')
        plotResults(sca_results, num_qubits, num_reps, 'reps', 'Shifted-Circular-Alternating')
