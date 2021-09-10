from Configuration import *
import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np

def plotResults(resultObjects):

    # graphs will be constructed for each number of qubits, so these lists will
    # hold their respective results to be plotted
    two_qubits = []
    four_qubits = []
    six_qubits = []
    eight_qubits = []
    ten_qubits = []
    twelve_qubits = []

    # allocate the results to their respective graph lists
    for r in resultObjects:
        if r.num_qubits == 2:
            two_qubits.append(r)

        elif r.num_qubits == 4:
            four_qubits.append(r)

        elif r.num_qubits == 6:
            six_qubits.append(r)

        elif r.num_qubits == 8:
            eight_qubits.append(r)

        elif r.num_qubits == 10:
            ten_qubits.append(r)

        elif r.num_qubits == 12:
            twelve_qubits.append(r)

    # create the values that will be plotted along the x-axis of the graphs.
    # they represent the optimizer iterations for which the gradients are calculated
    num_iterations = len(two_qubits[0].gradients)
    iterations = []
    for it in range(1, num_iterations + 1):
        iterations.append(it)

    # add all of the result lists to a master list to reuse plotting code
    graph_values = [two_qubits, four_qubits, six_qubits, eight_qubits, ten_qubits, twelve_qubits]

    fig, axs = plt.subplots(2, 3)
    index = 0
    for i in range(2):
        for j in range(3):
            for result in graph_values[index]:
                label_string = 'Depth of ' + str(result.depth)
                axs[i, j].plot(iterations, list(result.gradients.values()), label=label_string)

            title = str(graph_values[index][0].num_qubits) + " Qubits"
            axs[i, j].set_title(title)
            axs[i, j].legend()
            index += 1

    for ax in axs.flat:
        ax.set(xlabel='Optimizer Iteration', ylabel='Norm of Gradient')

    main_title = 'Variational forms using ' + two_qubits[0].entanglement + ' entanglement scheme'
    fig.suptitle(main_title)

    # Uncomment this code if functionality to save the figures is desired
    #folder_name = './Graph results/'
    #file_name = folder_name + two_qubits[0].entanglement + ".png"
    #fig.savefig(file_name, dpi=600, bbox_inches='tight')

    for q in graph_values:
        for result in q:
            for solution in result.eigenvalues:
                result.eigenvalues[solution] = np.real(result.eigenvalues[solution])

    fig, axs = plt.subplots(2, 3)
    index = 0
    for i in range(2):
        for j in range(3):
            for result in graph_values[index]:
                label_string = 'Depth of ' + str(result.depth)
                axs[i, j].plot(iterations, list(result.eigenvalues.values()), label=label_string)

            title = str(graph_values[index][0].num_qubits) + " Qubits"
            axs[i, j].set_title(title)
            axs[i, j].legend()
            index += 1

    for ax in axs.flat:
        ax.set(xlabel='Optimizer Iteration', ylabel='Solution')

    # for ax in axs.flat:
        # ax.label_outer()

    # fig.set_figheight()
    # fig.set_figwidth()

    main_title = 'Variational forms using ' + two_qubits[0].entanglement + ' entanglement scheme'
    fig.suptitle(main_title)

    # Uncomment this code if functionality to save the figures is desired
    #folder_name = './Graph results/'
    #file_name = folder_name + two_qubits[0].entanglement + ".png"
    #fig.savefig(file_name, dpi=600, bbox_inches='tight')


