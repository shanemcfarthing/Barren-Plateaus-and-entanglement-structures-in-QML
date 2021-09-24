from joblib import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plotResults(results, num_qubits, depth, mode, entanglement):

    plt.figure(figsize=(12, 6))

    if mode == "qubits":
        rep_count = 1
        for r in results:
            user_input = input('Plot next function? (y/n)\n>')
            if user_input == 'y':
                label_string = 'local cost, depth of ' + str(rep_count)
                rep_count += 1
                plt.semilogy(num_qubits, np.var(r, axis=1), 'o-', label=label_string)
                plt.legend(loc='best', fontsize='x-small')

                plt.title(entanglement + " Entanglement")
                plt.xlabel('number of qubits')
                plt.ylabel(r'$\mathrm{Var}[\partial_{\theta 1} \langle E(\theta) \rangle]$')
                plt.legend(loc='best', fontsize='x-small')
            else:
                break

    elif mode == "reps":
        qubit_count = 2
        for r in results:
            user_input = input('Plot next function? (y/n)\n>')
            if user_input == 'y':
                label_string = 'local cost, ' + str(qubit_count) + ' qubits'
                qubit_count += 1
                plt.semilogy(depth, r, 'o-', label=label_string)
                plt.legend(loc='best', fontsize='x-small')

                plt.title(entanglement + " Entanglement")
                plt.xlabel('number of repititions')
                plt.ylabel(r'$\mathrm{Var}[\partial_{\theta 1} \langle E(\theta) \rangle]$')
                plt.legend(loc='best', fontsize='x-small')
            else:
                break


def plotResultObjects(results, num_qubits, num_reps, mode, entanglement):

    plt.figure(figsize=(12, 6))

    if mode == "qubits":
        for r in range(len(results)):
            user_input = input('Plot next function? (y/n)\n>')
            if user_input == 'y':
                label_string = 'local cost, depth of ' + str(r + 1)
                plt.semilogy(num_qubits, np.var(results[r], axis=1), 'o-', label=label_string)
                plt.legend(loc='best', fontsize='x-small')

                plt.title(entanglement + " Entanglement")
                plt.xlabel('number of qubits')
                plt.ylabel(r'$\mathrm{Var}[\partial_{\theta 1} \langle E(\theta) \rangle]$')
                plt.legend(loc='best', fontsize='x-small')
            else:
                break

    elif mode == "reps":
        for r in results:
            user_input = input('Plot next function? (y/n)\n>')
            if user_input == 'y':
                label_string = 'local cost, ' + str(r.num_qubits) + ' qubits'
                plt.semilogy(depth, np.var(r.gradients, axis=1), 'o-', label=label_string)
                plt.legend(loc='best', fontsize='x-small')

                plt.title(r.entanglement_mode + " Entanglement")
                plt.xlabel('number of repititions')
                plt.ylabel(r'$\mathrm{Var}[\partial_{\theta 1} \langle E(\theta) \rangle]$')
                plt.legend(loc='best', fontsize='x-small')
            else:
                break
