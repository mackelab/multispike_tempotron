import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def get_memory_len(kernel_array, ratio):
    """
    Return the number of time bins until kernel_array has decreased by the factor 'ratio'
    'kernel_array' may initially rise, but must be monotonically decreasing from 
    the maximum.
    """
    arr = (kernel_array - ratio*kernel_array.max())[::-1]
        # The point where this array reaches zero is the desired memory time
        # We flip the order with [::-1] because np.searchsorted expects increasing order
    memory_len = len(kernel_array) - np.searchsorted(arr, 0)
    return memory_len

def find_first_spike(V, threshold):
    # Based on the equivalent code to py_find_1st (https://github.com/roebel/py_find_1st)
    ind = np.flatnonzero(V >= threshold)
    if len(ind):
        return ind[0]
    else:
        return -1

def calculate_Vt(unweighted_input, theta, omega):
    spike_time = []
    datalen = unweighted_input.shape[1]
    V = (omega[:,np.newaxis] * unweighted_input).sum(axis=0)
    ref_kernel = np.exp(- np.arange(1000) / 20) #tau_mem = 20
    ref_memory_len = get_memory_len(ref_kernel, ratio=0.001)
    new_ref_kernel = ref_kernel[:ref_memory_len]
    done = False
    while not done:
        spike_idx = find_first_spike(V, theta)
        if spike_idx == -1:
            done = True
        else:
            spike_time.append(spike_idx)
            mem_len = min(ref_memory_len, datalen - spike_idx)
            V[spike_idx:spike_idx+mem_len] -= theta * new_ref_kernel[:mem_len]
    return V, spike_time

def plt_V(in_data, omega, cycle):
    input_data = np.load(in_data)
    data, presyn_input, markers, n_fea_occur, fea_time, fea_order = input_data['arr_0'], input_data['arr_1'], input_data['arr_2'], input_data['arr_3'], input_data['arr_4'], input_data['arr_5']

    if cycle != None:
        updated_omega = np.load(omega)[cycle]
    else:
        updated_omega = np.load(omega)
    time = np.arange(0, len(presyn_input[1]), 1)
    V_t, spike_time = calculate_Vt(presyn_input, 1, updated_omega)

    matplotlib.rcParams['figure.figsize'] = (15.0, 3.5)
    plt.xlabel("Time (ms)", fontsize=10)
    plt.ylabel("Voltage", fontsize=10)
    plt.plot(time, V_t)
    for marker in markers:
        plt.gca().add_patch(marker)
    plt.ylim((0, 1.2))

def multi_Vplot(row, col, data, omega, cycle, title, fig_width, fig_height):
    fig, axs = plt.subplots(row, col, figsize=(fig_width, fig_height))

    for n in range(len(cycle)):
        axs = plt.subplot(row, col, n+1)
        plt_V(data, omega[n], cycle[n])
        axs.set_title(title[n], fontsize = 15)

    fig.tight_layout()
