import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plt_learning_curve(neural_res, ax, fig_width, fig_height, title, l_ylim, r_ylim, xlim):
    neural_responses = np.load(neural_res)
    R_fea_list, R_null_list = neural_responses['arr_0'], neural_responses['arr_1']

    matplotlib.rcParams['figure.figsize'] = (fig_width, fig_height)
    cycles = np.arange(0, R_null_list.shape[0], 1)

    color = ['r','b','g','m', '#FF6600', '#00ffff', '#FDEE00', '#D71868', 'y', 'c', 'k']
    
    ax1 = ax.twinx()
    ax.plot(cycles, R_null_list, 'k')
    count = 0
    for fea in R_fea_list:
        ax1.plot(cycles, fea, color[count])
        count += 1

    ax.set_xlabel('Cycles', fontsize=10)
    ax.set_ylabel('Rate (Hz)', fontsize=10)
    ax1.set_ylabel('Spikes', fontsize=10)

    if title:
        plt.title(title, fontsize=15)
    if l_ylim:
        ax.set_ylim((0, l_ylim))
    if r_ylim:
        ax1.set_ylim((0, r_ylim))
    if xlim:
        plt.xlim((0, xlim))

