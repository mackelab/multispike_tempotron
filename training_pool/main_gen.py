import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.patches as patches

def gen_background_data(n, fr, dt, duration): 
    gen_bg = np.random.random((n, np.rint(duration/dt).astype(int)))<fr*dt
    gen_bg = gen_bg.astype(int)
    return gen_bg

def order_occur(n_fea, n_fea_occur):
    count = 0
    order = []
    while count < n_fea:
        order += [count]*n_fea_occur[count]
        count += 1
    return np.random.permutation(order)

def gen_features(n_fea, n, fr, dt, T_fea): 
    a = 0
    features = {}
    while a < n_fea:
        features['feature_'+str(a)] = (np.random.random((n, np.rint(T_fea/dt).astype(int)))<fr*dt).astype(int)
        a += 1
    return features

def add_marker(marker_y, marker_height, time_occur, feature_order, T_fea, dt):
    index = time_occur/dt
    markers = []
    color = ['r','b','g','m', '#FF6600', '#00ffff', '#FDEE00', '#D71868', 'y', 'c', 'k']
    count = 0
    while count < len(index):
        index[count] += count*T_fea/dt
        markers.append(patches.Rectangle((index[count]-0.5, marker_y), T_fea/dt, marker_height, fc = color[feature_order[count]], ec = '#000000'))
        count += 1
    return markers

def time_occurence(n_fea_occur, T_null):
    return np.sort(np.round(np.random.random(np.sum(n_fea_occur))*T_null, 3))

def n_occur(cf_mean, n_fea):
    return np.random.poisson(cf_mean, n_fea)

def kernel_fn(data, tau_mem, tau_syn, time_ij):
    time = np.arange(0., len(data[0]), 1.) #ms
    kernel = np.zeros(len(data[0]))
    eta = tau_mem/tau_syn
    V_norm = eta**(eta/(eta-1))/(eta-1)
    for count in range(len(data[0])):
        kernel = V_norm*(np.exp(-(time-time_ij)/tau_mem)-np.exp(-(time-time_ij)/tau_syn))
    return kernel

### Alex's code ###
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
### End of Alex's code ###

def gen_input_data(seed, features, T_null, n, fr, dt, tau_mem, tau_syn, time_ij, syn_ratio, n_fea, T_fea, cf_mean, marker_y, marker_height):
    #np.random.seed(1000000000)
    #features = gen_features(n_fea, n, fr, dt, T_fea)
    np.random.seed(seed)
    n_fea_occur = n_occur(cf_mean, n_fea)
    data = gen_background_data(n, fr, dt, T_null)
    time_occur = time_occurence(n_fea_occur, T_null)
    feature_order = order_occur(n_fea, n_fea_occur)
    markers = add_marker(marker_y, marker_height, time_occur, feature_order, T_fea, dt)
    fea_time_idx = np.rint(time_occur/dt).astype(int)
    fea_time = []
    count = 0
    while count < len(fea_time_idx):
        fea_time.append(fea_time_idx[count])
        data = np.insert(data, fea_time_idx[count], features['feature_'+str(feature_order[count])].T, axis = 1)
        fea_time_idx += np.rint(T_fea/dt).astype(int)
        count += 1

    ### Alex's code ###
    datalen = data.shape[1] + 100
    # Create the synaptic kernel, and then remove the negligible tail
    syn_kernel = kernel_fn(data, tau_mem, tau_syn, time_ij)
    syn_memory_len = get_memory_len(syn_kernel, syn_ratio)
    # synaptic memory length
    syn_kernel = syn_kernel[:syn_memory_len]

    # Precompute the unweighted input - this is unaffected by a change in weights
    # Ultimately this should be returned by gen_input_data along with, or instead of, 'data'.
    presyn_input = np.zeros((data.shape[0], datalen))
    for neuron, ith_bin in zip(*np.where(data)):
        mem_len = min(syn_memory_len, datalen - ith_bin)
        presyn_input[neuron,ith_bin:ith_bin+mem_len] += syn_kernel[:mem_len]
    ### End of Alex's code ###
    
    return data, presyn_input, markers, n_fea_occur, fea_time, feature_order
    
