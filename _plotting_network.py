import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mp

from _util import color_dictionary
colors = color_dictionary()

#%%

def plot_A(A):

    color_map = mp.colors.ListedColormap([colors['grey1'],colors['black']]) # plt.cm.viridis
    
    num_nodes = np.shape(A)[0]  

    fig, ax = plt.subplots(1,1)
    # A_plot = ax.imshow(A, cmap = plt.cm.viridis, interpolation='none', extent=[0,num_nodes,0,num_nodes], aspect = 'equal', origin = 'lower')
    A_plot = ax.imshow(A, cmap = color_map, interpolation='none', extent=[0,num_nodes,0,num_nodes], aspect = 'equal', origin = 'lower')
    cbar = fig.colorbar(A_plot, extend='both')
    cbar.minorticks_on()     
    # fig.suptitle('Adjacency matrix')
    plt.title('Adjacency matrix')
    # ax.set_xlabel(r'{}'.format(x_label))
    # ax.set_ylabel(r'{}'.format(y_label))   
    plt.show()      
    # fig.savefig('figures/'+save_str+'__log.png') 
    
    return


def plot_network_spikes_raster(neuron_spikes__raster):
    
    color_map = mp.colors.ListedColormap([colors['grey1'],colors['black']]) # plt.cm.viridis
    
    num_nodes = np.shape(neuron_spikes__raster)[0]
    num_times = np.shape(neuron_spikes__raster)[1]

    fig, ax = plt.subplots(1,1)
    raster_plot = ax.imshow(neuron_spikes__raster, cmap = color_map, interpolation='none', extent=[0,num_times,0,num_nodes], aspect = 'auto', origin = 'lower')
    cbar = fig.colorbar(raster_plot, extend='both')
    cbar.minorticks_on()     
    plt.title('Network Spikes Raster')
    ax.set_xlabel(r'{}'.format('Time step'))
    ax.set_ylabel(r'{}'.format('Neuron index'))   
    plt.show()
    
    
    return


def plot_network_spikes_binned(network_spikes__binned):
    
    fig = plt.figure()    
    ax = fig.gca()
    
    ax.plot(network_spikes__binned, '-', color = colors['blue3'])                    
    ax.set_xlabel(r'Time bin')
    ax.set_ylabel(r'Num spikes')
    # ax.legend()
    
    plt.show()
        
    return


def plot_network_spikes_binned__mark_avalanches(network_spikes__binned,start_indices,stop_indices):
    
    fig = plt.figure()    
    ax = fig.gca()

    ax.plot(network_spikes__binned, '-', color = colors['blue3'])     
    min_spikes = np.min(network_spikes__binned)
    max_spikes = np.max(network_spikes__binned)
    for ii in range(len(start_indices)):
        if ii == 0:
            ax.plot([start_indices[ii],start_indices[ii]],[min_spikes,max_spikes], ':', color = colors['green3'], label = 'avalanche start')                   
            ax.plot([stop_indices[ii],stop_indices[ii]],[min_spikes,max_spikes], ':', color = colors['red3'], label = 'avalanche stop')   
        else:
            ax.plot([start_indices[ii],start_indices[ii]],[min_spikes,max_spikes], ':', color = colors['green3'])                   
            ax.plot([stop_indices[ii],stop_indices[ii]],[min_spikes,max_spikes], ':', color = colors['red3'])                 
    ax.set_xlabel(r'Time bin')
    ax.set_ylabel(r'Num spikes')
    ax.legend()
    
    plt.show()
        
    return

def plot_neuronal_avalanche_histograms(size,size_bins,duration,duration_bins):
    
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
    fig.suptitle('Histograms of Neuronal Avalanches')
    
    axs[0].plot(size_bins[0:-1],size, '-', color = colors['blue3'])  
    axs[0].set_xlabel(r'Size of neuronal avalanche')  
    axs[0].set_ylabel(r'Num avalanches of that size')
    
    axs[1].plot(duration_bins[0:-1],duration, '-', color = colors['green3'])  
    axs[1].set_xlabel(r'Duration of neuronal avalanches') 
    axs[1].set_ylabel(r'Num avalanches of that duration')
    
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
    fig.suptitle('Histograms of Neuronal Avalanches')
    
    size_gt_zero = np.where( size > 0 )[0] # eliminate zeros from log plots
    axs[0].loglog(size_bins[size_gt_zero],size[size_gt_zero], '-', color = colors['blue3'])  
    axs[0].set_xlabel(r'Size of neuronal avalanche')  
    axs[0].set_ylabel(r'Num avalanches of that size')
    
    dur_gt_zero = np.where( duration > 0 )[0] # eliminate zeros from log plots
    axs[1].loglog(duration_bins[dur_gt_zero],duration[dur_gt_zero], '-', color = colors['green3'])  
    axs[1].set_xlabel(r'Duration of neuronal avalanches') 
    axs[1].set_ylabel(r'Num avalanches of that duration')    
    
    # fig = plt.hist(size, bins = size_bins, align = 'mid', log = True, color = colors['blue3'])
    
    return

def plot_neuronal_avalanche_histograms__with_fits(size,size_bins,size_fit,size_vec_dense,size_power,size_residuals,duration,duration_bins,duration_fit,duration_vec_dense,duration_power,duration_residuals):
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
    fig.suptitle('Histograms of Neuronal Avalanches')
    
    size_gt_zero = np.where( size > 0 )[0] # eliminate zeros from log plots
    axs[0].loglog(size_bins[size_gt_zero],size[size_gt_zero], '-', color = colors['blue3'], label = 'Avalanche size data')  
    axs[0].loglog(size_vec_dense,size_fit, '-.', color = colors['red3'], label = 'fit: exponent = {:4.2f}; residual = {:4.2e}'.format(size_power,size_residuals)) 
    axs[0].set_xlabel(r'Size of neuronal avalanche')  
    axs[0].set_ylabel(r'Num avalanches of that size')
    axs[0].legend()
    
    duration_gt_zero = np.where( duration > 0 )[0] # eliminate zeros from log plots
    axs[1].loglog(duration_bins[duration_gt_zero],duration[duration_gt_zero], '-', color = colors['blue3'], label = 'Avalanche duration data')  
    axs[1].loglog(duration_vec_dense,duration_fit, '-.', color = colors['red3'], label = 'fit: exponent = {:4.2f}; residual = {:4.2e}'.format(duration_power,duration_residuals)) 
    axs[1].set_xlabel(r'Duration of neuronal avalanche')  
    axs[1].set_ylabel(r'Num avalanches of that duration')
    axs[1].legend()  
    
    return

