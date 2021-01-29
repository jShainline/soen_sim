import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mp
import powerlaw

from _util import color_dictionary
colors = color_dictionary()

#%%

def plot_hierarchy(hierarchy):
    
    num_nodes_0 = hierarchy['num_nodes_0']
    num_levels_hier = hierarchy['num_levels_hier']
    gamma = hierarchy['gamma']
    h_vec = hierarchy['h_vec']
    
    num_modules_list = hierarchy['num_modules_list']
    num_nodes_list = hierarchy['num_nodes_list']
    num_nodes_per_module = hierarchy['num_nodes_per_module']
    inter_modular_nodes = hierarchy['inter_modular_nodes']
    total_nodes = hierarchy['total_nodes']
    
    fig, ax = plt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = False)
    fig.suptitle('Population of network hierarchy, power-law construction\nnum_levels_hier = {:d}, num_nodes_0 = {:d}, gamma = {:5.2f}, num_mod_H = {:d}, \nTotal nodes = {:5.2e}'.format(num_levels_hier,num_nodes_0,gamma,num_modules_list[-1].astype(int),total_nodes))
    
    ax[0,0].plot(h_vec,num_modules_list, '-o', color = colors['blue3'])
    ax[0,0].set_xlabel(r'Hierarchy Level')
    ax[0,0].set_ylabel(r'Num Modules')
    # ax[0].set_ylim([0,num_nodes_0*1.1])
    # ax[0].legend()
    
    ax[0,1].semilogy(h_vec,num_nodes_per_module, '-o', color = colors['blue3'])
    ax[0,1].set_xlabel(r'Hierarchy Level')
    ax[0,1].set_ylabel(r'Neurons per module at this level of hierarchy')
    
    ax[1,0].semilogy(h_vec,num_nodes_list, '-o', color = colors['blue3'])
    ax[1,0].set_xlabel(r'Hierarchy Level')
    ax[1,0].set_ylabel(r'Total neurons at this level of hierarchy')
    # ax[1].legend()
    
    ax[1,1].semilogy(h_vec,inter_modular_nodes, '-o', color = colors['blue3'])
    ax[1,1].set_xlabel(r'Hierarchy Level')
    ax[1,1].set_ylabel(r'Number of inter-modular neurons at this level of hierarchy')
    
    plt.show()

    return


def plot_out_degree_distribution(out_degree_distribution,num_bins):
    
    functional_form = out_degree_distribution['functional_form']
    num_nodes = out_degree_distribution['num_nodes']
    node_degrees = out_degree_distribution['node_degrees']
    tot_edges = np.sum(out_degree_distribution['node_degrees'])
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False)
    node_index_vec = np.linspace(1,num_nodes,num_nodes)
    if functional_form == 'gaussian':
        fig.suptitle('gaussian out-degree distribution\ncenter = {:5.2f}, standard deviation = {:5.2f}\ntotal edges = {:5.2f}, average out-degree = {:5.2f}'.format(out_degree_distribution['center'],out_degree_distribution['st_dev'],tot_edges,tot_edges/num_nodes))
        ax[0].plot(node_index_vec,node_degrees, '-', color = colors['blue3'])
        
    if functional_form == 'power_law':
        fig.suptitle('power-law out-degree distribution\ngenerating k_out_min = {:5.2f}, generating alpha = {:5.2f}\ntotal edges = {:5.2f}, average out-degree = {:5.2f}'.format(out_degree_distribution['k_out_min'],out_degree_distribution['alpha'],tot_edges,tot_edges/num_nodes))
        ax[0].semilogy(node_index_vec,node_degrees, '-', color = colors['blue3'])
            
    ax[0].set_xlabel(r'Node Index')
    ax[0].set_ylabel(r'Node Degree')
    
    if functional_form == 'gaussian':
        
        degree_hist, bin_edges = np.histogram(node_degrees,num_bins)
        bin_centers = bin_edges[0:-1]+np.diff(bin_edges)/2
        ax[1].plot(bin_centers,degree_hist, '-o', color = colors['blue3'])
        ax[1].set_xlabel(r'bin center value')
        ax[1].set_ylabel(r'bin occupation fraction')
        
    elif functional_form == 'power_law':
        
        k_out_min = out_degree_distribution['k_out_min']
        alpha = out_degree_distribution['alpha']
        
        fit = powerlaw.Fit(node_degrees)        
        
        bin_min, bin_max = np.min(node_degrees), np.max(node_degrees)
        bins = 10**(np.linspace(np.log10(bin_min), np.log10(bin_max), num_bins))
        node_degrees_hist, bin_edges = np.histogram(node_degrees, bins, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.
        
        ax[1].loglog(bin_centers,node_degrees_hist, 'o', color = colors['blue3'], label = 'generated data')
    
        bins_dense = np.linspace(bin_centers[0], bin_centers[-1], 1000)
        k_out_min = fit.power_law.xmin
        alpha = fit.power_law.alpha
        ax[1].loglog(bins_dense, [(alpha-1)*k_out_min**(alpha-1)*x**(-alpha) for x in bins_dense], color = colors['red3'], label = 'fit, k_out_min = {:5.2f}, alpha = {:5.2f}'.format(k_out_min,alpha))
        
        ax[1].set_xlabel(r'bin center value')
        ax[1].set_ylabel(r'bin frequency')
        ax[1].legend()
    
    plt.show()

    return

def plot_node_degree_vs_space():
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)
    fig.suptitle('x-y positions of nodes')
    
    degree_vec = np.linspace(1,num_nodes,num_nodes)
    ax.plot(node_x_coords,node_y_coords, '-o', color = colors['blue3'])
    ax.plot(node_x_coords[0],node_y_coords[0], '-o', color = colors['green3'], label = 'first')
    ax.plot(node_x_coords[1],node_y_coords[1], '-o', color = colors['yellow3'], label = 'second')
    ax.plot(node_x_coords[-1],node_y_coords[-1], '-o', color = colors['red3'], label = 'last')
    ax.set_xlabel(r'x coord')
    ax.set_ylabel(r'y coord')
    ax.set_xlim([-1,num_row_col])
    ax.set_ylim([-1,num_row_col])
    ax.legend()
        
    degree__mn = np.zeros([num_row_col,num_row_col])
    mm_ii = node_x_coords
    nn_ii = node_y_coords
    for ii in range(num_nodes):
        # print('mm_ii = {}, nn_ii = {}'.format(mm_ii,nn_ii))
        degree__mn[mm_ii[ii],nn_ii[ii]] = node_degrees[ii]
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)
    
    degree = ax.imshow(np.transpose(degree__mn[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[0,num_row_col-1,0,num_row_col-1], aspect = 'auto', origin = 'lower')
    cbar = fig.colorbar(degree, extend='both')
    cbar.minorticks_on()     
    fig.suptitle('designed node degrees vs x-y positions')
    ax.set_xlabel(r'x coord')
    ax.set_ylabel(r'y coord')   
    plt.show()
    
    return


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

