import numpy as np

from _plotting_network import plot_network_spikes_raster, plot_network_spikes_binned, plot_network_spikes_binned__mark_avalanches, plot_neuronal_avalanche_histograms, plot_neuronal_avalanche_histograms__with_fits

#%% notes

# this code produces random spike-train sequences for a given number of neurons. 
# the activity is summed and binned to give total network spikes in each time bin. 
# empty bins are used to define the beginning and end of each neuronal avalanche.
# the number of spikes and the duration of each avalanche are calculated.
# these quantities are histogrammed, and these histograms are fit to a power-law functional form.
# the plotting functions are in soen_sim/_plotting_network.py

#%% time vector

dt = 1 # ns
tf = 1e5 # ns
time_vec = np.arange(0,tf+dt,dt)
nt = len(time_vec)

#%% avalanche binning

Dt_bin = 50 # ns
Ndt_bin = np.round(Dt_bin/dt, decimals = 0).astype(int)
N_bins = np.round(tf/Dt_bin).astype(int)

#%% make pretend time series of spike events for each neuron

num_neurons = 50
tau_refractory = 2000 # ns
max_spikes = np.floor(tf/tau_refractory)
num_spikes__array = np.random.randint(0,max_spikes+1,size=[num_neurons]) # just to get going, each neuron has a random number of spikes in the time interval
neuron_spikes__raster = np.zeros([num_neurons,nt])
for ii in range(num_neurons):
    num_spikes = num_spikes__array[ii]
    for jj in range(num_spikes):
        spike_times = np.random.randint(0,tf+dt,size=[num_spikes])
    neuron_spikes__raster[ii,spike_times] = 1
    
plot_network_spikes_raster(neuron_spikes__raster)

#%% bin spike activity

temp_vec = np.sum(neuron_spikes__raster, axis = 0)
network_spikes__binned = np.zeros([N_bins-1])
for ii in range(N_bins-1):
    network_spikes__binned[ii] = np.sum(temp_vec[ii*Ndt_bin:(ii+1)*Ndt_bin])
plot_network_spikes_binned(network_spikes__binned)

#%% find size and duration of neuronal avalanches

network_spikes__zeros = np.where( network_spikes__binned == 0 )[0] # finds zeros in binned network spikes

avalanche_start_time_bin_indices = []
avalanche_stop_time_bin_indices = []
for ii in range(len(network_spikes__zeros)-1):
    if network_spikes__zeros[ii+1]-network_spikes__zeros[ii] > 1:
        avalanche_start_time_bin_indices.append(network_spikes__zeros[ii])
        avalanche_stop_time_bin_indices.append(network_spikes__zeros[ii+1])

num_avalanches = len(avalanche_start_time_bin_indices)
size_avalanches = np.zeros([num_avalanches])
duration_avalanches = np.zeros([num_avalanches])
for ii in range(num_avalanches):
    size_avalanches[ii] = np.sum(network_spikes__binned[avalanche_start_time_bin_indices[ii]:avalanche_stop_time_bin_indices[ii]])
    duration_avalanches[ii] = Ndt_bin*dt*(avalanche_stop_time_bin_indices[ii]-avalanche_start_time_bin_indices[ii])

plot_network_spikes_binned__mark_avalanches(network_spikes__binned,avalanche_start_time_bin_indices,avalanche_stop_time_bin_indices)

#%% histogram size and duration of neuronal avalanches
num_bins__size = 10
num_bins__duration = 10
[size_avalanches__histogram, size_avalanches__bin_edges] = np.histogram(size_avalanches, bins = num_bins__size)
[duration_avalanches__histogram, duration_avalanches__bin_edges] = np.histogram(duration_avalanches, bins = num_bins__duration)

plot_neuronal_avalanche_histograms(size_avalanches__histogram, size_avalanches__bin_edges, duration_avalanches__histogram, duration_avalanches__bin_edges)

#%% fit neuronal avalanches to functional forms

size_avalanches__bin_centers = size_avalanches__bin_edges[0:-1] + np.diff(size_avalanches__bin_edges)/2

size_gt_zero = np.where( size_avalanches__histogram > 0 )[0]
size_p, size_residuals, _, _, _ = np.polyfit(np.log10(size_avalanches__bin_centers[size_gt_zero]), np.log10(size_avalanches__histogram[size_gt_zero]), 1, full = True)
size_power = size_p[0]
size_coeff = 10**(size_p[1])

size_vec__dense = np.linspace(size_avalanches__bin_edges[0],size_avalanches__bin_edges[-1],100)
size_fit = size_coeff*size_vec__dense**size_power

duration_avalanches__bin_centers = duration_avalanches__bin_edges[0:-1] + np.diff(duration_avalanches__bin_edges)/2

duration_gt_zero = np.where( duration_avalanches__histogram > 0 )[0]
duration_p, duration_residuals, _, _, _ = np.polyfit(np.log10(duration_avalanches__bin_centers[duration_gt_zero]), np.log10(duration_avalanches__histogram[duration_gt_zero]), 1, full = True)
duration_power = duration_p[0]
duration_coeff = 10**(duration_p[1])

duration_vec__dense = np.linspace(duration_avalanches__bin_edges[0],duration_avalanches__bin_edges[-1],100)
duration_fit = duration_coeff*duration_vec__dense**duration_power

plot_neuronal_avalanche_histograms__with_fits(size_avalanches__histogram, size_avalanches__bin_edges, size_fit, size_vec__dense, size_power, size_residuals[0],
                                              duration_avalanches__histogram, duration_avalanches__bin_edges, duration_fit, duration_vec__dense, duration_power, duration_residuals[0])

