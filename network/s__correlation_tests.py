import numpy as np

from _plotting_network import plot_network_spikes_raster

#%% notes

#%% time vector

dt = 1 # ns
tf = 1e5 # ns
time_vec = np.arange(0,tf+dt,dt)
nt = len(time_vec)

#%% make pretend time series of spike events for each neuron

num_neurons = 10
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

#%% convert to lists of spike times

neuron_spikes__times = []
for ii in range(num_neurons):
    spike_indices = np.where(neuron_spikes__raster[ii,:] == 1)
    neuron_spikes__times.append(time_vec[spike_indices])
    

