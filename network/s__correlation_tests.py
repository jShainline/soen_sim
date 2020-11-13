import numpy as np

from _plotting_network import plot_network_spikes_raster

#%% notes

#%% time

dt = 1 # ns
tf = 1e5 # ns
time_vec = np.arange(0,tf+dt,dt)
nt = len(time_vec)
t_ref = 50

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
    
#%% correlation function 1: symmetrical spike time

C1 = np.zeros([num_neurons,num_neurons])
for ii in range(num_neurons):
    spike_times__i = neuron_spikes__times[ii]
    for jj in range(num_neurons):
        spike_times__j = neuron_spikes__times[jj]
        if jj != ii:
            for kk in range(len(spike_times__i)):
                _ind = np.abs( spike_times__i[kk] - spike_times__j[:] ).argmin()
                C1[ii,jj] += ( np.abs(spike_times__i[kk] - spike_times__j[_ind]) + t_ref )**(-1)
            
        
    

