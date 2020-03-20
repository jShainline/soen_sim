#%%
import numpy as np
from matplotlib import pyplot as plt
from pylab import *

from soen_sim import synapse, neuron
#%% temporal
t0 = 0
tf = 10e-6
dt = 10e-9
time_vec = np.arange(t0,tf+dt,dt)

tau_si_vec = [100e-9,200e-9,300e-9]
rate_vec = np.linspace(100e3,20e6,20)
num_spikes_mat = np.zeros([len(rate_vec),len(tau_si_vec)])
isi_vec = 1/rate_vec
observation_time_start = 5e-6
observation_time_end = tf
for ii in range(len(tau_si_vec)):
    print('\n\nii = {} of {}'.format(ii+1,len(tau_si_vec)))
    
    for jj in range(len(rate_vec)):
        print('jj = {} of {}'.format(jj+1,len(rate_vec)))
                
        
        # initialize synapse
        colloquial_name__s = 'input_synapse__{}'.format(ii*len(rate_vec)+jj)
        synapse_1 = synapse(colloquial_name__s, loop_temporal_form = 'exponential', time_constant = tau_si_vec[ii], 
                            integration_loop_self_inductance = 50e-9, integration_loop_output_inductance = 200e-12, 
                            synaptic_bias_current = 37e-6, loop_bias_current = 31e-6)
        # print('synapse time constant = {}'.format(synapse_1.time_constant))
        
        #initialize neuron
        colloquial_name__n = 'rate_encoding_neuron__{}'.format(ii*len(rate_vec)+jj)
        neuron_1 = neuron(colloquial_name__n, input_connections = {colloquial_name__s}, input_inductances = [[10e-12,0.5],[10e-12,0.5]],
                          thresholding_junction_critical_current = 40e-6, threshold_bias_current = 35e-6, 
                          refractory_temporal_form = 'exponential', refractory_loop_self_inductance = 1e-9, refractory_loop_output_inductance = 200e-12)
               
        #propagate in time        
        synapse_1.input_spike_times = np.arange(20e-9,tf+dt,isi_vec[jj])  
        # synapse_1.input_spike_times = [100e-9,150e-9,300e-9,350e-9,700e-9,730e-9] 
        neuron_1.run_sim(time_vec)
        # neuron_1.plot_receiving_loop_current(time_vec)
        
        #calculate num_spikes in last half of simulation
        idx_obs_start = (np.abs(time_vec-observation_time_start)).argmin()
        idx_obs_end = (np.abs(time_vec-observation_time_end)).argmin()
        num_spikes_mat[jj,ii] = sum(neuron_1.spike_vec[idx_obs_start:idx_obs_end+1])        
        
        #delete synapses and neurons
        # del synapse_1, neuron_1
        
#%% plot
title_font_size = 20
axes_labels_font_size = 16
tick_labels_font_size = 12
plt.rcParams['figure.figsize'] = (20,16)
#nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('Output rate versus input rate', fontsize = title_font_size)

#upper panel, total I_nr
for ii in range(len(tau_si_vec)):
    ax.plot(rate_vec*1e-6,1e-3*num_spikes_mat[:,ii]/(observation_time_end-observation_time_start), 'o-', linewidth = 1, markersize = 3, label = 'tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
ax.set_xlabel(r'Input rate [MHz]', fontsize = axes_labels_font_size)
ax.set_ylabel(r'Output rate [kHz]', fontsize = axes_labels_font_size)
ax.tick_params(axis='both', which='major', labelsize = tick_labels_font_size)
# ax.set_title('Total current in NR loop')
ax.legend(loc = 'best')
ax.grid(b = True, which='major', axis='both')

plt.show()
# save_str = 'I_nr__'+neuron_1.colloquial_name        
# fig.savefig(save_str)

#%%
# input_spike_times_1 = np.arange(20e-9,tf+dt,isi_vec[0])  
# input_spike_times_2 = np.arange(20e-9,tf+dt,isi_vec[1])
# t0 = 0
# tf = 10e-6
# dt = 10e-9
# time_vec = np.arange(t0,tf+dt,dt)
# observation_time_start = 5e-6
# observation_time_end = 10e-6
# idx_obs_start = (np.abs(time_vec-observation_time_start)).argmin()
# idx_obs_end = (np.abs(time_vec-observation_time_end)).argmin()
# print(time_vec[idx_obs_start])
# print(time_vec[idx_obs_end])
# time_vec_partial = time_vec[idx_obs_start:idx_obs_end+1]

