#%%
import numpy as np
from matplotlib import pyplot as plt
from pylab import *

from soen_sim import synapse, neuron
from _functions import plot_params

# plt.close('all')

#%% temporal
t0 = 0
num_tau_sim = 10
observation_duration = 200e-6
dt = 10e-9

tau_si_vec = np.linspace(250e-9,1000e-9,4)#[500e-9]#
rate_vec = np.logspace(np.log10(2e5),np.log10(2e7),500)#np.linspace(2e5,20e6,100)#np.array([1e6])#

num_spikes_mat = np.zeros([len(rate_vec),len(tau_si_vec)])
isi_output_mat = 1e9*np.ones([len(rate_vec),len(tau_si_vec)])
isi_output_avg_mat = 1e9*np.ones([len(rate_vec),len(tau_si_vec)])
isi_input_vec = 1/rate_vec
for ii in range(len(tau_si_vec)):
    print('\n\nii = {} of {}'.format(ii+1,len(tau_si_vec)))
    
    for jj in range(len(rate_vec)):
        print('jj = {} of {}'.format(jj+1,len(rate_vec)))
                
        
        # initialize synapse
        name__s = 'input_synapse__{}'.format(ii*len(rate_vec)+jj)
        synapse_1 = synapse(name__s, loop_temporal_form = 'exponential', time_constant = tau_si_vec[ii], 
                            integration_loop_self_inductance = 50e-9, integration_loop_output_inductance = 200e-12, 
                            synaptic_bias_current = 37e-6, loop_bias_current = 31e-6)
        # print('synapse time constant = {}'.format(synapse_1.time_constant))
        
        #initialize neuron
        name__n = 'rate_encoding_neuron__{}'.format(ii*len(rate_vec)+jj)
        neuron_1 = neuron(name__n, input_connections = {name__s}, input_inductances = [[10e-12,0.5],[10e-12,0.5]],
                          thresholding_junction_critical_current = 40e-6, threshold_bias_current = 35e-6, 
                          refractory_temporal_form = 'exponential', refractory_time_constant = 50e-9, 
                          refractory_loop_self_inductance = 10e-9, refractory_loop_output_inductance = 200e-12)
               
        #propagate in time         
        t1 = t0+num_tau_sim*synapse_1.time_constant
        tf = t1+observation_duration
        time_vec = np.arange(t0,tf+dt,dt)
        synapse_1.input_spike_times = np.arange(5e-9,tf+dt,isi_input_vec[jj])
        neuron_1.run_sim(time_vec)
        # neuron_1.plot_receiving_loop_current(time_vec)
        
        #calculate output rate in various ways by looking at spikes in observation_duration
        idx_obs_start = (np.abs(time_vec-t1)).argmin()
        idx_obs_end = (np.abs(time_vec-tf)).argmin()
        num_spikes_mat[jj,ii] = sum(neuron_1.spike_vec[idx_obs_start:idx_obs_end+1])
        if len(neuron_1.spike_times) > 1:
            idx_avg_start = (np.abs(np.asarray(neuron_1.spike_times)-t1)).argmin()
            idx_avg_end = (np.abs(np.asarray(neuron_1.spike_times)-tf)).argmin()            
            isi_output_mat[jj,ii] = neuron_1.spike_times[-1]-neuron_1.spike_times[-2]
            isi_output_avg_mat[jj,ii] = np.mean(neuron_1.inter_spike_intervals[idx_avg_start:idx_avg_end])
        # elif len(neuron_1.spike_times) > 1:
        #     isi_vec = diff(neuron_1.spike_times[:])
        #     if len(isi_vec) = 1:
        #         isi_output_mat[jj,ii] = isi_vec
        #     elif len(isi_vec)
                
        
        #delete synapses and neurons
        # del synapse_1, neuron_1
        
#%% plot
pp = plot_params()
plt.rcParams['figure.figsize'] = pp['fig_size']
save_str = 'rate_transfer_function__tau_si={}-{}ns__rate_in={}-{}MHz__obs_dur={}us__dt={}ns'.format(tau_si_vec[0]*1e9,tau_si_vec[-1]*1e9,rate_vec[0]*1e-6,rate_vec[-1]*1e-6,observation_duration*1e6,dt*1e9)

#nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('Output rate versus input rate', fontsize = pp['title_font_size'])

#upper panel, total I_nr
for ii in range(len(tau_si_vec)):
    # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = 1, markersize = 3, label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
    # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = 1, markersize = 3, label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
    ax.plot(rate_vec*1e-6,1e-6*1/isi_output_avg_mat[:,ii], 'o-', linewidth = 1, markersize = 3, label = 'mean firing rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
ax.set_xlabel(r'Input rate [MHz]', fontsize = pp['axes_labels_font_size'])
ax.set_ylabel(r'Output rate [MHz]', fontsize = pp['axes_labels_font_size'])
ax.tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
# ax.set_title('Total current in NR loop')
ax.legend(loc = 'best')
ax.grid(b = True, which='major', axis='both')
plt.show()


#         
# fig.savefig(save_str)

#%%
# print(1e-3*1/isi_output_mat[:,0])
# aa = [1,2,3,4,5]
# print(aa[2:])
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

