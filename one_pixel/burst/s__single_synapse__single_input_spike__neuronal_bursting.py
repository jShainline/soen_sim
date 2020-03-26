#%%
import numpy as np
from matplotlib import pyplot as plt
from pylab import *

from soen_sim import input_signal, synapse, neuron
from _functions import load_neuron_data
from _plotting import plot_receiving_loop_current, plot_rate_vs_num_active_synapses, plot_synaptic_integration_loop_current, plot_neuronal_response__single_synaptic_pulse

plt.close('all')

#%% temporal
num_tau_sim = 0
observation_duration = 4e-6
dt = 5e-9

tau_ref_vec = [25e-9,50e-9,100e-9]#[100e-9]#
tau_si_vec = [200e-9,500e-9,1000e-9,2000e-9]#np.linspace(200e-9,1000e-9,5)#[3000e-9]#
I_sy = 37e-6
I_th = 35e-6

num_spikes_out_mat = np.zeros([len(tau_ref_vec),len(tau_si_vec)])
isi_output_mat = 1e9*np.ones([len(tau_ref_vec),len(tau_si_vec)])
isi_output_avg_mat = 1e9*np.ones([len(tau_ref_vec),len(tau_si_vec)])

#%% run it
num_id = 0   
for ii in range(len(tau_ref_vec)):
    print('\n\nii = {} of {} (tau_ref)\n'.format(ii+1,len(tau_ref_vec)))
    tau_ref = tau_ref_vec[ii]
        
    for jj in range(len(tau_si_vec)):
        print('jj = {} of {} (tau_si)'.format(jj+1,len(tau_si_vec)))
        tau_si = tau_si_vec[jj]
        
        num_id += 1
                   
        # initialize input signal
        name__i = 'input_signal__{:d}_{:d}'.format(ii,jj)
        input_1 = input_signal(name__i, input_temporal_form = 'single_spike', spike_times = np.array([5e-9]),
        stochasticity = 'none')   
        # print(input_1.spike_times)
        
        # initialize synapses
        name__s = 'input_synapse__{:d}_{:d}'.format(ii,jj)
        synapse_1 = synapse(name__s, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si_vec[jj], 
                            integration_loop_self_inductance = 10e-9, integration_loop_output_inductance = 200e-12, 
                            synaptic_bias_current = I_sy, integration_loop_bias_current = 31e-6,
                            input_signal_name = name__i)
    
        #initialize neuron
        name__n = 'bursting_neuron__{:d}_{:d}'.format(ii,jj)
        neuron_1 = neuron(name__n, input_synaptic_connections = [name__s], input_synaptic_inductances = [[10e-12,0.5],[10e-12,0.5]],
                          thresholding_junction_critical_current = 40e-6, thresholding_junction_bias_current = I_th, 
                          refractory_temporal_form = 'exponential', refractory_time_constant = tau_ref_vec[ii], 
                          refractory_loop_self_inductance = 1e-9, refractory_loop_output_inductance = 200e-12,
                          refractory_loop_synaptic_bias_current = 39e-6, refractory_loop_saturation_current = 50e-6)
               
        # propagate in time                         
        sim_params = dict()
        sim_params['dt'] = dt
        sim_params['pre_observation_duration'] = 0
        sim_params['observation_duration'] = observation_duration
        neuron_1.sim_params = sim_params
        neuron_1.run_sim()
        
        # plot temporal response
        plot_save_string = 'I_th={:2.2f}uA__I_sy={:2.2f}uA__tau_ref={:04.2f}ns__tau_si={:04.2f}ns__dt={:04.2f}ns__obs={:04.2f}us'.format(1e6*I_th,1e6*I_sy,1e9*tau_ref_vec[ii],1e9*tau_si_vec[jj],1e9*dt,observation_duration*1e6)
        plot_neuronal_response__single_synaptic_pulse(neuron_1,plot_save_string)
        plt.close('all')
        
        # fill observation matrix                
        num_spikes_out_mat[ii,jj] = neuron_1.num_spikes
          
        #save neuron data
        data_save_string = plot_save_string
        neuron_1.save_neuron_data(data_save_string)
        
        #delete synapses and neurons
        # del synapse_1, neuron_1

