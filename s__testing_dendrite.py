#%%
import numpy as np
from matplotlib import pyplot as plt

from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_receiving_loop_current, plot_rate_vs_num_active_synapses, plot_synaptic_integration_loop_current, plot_neuronal_response__single_synaptic_pulse, plot_burst_size_vs_num_active_synapses

plt.close('all')

#%% temporal
num_tau_sim = 0
observation_duration = 4e-6
dt = 5e-9

I_sy = 37e-6
I_th_d = 35e-6
I_th_n = 35e-6
tau_si = 1000e-9
tau_di = 1000e-9
tau_ref = 25e-9
jitter_params = [0,25e-9]#[gaussian center, gaussian deviation]

# num_spikes_out_mat = np.zeros([len(I_sy_vec),num_synapses_tot])

#%% run it
            
# initialize input signal
input_1 = input_signal('input_signal', input_temporal_form = 'single_spike', spike_times = np.array([100e-9]),
                       stochasticity = 'gaussian', jitter_params = jitter_params)   

# initialize synapse
synapse_1 = synapse('input_synapse', inhibitory_or_excitatory = 'excitatory',
                    loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si, 
                    integration_loop_self_inductance = 20e-9, integration_loop_output_inductance = 200e-12, 
                    synaptic_bias_current = I_sy, integration_loop_bias_current = 31e-6,
                    input_signal_name = 'input_signal')
            
# initialize dendrite
dendrite_1 = dendrite('intermediate_dendrite', inhibitory_or_excitatory = 'excitatory', receiving_loop_self_inductance = 20e-9, 
                      input_synaptic_connections = ['input_synapse'], input_synaptic_inductances = [[10e-12,0.5]], 
                      input_dendritic_connections = [], input_dendritic_inductances = [],                      
                      thresholding_junction_critical_current = 40e-6, thresholding_junction_bias_current = I_th_d,
                      integration_loop_self_inductance = 50e-9, integration_loop_output_inductance = 200e-12,
                      integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
                      integration_loop_bias_current = 31e-6)
        
# initialize neuron
neuron_1 = neuron('output_neuron', input_dendritic_connections = 'intermediate_dendrite', input_dendritic_inductances = [[10e-12,0.5],[10e-12,0.5]],
                  thresholding_junction_critical_current = 40e-6, thresholding_junction_bias_current = I_th_n, 
                  refractory_temporal_form = 'exponential', refractory_time_constant = tau_ref, 
                  refractory_loop_self_inductance = 1e-9, refractory_loop_output_inductance = 200e-12,
                  refractory_loop_synaptic_bias_current = 39e-6, refractory_loop_saturation_current = 50e-6)
               
# propagate in time                         
sim_params = dict()
sim_params['dt'] = dt
sim_params['pre_observation_duration'] = num_tau_sim*synapse_1.time_constant
sim_params['observation_duration'] = observation_duration
sim_params['num_tau_sim'] = num_tau_sim
neuron_1.sim_params = sim_params
neuron_1.run_sim()

# plot temporal response
plot_save_string = 'testing'#'I_sy={:2.2f}uA__tau_ref={:04.2f}ns__tau_si={:04.2f}ns__I_th={:2.2f}uA__dt={:04.2f}ns__obs={:04.2f}us__jitter={}ns'.format(1e6*I_sy_vec[kk],1e9*tau_ref,1e9*tau_si,I_th,1e9*dt,observation_duration*1e6,jitter_params[1])
plot_receiving_loop_current(neuron_1,plot_save_string)
plt.close('all')

# fill observation matrices                
num_spikes_out_mat[kk,rr] = neuron_1.num_spikes
  
#save neuron data
data_save_string = plot_save_string
neuron_1.save_neuron_data(data_save_string)

  
#%% plot
neuron_1.num_active_synapses_vec = np.arange(1,num_synapses_tot+1,1)
neuron_1.tau_si = tau_si
neuron_1.tau_ref = tau_ref
neuron_1.I_sy_vec = I_sy_vec
neuron_1.num_synapses_tot = num_synapses_tot
neuron_1.num_spikes_out_mat = num_spikes_out_mat
plot_save_string = 'I_sy={:2.2f}-{:2.2f}uA__tau_ref={:04.2f}ns__tau_si={:04.2f}ns__I_th={:2.2f}uA__dt={:04.2f}ns__obs={:04.2f}us__jitter={}ns'.format(1e6*I_sy_vec[0],1e6*I_sy_vec[-1],1e9*tau_ref,1e9*tau_si,I_th,1e9*dt,observation_duration*1e6,jitter_params[1])
plot_burst_size_vs_num_active_synapses(neuron_1,plot_save_string)
# neuron_1.plot_rate_transfer_function__no_lines(plot_save_string = plot_save_string)


