#%%
import numpy as np
# import copy
# import time

from soen_sim import synapse, neuron

#%% set up synapses

synapse_0 = synapse()

synapse_1 = synapse('test_synapse__exp', loop_temporal_form = 'exponential', time_constant = 200e-9, 
                    integration_loop_self_inductance = 50e-9, integration_loop_output_inductance = 200e-12, 
                    synaptic_bias_current = 36e-6, loop_bias_current = 31e-6)

synapse_2 = synapse('test_synapse__power_law', loop_temporal_form = 'power_law', power_law_exponent = -1.1, 
                    integration_loop_self_inductance = 100e-9, integration_loop_output_inductance = 200e-12,
                    synaptic_bias_current = 37e-6, loop_bias_current = 33e-6)

# for obj in synapse.get_instances():
#     print(obj.colloquial_name)

#%% propagate in time
# t0 = 0
# tf = 1e-6
# dt = 10e-9
# time_vec = np.arange(t0,tf+dt,dt)

# synapse_1.input_spike_times = [100e-9]
# synapse_1.run_sim(time_vec)
# synapse_1.plot_integration_loop_current(time_vec)

# rep_rate = 1/100e-9
# synapse_1.input_spike_times = np.arange(20e-9,1e-6,1/rep_rate)
# synapse_1.run_sim(time_vec)
# synapse_1.plot_integration_loop_current(time_vec)

# synapse_1.synaptic_bias_current = 35e-6
# synapse_1.run_sim(time_vec)
# synapse_1.plot_integration_loop_current(time_vec)

# synapse_1.synaptic_bias_current = 39e-6
# synapse_1.run_sim(time_vec)
# synapse_1.plot_integration_loop_current(time_vec)

#%% set up neuron

neuron_1 = neuron('test_neuron', input_connections = {'s0','s1'}, input_inductances = [[10e-12,0.5],[10e-12,0.5]],
                  thresholding_junction_critical_current = 40e-6, threshold_bias_current = 35e-6, 
                  refractory_temporal_form = 'exponential', )

# for obj in neuron.get_instances():
#     print(obj.colloquial_name)
    
# for obj in synapse.get_instances():
#     # print(obj.colloquial_name)
#     if obj.unique_label in neuron_1.input_connections:
#         print(obj.unique_label+'; '+obj.colloquial_name)

#%% propagate in time
t0 = 0
tf = 1e-6
dt = 10e-9
time_vec = np.arange(t0,tf+dt,dt)     
   
synapse_0.input_spike_times = [100e-9,200e-9]
synapse_1.input_spike_times = [220e-9,500e-9]
neuron_1.run_sim(time_vec)
# print(neuron_1.receiving_loop_self_inductance)
# print(neuron_1.receiving_loop_total_inductance)

neuron_1.plot_receiving_loop_current(time_vec)

#%%



