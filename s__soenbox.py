#%%
import numpy as np
# import copy
# import time

# from f__physical_constants import physical_constants

from soen_sim import synapse

#%% set up synapses

synapse_1 = synapse()

synapse_2 = synapse('test_synapse__exp', loop_temporal_form = 'exponential', 
                    time_constant = 200e-9, integration_loop_inductance = 10e-9, 
                    synaptic_bias_current = 34e-6, loop_bias_current = 31e-6)

synapse_3 = synapse('test_synapse__power_law', loop_temporal_form = 'power_law', 
                    power_law_exponent = -1.1, integration_loop_inductance = 100e-9, 
                    synaptic_bias_current = 37e-6, loop_bias_current = 33e-6)

#%% propagate in time
t0 = 0
tf = 1e-6
dt = 10e-9
time_vec = np.arange(t0,tf+dt,dt)

input_spike_times = [100e-9]
synapse_2.run_sim(time_vec, input_spike_times)
synapse_2.plot_time_trace(time_vec, input_spike_times)

rep_rate = 1/100e-9
input_spike_times = np.arange(20e-9,1e-6,1/rep_rate)
synapse_2.run_sim(time_vec, input_spike_times)
synapse_2.plot_time_trace(time_vec, input_spike_times)

synapse_2.synaptic_bias_current = 35e-6
synapse_2.run_sim(time_vec, input_spike_times)
synapse_2.plot_time_trace(time_vec, input_spike_times)

synapse_2.synaptic_bias_current = 39e-6
synapse_2.run_sim(time_vec, input_spike_times)
synapse_2.plot_time_trace(time_vec, input_spike_times)
