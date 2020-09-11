import numpy as np
from matplotlib import pyplot as plt
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_neuronal_response, plot_phase_portrait
from _functions import read_wr_data, chi_squared_error, dendritic_drive__piecewise_linear, dendritic_drive__exp_pls_train__LR, dendritic_drive__square_pulse_train
from soen_sim import input_signal, synapse, dendrite, neuron

plt.close('all')

#%% sim params

num_jjs = 4

spike_times = [5e-9]    
I_sy = 75e-6
L_si = 77.5e-9
tau_si = 2.1e-6 # 250e-9 # 

dt = 100e-12
tf = 2e-6

#%% synapse

# spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]    
# I_sy = 33e-6
# L_si = 77.5e-9
# tau_si = 250e-9

# I_sy = 79e-6
# L_si = 77.5e-9
# tau_si = 2.1e-6

# initialize input signal
input_1 = input_signal(name = 'in', 
                        input_temporal_form = 'single_spike', # 'single_spike' or 'constant_rate' or 'arbitrary_spike_train'
                        spike_times = spike_times)            
        
# initialize synapse
synapse_1 = synapse(name = 'sy',
                    synaptic_circuit_inductors = [100e-9,100e-9,250e-12],
                    synaptic_circuit_resistors = [5e3,4.008],
                    synaptic_hotspot_duration = 200e-12,
                    synaptic_spd_current = 10e-6,
                    input_direct_connections = ['in'],
                    num_jjs = num_jjs,
                    inhibitory_or_excitatory = 'excitatory',
                    synaptic_dendrite_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                    synaptic_dendrite_input_synaptic_inductance = [20e-12,1],
                    junction_critical_current = 40e-6,
                    bias_currents = [I_sy, 36e-6, 35e-6],
                    integration_loop_self_inductance = L_si,
                    integration_loop_output_inductance = 200e-12,
                    integration_loop_time_constant = tau_si)

#%% neuron

neuron_1 = neuron(name = 'ne', num_jjs = 4,
                  input_synaptic_connections = ['sy'],
                  input_synaptic_inductances = [[20e-12,1]],
                  junction_critical_current = 40e-6,
                  bias_currents = [76e-6, 36e-6, 35e-6],
                  circuit_inductances = [0e-12,0e-12,200e-12,77.5e-12],  
                  refractory_dendrite_num_jjs = 4,
                  refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                  refractory_time_constant = 25e-9,
                  refractory_junction_critical_current = 40e-6,
                  refractory_loop_self_inductance = 200e-12,
                  refractory_loop_output_inductance = 100e-12,
                  refractory_bias_currents = [76e-6,36e-6,35e-6],
                  refractory_receiving_input_inductance = [20e-12,1],
                  neuronal_receiving_input_refractory_inductance = [20e-12,1],
                  integration_loop_self_inductance = 0,
                  integration_loop_time_constant = 25e-9,
                  integration_loop_output_inductances = [[0e-12,1],[100e-12,1]], # first is to drive latching JJ, second is to drive refractory dendrite; both are of the form [L_self,k]
                  time_params = dict([['dt',dt],['tf',tf]])) 
              
neuron_1.run_sim()

#%% plot
plot_neuronal_response(neuron_1)
# plot_phase_portrait(neuron_1)



