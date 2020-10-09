import numpy as np
from matplotlib import pyplot as plt
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_neuronal_response
from _functions import read_wr_data, chi_squared_error, dendritic_drive__piecewise_linear, dendritic_drive__exp_pls_train__LR, dendritic_drive__square_pulse_train
from soen_sim import input_signal, synapse, dendrite, neuron

# plt.close('all')

#%% sim params

dt = 100e-12
tf = 1e-6

# create sim_params dictionary
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf
sim_params['synapse_model'] = 'lookup_table'


#%% synapse
I_spd = 20e-6

# spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]    
# I_sy = 33e-6
# L_si = 77.5e-9
# tau_si = 250e-9

spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]    
I_sy = 33e-6
L_si = 775e-9
tau_si = 500e-9

# initialize input signal
input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
    
# initialize synapse
synapse_1 = synapse('sy', num_jjs = 3, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si, 
                    integration_loop_self_inductance = L_si, integration_loop_output_inductance = 400e-12, 
                    synaptic_bias_currents = [I_spd,I_sy,36e-6,35e-6],
                    input_signal_name = 'in', synapse_model_params = sim_params)

# synapse_1.run_sim() 

# actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
# actual_drive_array.append(actual_drive)
# actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:])) 
# sf_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_sf[:]))
# actual_data_array.append(actual_data)

#%% neuron

time_params = dict()
time_params['dt'] = dt
time_params['tf'] = tf
neuron_1 = neuron('ne', num_jjs = 2,
                  circuit_inductances = [10e-12,0e-12,200e-12,77.5e-12],
                  input_synaptic_connections = ['sy'], 
                  input_synaptic_inductances = [[10e-12,1]],                     
                  thresholding_junction_critical_current = 40e-6,
                  bias_currents = [76e-6,36e-6,35e-6],
                  integration_loop_self_inductance = 775e-12, 
                  integration_loop_output_inductances = [[400e-12,1],[300e-12,1]], # first is to drive latching JJ, second is to drive refractory dendrite; both are of the form [L_self,k]
                  integration_loop_temporal_form = 'exponential',
                  integration_loop_time_constant = 5e-9,
                  refractory_temporal_form = 'exponential',
                  refractory_time_constant = 10e-9,
                  refractory_thresholding_junction_critical_current = 40e-6,
                  refractory_loop_self_inductance = 775e-12,
                  refractory_loop_output_inductance = 200e-12,
                  refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                  refractory_bias_currents = [72e-6,36e-6,35e-6],
                  neuronal_receiving_input_refractory_inductance = [20e-12,1],
                  homeostatic_temporal_form = 'exponential',
                  homeostatic_time_constant = 10e-9,
                  homeostatic_thresholding_junction_critical_current = 40e-6,
                  homeostatic_loop_self_inductance = 775e-12,
                  homeostatic_loop_output_inductance = 200e-12,
                  homeostatic_loop_circuit_inductances = [20e-12,20e-12,200e-12,77.5e-12],
                  homeostatic_bias_currents = [71.5e-6,36e-6,35e-6],
                  neuronal_receiving_input_homeostatic_inductance = [20e-12,1],
                  time_params = time_params)
              
neuron_1.run_sim()

#%% plot
plot_neuronal_response(neuron_1)

#%% dendrite
# setup soen sim for exp pulse seq


# L_di = 775e-9
# tau_di = 10e-9

# dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
#                                       input_synaptic_connections = ['sy'], input_synaptic_inductances = [[]], 
#                                       input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
#                                       input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
#                                       thresholding_junction_critical_current = 40e-6, bias_currents = [71.5e-6,36e-6,35e-6],
#                                       integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
#                                       integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
#                                       dendrite_model_params = sim_params)

# dendrite_1.run_sim()
                                
# actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))    
# error__signal = chi_squared_error(target_data,actual_data)

# plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)

#%% 


