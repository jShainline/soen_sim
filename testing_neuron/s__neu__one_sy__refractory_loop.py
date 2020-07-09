import numpy as np
from matplotlib import pyplot as plt
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__dend_drive_and_response
from _functions import read_wr_data, chi_squared_error, dendritic_drive__piecewise_linear, dendritic_drive__exp_pls_train__LR, dendritic_drive__square_pulse_train
from soen_sim import input_signal, synapse, dendrite, neuron

plt.close('all')

#%% sim params

dt = 1e-9
tf = 2e-9

# create sim_params dictionary
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf
sim_params['synapse_model'] = 'lookup_table'


#%% synapse
I_spd = 20e-6

spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]    
I_sy_vec = [23e-6,28e-6,33e-6,38e-6,29e-6,29e-6,29e-6,29e-6,32e-6,32e-6,32e-6,32e-6]
L_si_vec = [77.5e-9,77.5e-9,77.5e-9,77.5e-9,7.75e-9,77.5e-9,775e-9,7.75e-6,775e-9,775e-9,775e-9,775e-9]
tau_si_vec = [250e-9,250e-9,250e-9,250e-9,250e-9,250e-9,250e-9,250e-9,10e-9,50e-9,250e-9,1.25e-6]

# initialize input signal
input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
    
# initialize synapse
ii = 0    
synapse_1 = synapse('sy', num_jjs = 3, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si_vec[ii], 
                    integration_loop_self_inductance = L_si_vec[ii], integration_loop_output_inductance = 200e-12, 
                    synaptic_bias_currents = [I_spd,I_sy_vec[ii],36e-6,35e-6],
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
neuron_1 = neuron('ne', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                  input_synaptic_connections = ['sy'], 
                  input_synaptic_inductances = [[20e-12,1]],                     
                  thresholding_junction_critical_current = 40e-6,
                  bias_currents = [71.5e-6,36e-6,35e-6],
                  integration_loop_self_inductance = 775e-12, 
                  integration_loop_output_inductances = [[400e-12,1],[200e-12,1]],
                  integration_loop_temporal_form = 'exponential',
                  integration_loop_time_constant = 5e-9,
                  refractory_temporal_form = 'exponential',
                  refractory_time_constant = 10e-9,
                  refractory_thresholding_junction_critical_current = 40e-6,
                  refractory_loop_self_inductance = 775e-12,
                  refractory_loop_output_inductance = 100e-12,
                  refractory_loop_circuit_inductances = [20e-12,20e-12,200e-12,77.5e-12],
                  refractory_bias_currents = [71.5e-6,36e-6,35e-6],
                  neuronal_receiving_input_refractory_inductance = [20e-12,1],
                  time_params = time_params)
              
neuron_1.run_sim()

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


