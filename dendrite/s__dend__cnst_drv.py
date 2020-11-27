import numpy as np
from matplotlib import pyplot as plt
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_dendritic_integration_loop_current
from _functions import read_wr_data, chi_squared_error, dendritic_drive__piecewise_linear, dendritic_drive__exp_pls_train__LR, dendritic_drive__square_pulse_train
from soen_sim import input_signal, dendrite, neuron #synapse, 

plt.close('all')

#%% units

# all currents in uA
# all inductances in pH
# all times in ns
# all resistances in pH/ns ( aka milliohms )

            
#%% linear ramp

# plt.close('all')

dt = 0.1
tf = 500

num_jjs = 4
I_drive_vec = np.asarray([14,14])
L_di_vec = np.asarray([77.5])*1e3
tau_di_vec = np.asarray([77.5e6]) # 


for kk in range(len(tau_di_vec)):
    for jj in range(len(L_di_vec)):
        L_di = L_di_vec[jj]
        if kk == 0:
            tau_di = L_di/1e-3
        else:
            tau_di = tau_di_vec[kk]
        
        print('\n\njj = {} of {}; kk = {} of {}'.format(jj+1,len(L_di_vec),kk+1,len(tau_di_vec)))
        
        # load WR data


        # setup soen sim for linear ramp            
        pwl_drive = [[0,0],[1,0],[2,I_drive_vec[0]],[42,I_drive_vec[1]]]
        input_1 = input_signal(name = 'input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200, 
                                time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)            

        dendrite_1 = dendrite(name = 'dendrite_under_test', num_jjs = num_jjs,
                                inhibitory_or_excitatory = 'excitatory', circuit_inductances = [0,20,200,77.5], 
                                input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[20,1]],
                                thresholding_junction_critical_current = 40, bias_currents = [72,36,35],
                                integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0,
                                integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di)

        time_params = dict()
        time_params['dt'] = dt
        time_params['tf'] = tf
        neuron_1 = neuron(name = 'dummy_neuron', input_dendritic_connections = ['dendrite_under_test'], 
                          circuit_inductances = [0,0,200,77.5],
                          input_dendritic_inductances = [[20,1]], 
                          refractory_loop_circuit_inductances = [0,2,200,77.5],
                          refractory_time_constant = 50,
                          refractory_thresholding_junction_critical_current = 40,
                          refractory_loop_self_inductance =775,
                          refractory_loop_output_inductance = 100,
                          refractory_bias_currents = [74,36,35],
                          refractory_receiving_input_inductance = [20,1],
                          neuronal_receiving_input_refractory_inductance = [20,1],
                          time_params = time_params)           
        
        neuron_1.run_sim()
        dendrite_1.time_vec = neuron_1.time_vec
        dendrite_1.I_di = neuron_1.dendrites['dendrite_under_test'].I_di_vec
                                        
        plot_dendritic_integration_loop_current(dendrite_1)            


