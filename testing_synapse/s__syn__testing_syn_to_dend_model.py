import numpy as np
from matplotlib import pyplot as plt

from _plotting import plot_wr_comparison__synapse
from _functions import read_wr_data, chi_squared_error
from soen_sim import input_signal, synapse, neuron # dendrite, 

# plt.close('all')

#%% single pulse
I_spd = 10e-6

dt = 0.01e-9
spike_times = [5e-9]

num_jjs = 4
if num_jjs == 2:
    I_de_vec = np.asarray([8.13,12.91])*1e-6
elif num_jjs == 4:
    I_de_vec = np.asarray([70])*1e-6
L_di_vec = np.asarray([7.75,77.5,775,7750])*1e-9 # ,775,7750
tau_di_vec = np.asarray([10,25,50,250,1250])*1e-9 # 10,25,50,

for ii in range(len(I_de_vec)):
    I_de = I_de_vec[ii]
    for jj in range(len(L_di_vec)):
        L_di = L_di_vec[jj]
        for kk in range(len(tau_di_vec)+1):
            if kk == 0:
                tau_di = L_di/1e-6
                temp_str = 'ms'
                temp_num = 1e3
            else:
                tau_di = tau_di_vec[kk-1]
                temp_str = 'ns'
                temp_num = 1e9
            
            print('\n\nii = {} of {}; jj = {} of {}; kk = {} of {}'.format(ii+1,len(I_de_vec),jj+1,len(L_di_vec),kk+1,len(tau_di_vec)+1))
            
            # load WR data
            directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
            file_name = 'syn_one_pls_{:d}jj_Ispd{:05.2f}uA_Ide{:05.2f}uA_Ldr20.0pH20.0pH_Ldi{:07.2f}nH_taudi{:07.2f}{}_dt01.0ps.dat'.format(num_jjs,I_spd*1e6,I_de*1e6,L_di*1e9,tau_di*temp_num,temp_str)
            data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
            if num_jjs == 2:
                I_di_str = 'L0#branch'
                I_drive_str = 'L2#branch'
            elif num_jjs == 4:
                I_di_str = 'L3#branch'
                I_drive_str = 'L0#branch'
            target_data = np.vstack((data_dict['time'],data_dict[I_di_str]))
            tf = np.round(data_dict['time'][-1]/1e-9)*1e-9
            target_data__drive = np.vstack((data_dict['time'],data_dict[I_drive_str]))

            # setup soen sim for exp pulse seq
            input_1 = input_signal(name = 'input_synaptic_drive', 
                                   input_temporal_form = 'single_spike', # 'single_spike' or 'constant_rate' or 'arbitrary_spike_train'
                                   spike_times = spike_times)            
        
            synapse_1 = synapse(name = 'synapse_under_test',
                                synaptic_circuit_inductors = [100e-9,100e-9,400e-12],
                                synaptic_circuit_resistors = [5e3,4.008],
                                synaptic_hotspot_duration = 200e-12,
                                synaptic_spd_current = 10e-6,
                                input_direct_connections = ['input_synaptic_drive'],
                                num_jjs = num_jjs,
                                inhibitory_or_excitatory = 'excitatory',
                                synaptic_dendrite_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                                synaptic_dendrite_input_synaptic_inductance = [20e-12,1],
                                junction_critical_current = 40e-6,
                                bias_currents = [I_de, 36e-6, 35e-6],
                                integration_loop_self_inductance = L_di,
                                integration_loop_output_inductance = 0e-12,
                                integration_loop_time_constant = tau_di)
       
            neuron_1 = neuron('dummy_neuron',
                              input_synaptic_connections = ['synapse_under_test'],
                              input_synaptic_inductances = [[20e-12,1]],
                              junction_critical_current = 40e-6,
                              circuit_inductances = [0e-12,0e-12,200e-12,77.5e-12],                              
                              refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                              refractory_time_constant = 50e-9,
                              refractory_thresholding_junction_critical_current = 40e-6,
                              refractory_loop_self_inductance =775e-12,
                              refractory_loop_output_inductance = 100e-12,
                              refractory_bias_currents = [74e-6,36e-6,35e-6],
                              refractory_receiving_input_inductance = [20e-12,1],
                              neuronal_receiving_input_refractory_inductance = [20e-12,1],
                              integration_loop_time_constant = 25e-9,
                              time_params = dict([['dt',dt],['tf',tf]]))           
            
            neuron_1.run_sim()
                                    
            actual_data__drive = np.vstack((neuron_1.time_vec[:],1e-6*neuron_1.synapses['synapse_under_test'].I_spd2_vec[:])) 
            error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                                            
            actual_data = np.vstack((neuron_1.time_vec[:],1e-6*neuron_1.synapses['synapse_under_test'].I_di_vec[:]))    
            error__signal = chi_squared_error(target_data,actual_data)
            
            plot_wr_comparison__synapse(file_name,spike_times,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)

