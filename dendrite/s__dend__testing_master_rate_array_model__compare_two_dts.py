import numpy as np
from matplotlib import pyplot as plt

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__dend_drive_and_response__compare_two
from _functions import read_wr_data, chi_squared_error
from soen_sim import input_signal, dendrite, neuron #synapse, 

plt.close('all')
            
#%% linear ramp

num_jjs = 2
dt_vec = [100e-12,1e-9]

if num_jjs == 2:
    I_drive_vec = np.asarray([6.2,16])*1e-6
elif num_jjs == 4:
    I_drive_vec = np.asarray([8.6,16])*1e-6
L_di_vec = np.asarray([77.5,775,7750])*1e-9
tau_di_vec = np.asarray([77.5e-3,250e-9]) # 

tf = 50e-9
for kk in range(len(tau_di_vec)):
    for jj in range(len(L_di_vec)):
        L_di = L_di_vec[jj]
        if kk == 0:
            tau_di = L_di/1e-6
        else:
            tau_di = tau_di_vec[kk]
            
        # load WR data
        directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
        if tau_di > 1e-2:
            print('here1')
            file_name = 'dend_lin_ramp_{:d}jj_Idrv{:05.2f}uA{:05.2f}uA_Ide72.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ms_dt01.0ps.dat'.format(num_jjs,I_drive_vec[0]*1e6,I_drive_vec[1]*1e6,L_di*1e9,tau_di*1e3)
        if tau_di < 1e-4:
            file_name = 'dend_lin_ramp_{:d}jj_Idrv{:05.2f}uA{:05.2f}uA_Ide72.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,I_drive_vec[0]*1e6,I_drive_vec[1]*1e6,L_di*1e9,tau_di*1e9)    
        data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
        if num_jjs == 2:
            I_di_str = 'L0#branch'
            I_drive_str = 'L1#branch'
        elif num_jjs == 4:
            I_di_str = 'L2#branch'
            I_drive_str = 'L3#branch'
        target_data = np.vstack((data_dict['time'],data_dict[I_di_str]))
        target_data__drive = np.vstack((data_dict['time'],data_dict[I_drive_str]))
        # tf = data_dict['time'][-1]
                    
        for mm in range(len(dt_vec)):
            dt = dt_vec[mm]
        
            print('\n\nkk = {} of {} (tau_di); jj = {} of {} (L_di); mm = {} of {} (dt)'.format(kk+1,len(tau_di_vec),jj+1,len(L_di_vec),mm+1,len(dt_vec)))
    
            # setup soen sim for linear ramp            
            pwl_drive = [[0e-9,0e-6],[1e-9,0e-6],[2e-9,I_drive_vec[0]],[42e-9,I_drive_vec[1]]]
            input_1 = input_signal(name = 'input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)            
    
            dendrite_1 = dendrite('dendrite_under_test', num_jjs = num_jjs,
                                    inhibitory_or_excitatory = 'excitatory', circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12], 
                                    input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                    input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                    input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[20e-12,1]],
                                    thresholding_junction_critical_current = 40e-6, bias_currents = [72e-6,36e-6,35e-6],
                                    integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
                                    integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di)
    
            neuron_1 = neuron('dummy_neuron', input_dendritic_connections = ['dendrite_under_test'], 
                              circuit_inductances = [0e-12,0e-12,200e-12,77.5e-12],
                              input_dendritic_inductances = [[20e-12,1]], 
                              refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                              refractory_time_constant = 50e-9,
                              refractory_thresholding_junction_critical_current = 40e-6,
                              refractory_loop_self_inductance =775e-12,
                              refractory_loop_output_inductance = 100e-12,
                              refractory_bias_currents = [74e-6,36e-6,35e-6],
                              refractory_receiving_input_inductance = [20e-12,1],
                              neuronal_receiving_input_refractory_inductance = [20e-12,1],
                              time_params = dict([['dt',dt],['tf',tf]]))           
            
            neuron_1.run_sim()
                                    
            actual_data__drive = np.vstack((neuron_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].direct_connections['input_dendritic_drive'].drive_signal[:])) 
            error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                                            
            actual_data = np.vstack((input_1.time_vec[:],1e-6*neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
            error__signal = chi_squared_error(target_data,actual_data)
            
            if mm == 0:
                actual_data__drive__1 = actual_data__drive
                actual_data__1 = actual_data
                error__drive__1 = error__drive
                error__signal__1 = error__signal
            if mm == 1:
                actual_data__drive__2 = actual_data__drive
                actual_data__2 = actual_data
                error__drive__2 = error__drive
                error__signal__2 = error__signal
        
        plot_wr_comparison__dend_drive_and_response__compare_two(file_name,target_data__drive,actual_data__drive__1,actual_data__drive__2,target_data,actual_data__1,actual_data__2,error__drive__1,error__drive__2,error__signal__1,error__signal__2,dt_vec)            


#%% sq pls seq

# plt.close('all')

dt_vec = [100e-12,1e-9]

L_di_vec = np.asarray([77.5,775,7750])*1e-9
tau_di_vec = np.asarray([10,100,1000])*1e-9 

tf = 100e-9
for jj in range(len(L_di_vec)):
    L_di = L_di_vec[jj]
    tau_di_vec
    for kk in range(len(tau_di_vec)):
        tau_di = tau_di_vec[kk]
        
        print('\n\njj = {} of {}; kk = {} of {}'.format(jj+1,len(L_di_vec),kk+1,len(tau_di_vec)))
        
        # load WR data
        directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
        file_name = 'dend_sq_pls_seq_{:d}jj_Ide74.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,L_di*1e9,tau_di*1e9)
        data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
        if num_jjs == 2:
            I_di_str = 'L0#branch'
            I_drive_str = 'L1#branch'
        elif num_jjs == 4:
            I_di_str = 'L2#branch'
            I_drive_str = 'L3#branch'
        tf = data_dict['time'][-1]
        target_data = np.vstack((data_dict['time'],data_dict[I_di_str]))
        target_data__drive = np.vstack((data_dict['time'],data_dict[I_drive_str]))
        tf = data_dict['time'][-1]
        
        for mm in range(len(dt_vec)):
            dt = dt_vec[mm]
        
            print('\n\nkk = {} of {} (tau_di); jj = {} of {} (L_di); mm = {} of {} (dt)'.format(kk+1,len(tau_di_vec),jj+1,len(L_di_vec),mm+1,len(dt_vec)))

            # setup soen sim for linear ramp            
            pwl_drive = [[0e-9,0e-6],[4.9e-9,0e-6],[5e-9,7.84e-6],[15e-9,7.84e-6],[15.1e-9,0e-6],[24.9e-9,0e-6],[25e-9,13.21e-6],[35e-9,13.21e-6],[35.1e-9,0e-6],[44.9e-9,0e-6],[45e-9,9.17e-6],[55e-9,9.17e-6],[55.1e-9,0e-6],[64.9e-9,0e-6],[65e-9,15.84e-6],[75e-9,15.84e-6],[75.1e-9,0e-6],]
            input_1 = input_signal(name = 'input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)            
    
            dendrite_1 = dendrite('dendrite_under_test', num_jjs = num_jjs,
                                    inhibitory_or_excitatory = 'excitatory', circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12], 
                                    input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                    input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                    input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[20e-12,1]],
                                    thresholding_junction_critical_current = 40e-6, bias_currents = [74e-6,36e-6,35e-6],
                                    integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
                                    integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di)
    
            neuron_1 = neuron('dummy_neuron', input_dendritic_connections = ['dendrite_under_test'], 
                              circuit_inductances = [0e-12,0e-12,200e-12,77.5e-12],
                              input_dendritic_inductances = [[20e-12,1]], 
                              refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                              refractory_time_constant = 50e-9,
                              refractory_thresholding_junction_critical_current = 40e-6,
                              refractory_loop_self_inductance =775e-12,
                              refractory_loop_output_inductance = 100e-12,
                              refractory_bias_currents = [74e-6,36e-6,35e-6],
                              refractory_receiving_input_inductance = [20e-12,1],
                              neuronal_receiving_input_refractory_inductance = [20e-12,1],
                              time_params = dict([['dt',dt],['tf',tf]]))           
            
            neuron_1.run_sim()
                                    
            actual_data__drive = np.vstack((neuron_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].direct_connections['input_dendritic_drive'].drive_signal[:])) 
            error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                                            
            actual_data = np.vstack((input_1.time_vec[:],1e-6*neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
            error__signal = chi_squared_error(target_data,actual_data)
            
            if mm == 0:
                actual_data__drive__1 = actual_data__drive
                actual_data__1 = actual_data
                error__drive__1 = error__drive
                error__signal__1 = error__signal
            if mm == 1:
                actual_data__drive__2 = actual_data__drive
                actual_data__2 = actual_data
                error__drive__2 = error__drive
                error__signal__2 = error__signal
        
        plot_wr_comparison__dend_drive_and_response__compare_two(file_name,target_data__drive,actual_data__drive__1,actual_data__drive__2,target_data,actual_data__1,actual_data__2,error__drive__1,error__drive__2,error__signal__1,error__signal__2,dt_vec)            
