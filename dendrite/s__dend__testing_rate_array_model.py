import numpy as np
from matplotlib import pyplot as plt
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__dend_drive_and_response
from _functions import read_wr_data, chi_squared_error, dendritic_drive__piecewise_linear, dendritic_drive__exp_pls_train__LR, dendritic_drive__square_pulse_train
from soen_sim import input_signal, dendrite, neuron #synapse, 

plt.close('all')

#%% units
# all currents in uA
# all inductances in pH
# all times in ns
# all resistances in pH/ns ( aka milliohms )

#%% exp pulse seq

# plt.close('all')

dt = 0.1 # ns
tf = 255 # ns

exp_pls_trn_params = dict()
exp_pls_trn_params['t_r1_start'] = 5 # ns
exp_pls_trn_params['t_r1_rise'] = 1 # ns
exp_pls_trn_params['t_r1_pulse'] = 1 # ns
exp_pls_trn_params['t_r1_fall'] = 1 # ns
exp_pls_trn_params['t_r1_period'] = 50 # ns
exp_pls_trn_params['value_r1_off'] = 0 # pH/ns
exp_pls_trn_params['value_r1_on'] = 5e6 # mOhm
exp_pls_trn_params['r2'] = 5.004e3 # mOhm
exp_pls_trn_params['L1'] = 250e3 # pH
exp_pls_trn_params['L2'] = 200 # pH
exp_pls_trn_params['Ib'] = 28.13 # uA

num_jjs = 4
if num_jjs == 2:
    I_drive_vec = np.asarray([8.13,12.91]) # uA
elif num_jjs == 4:
    I_drive_vec = np.asarray([9.19,13.06]) # uA
L_di_vec = np.asarray([77.5,775])*1e3 # pH
tau_di_vec = np.asarray([10,100,1000]) # ns    

for ii in range(len(I_drive_vec)):
    I_drive = I_drive_vec[ii]
    exp_pls_trn_params['Ib'] = I_drive
    for jj in range(len(L_di_vec)):
        L_di = L_di_vec[jj]
        for kk in range(len(tau_di_vec)):
            tau_di = tau_di_vec[kk]
            
            print('\n\nii = {} of {}; jj = {} of {}; kk = {} of {}'.format(ii+1,len(I_drive_vec),jj+1,len(L_di_vec),kk+1,len(tau_di_vec)))
            
            # load WR data
            directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
            file_name = 'dend_exp_pls_seq_{:d}jj_Idrv{:05.2f}uA_Ide74.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,I_drive,L_di*1e-3,tau_di)
            data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
            if num_jjs == 2:
                I_di_str = 'L0#branch'
                I_drive_str = 'L2#branch'
            elif num_jjs == 4:
                I_di_str = 'L2#branch'
                I_drive_str = 'L4#branch'
            target_data = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_di_str]))
            target_data__drive = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_drive_str]))

            # setup soen sim for exp pulse seq
            input_1 = input_signal(name = 'input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200, 
                                    time_vec = np.arange(0,tf+dt,dt), exponential_pulse_train = exp_pls_trn_params)            

            dendrite_1 = dendrite(name = 'dendrite_under_test', num_jjs = num_jjs,
                                    inhibitory_or_excitatory = 'excitatory', circuit_inductances = [0,20,200,77.5], 
                                    input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                    input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                    input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[20,1]],
                                    junction_critical_current = 40, bias_currents = [74,36,35],
                                    integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0,
                                    integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di)

            neuron_1 = neuron(name = 'dummy_neuron', input_dendritic_connections = ['dendrite_under_test'],
                              junction_critical_current = 40,
                              circuit_inductances = [0,0,200,77.5],
                              input_dendritic_inductances = [[20,1]], 
                              refractory_loop_circuit_inductances = [0,20,200,77.5],
                              refractory_time_constant = 50,
                              refractory_junction_critical_current = 40,
                              refractory_loop_self_inductance =775,
                              refractory_loop_output_inductance = 100,
                              refractory_bias_currents = [74,36,35],
                              refractory_receiving_input_inductance = [20,1],
                              neuronal_receiving_input_refractory_inductance = [20,1],
                              integration_loop_time_constant = 25,
                              time_params = dict([['dt',dt],['tf',tf]]))           
            
            neuron_1.run_sim()
                                    
            actual_data__drive = np.vstack((neuron_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].direct_connections['input_dendritic_drive'].drive_signal[:])) 
            error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                                            
            actual_data = np.vstack((input_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
            error__signal = chi_squared_error(target_data,actual_data)
            
            plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)
            
#%% linear ramp

# plt.close('all')

dt = 0.1
tf = 50

num_jjs = 4
if num_jjs == 2:
    I_drive_vec = np.asarray([6.2,16])
elif num_jjs == 4:
    I_drive_vec = np.asarray([8.6,16])
L_di_vec = np.asarray([77.5,775,7750])*1e3
tau_di_vec = np.asarray([77.5e6,250]) # 


for kk in range(len(tau_di_vec)):
    for jj in range(len(L_di_vec)):
        L_di = L_di_vec[jj]
        if kk == 0:
            tau_di = L_di/1e-3
        else:
            tau_di = tau_di_vec[kk]
        
        print('\n\nii = {} of {}; jj = {} of {}; kk = {} of {}'.format(ii+1,len(I_drive_vec),jj+1,len(L_di_vec),kk+1,len(tau_di_vec)))
        
        # load WR data
        directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
        if tau_di > 1e7:
            file_name = 'dend_lin_ramp_{:d}jj_Idrv{:05.2f}uA{:05.2f}uA_Ide72.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ms_dt01.0ps.dat'.format(num_jjs,I_drive_vec[0],I_drive_vec[1],L_di*1e-3,tau_di*1e-6)
        if tau_di < 1e5:
            file_name = 'dend_lin_ramp_{:d}jj_Idrv{:05.2f}uA{:05.2f}uA_Ide72.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,I_drive_vec[0],I_drive_vec[1],L_di*1e-3,tau_di)    
        data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
        if num_jjs == 2:
            I_di_str = 'L0#branch'
            I_drive_str = 'L1#branch'
        elif num_jjs == 4:
            I_di_str = 'L2#branch'
            I_drive_str = 'L3#branch'
        target_data = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_di_str]))
        target_data__drive = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_drive_str]))

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
                                
        actual_data__drive = np.vstack((neuron_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].direct_connections['input_dendritic_drive'].drive_signal[:])) 
        error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                                        
        actual_data = np.vstack((input_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
        error__signal = chi_squared_error(target_data,actual_data)
        
        plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)            


#%% sq pls seq

# plt.close('all')

dt = 0.1

num_jjs = 4
L_di_vec = np.asarray([77.5,775,7750])*1e3
tau_di_vec = np.asarray([10,100,1000])

for jj in range(len(L_di_vec)):
    L_di = L_di_vec[jj]
    tau_di_vec
    for kk in range(len(tau_di_vec)):
        tau_di = tau_di_vec[kk]
        
        print('\n\njj = {} of {}; kk = {} of {}'.format(jj+1,len(L_di_vec),kk+1,len(tau_di_vec)))
        
        # load WR data
        directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
        file_name = 'dend_sq_pls_seq_{:d}jj_Ide74.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,L_di*1e-3,tau_di)
        data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
        if num_jjs == 2:
            I_di_str = 'L0#branch'
            I_drive_str = 'L1#branch'
        elif num_jjs == 4:
            I_di_str = 'L2#branch'
            I_drive_str = 'L3#branch'
        target_data = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_di_str]))
        target_data__drive = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_drive_str]))

        # setup soen sim for linear ramp            
        pwl_drive = [[0,0],[4.9,0],[5,7.84],[15,7.84],[15.1,0],[24.9,0],[25,13.21],[35,13.21],[35.1,0],[44.9,0],[45,9.17],[55,9.17],[55.1,0],[64.9,0],[65,15.84],[75,15.84],[75.1,0]]
        tf = 1e9*data_dict['time'][-1]
        input_1 = input_signal(name = 'input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200, 
                                time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)            

        dendrite_1 = dendrite(name = 'dendrite_under_test', num_jjs = num_jjs,
                                inhibitory_or_excitatory = 'excitatory', circuit_inductances = [0,20,200,77.5], 
                                input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[20,1]],
                                thresholding_junction_critical_current = 40, bias_currents = [74,36,35],
                                integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0,
                                integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di)

        time_params = dict()
        time_params['dt'] = dt
        time_params['tf'] = tf
        neuron_1 = neuron(name = 'dummy_neuron', input_dendritic_connections = ['dendrite_under_test'], 
                          circuit_inductances = [0,0,200,77.5],
                          input_dendritic_inductances = [[20,1]], 
                          refractory_loop_circuit_inductances = [0,20,200,77.5],
                          refractory_time_constant = 50,
                          refractory_thresholding_junction_critical_current = 40,
                          refractory_loop_self_inductance = 775,
                          refractory_loop_output_inductance = 100,
                          refractory_bias_currents = [74,36,35],
                          refractory_receiving_input_inductance = [20,1],
                          neuronal_receiving_input_refractory_inductance = [20,1],
                          time_params = time_params)           
        
        neuron_1.run_sim()
                                
        actual_data__drive = np.vstack((neuron_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].direct_connections['input_dendritic_drive'].drive_signal[:])) 
        error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                                        
        actual_data = np.vstack((input_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
        error__signal = chi_squared_error(target_data,actual_data)
        
        plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)    

# sq_pls_trn_params = dict()
# sq_pls_trn_params['t_start'] = 5
# sq_pls_trn_params['t_rise'] = 1
# sq_pls_trn_params['t_pulse'] = 5
# sq_pls_trn_params['t_fall'] = 1
# sq_pls_trn_params['t_period'] = 20
# sq_pls_trn_params['value_off'] = 0
# sq_pls_trn_params['value_on'] = 23.81
