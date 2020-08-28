import numpy as np
from matplotlib import pyplot as plt
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__dend_drive_and_response
from _functions import read_wr_data, chi_squared_error, dendritic_drive__piecewise_linear, dendritic_drive__exp_pls_train__LR, dendritic_drive__square_pulse_train
from soen_sim import input_signal, synapse, neuron # dendrite, 

plt.close('all')

#%% exp pulse seq

dt = 0.1e-9
tf = 255e-9

exp_pls_trn_params = dict()
exp_pls_trn_params['t_r1_start'] = 5e-9
exp_pls_trn_params['t_r1_rise'] = 1e-9
exp_pls_trn_params['t_r1_pulse'] = 1e-9
exp_pls_trn_params['t_r1_fall'] = 1e-9
exp_pls_trn_params['t_r1_period'] = 50e-9
exp_pls_trn_params['value_r1_off'] = 0
exp_pls_trn_params['value_r1_on'] = 5e3
exp_pls_trn_params['r2'] = 5.004
exp_pls_trn_params['L1'] = 250e-9
exp_pls_trn_params['L2'] = 200e-12
exp_pls_trn_params['Ib'] = 28.13e-6

num_jjs = 4
if num_jjs == 2:
    I_drive_vec = np.asarray([8.13,12.91])*1e-6
elif num_jjs == 4:
    I_drive_vec = np.asarray([9.19,13.06])*1e-6
L_di_vec = np.asarray([77.5,775])*1e-9
tau_di_vec = np.asarray([10,100,1000])*1e-9    

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
            file_name = 'dend_exp_pls_seq_{:d}jj_Idrv{:05.2f}uA_Ide74.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,I_drive*1e6,L_di*1e9,tau_di*1e9)
            data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
            if num_jjs == 2:
                I_di_str = 'L0#branch'
                I_drive_str = 'L2#branch'
            elif num_jjs == 4:
                I_di_str = 'L2#branch'
                I_drive_str = 'L4#branch'
            target_data = np.vstack((data_dict['time'],data_dict[I_di_str]))
            target_data__drive = np.vstack((data_dict['time'],data_dict[I_drive_str]))

            # setup soen sim for exp pulse seq
            input_1 = input_signal(name = 'input_synaptic_drive', 
                                   input_temporal_form = 'single_spike', # 'single_spike' or 'constant_rate' or 'arbitrary_spike_train'
                                   spike_times = [5e-9])            
        
            synapse_1 = synapse(name = 'synapse_under_test',
                                synaptic_circuit_inductors = [100e-9,100e-9,400e-12],
                                synaptic_circuit_resistors = [5e3,5],
                                synaptic_hotspot_duration = 200e-12,
                                synaptic_spd_current = 10e-6,
                                input_direct_connectons = 'input_synaptic_drive',
                                num_jjs = num_jjs,
                                inhibitory_or_excitatory = 'excitatory',
                                synaptic_dendrite_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                                synaptic_dendrite_input_synaptic_inductance = [20e-12,1],
                                junction_critical_current = 40e-6,
                                bias_currents = [72e-6, 36e-6, 35e-6],
                                integration_loop_self_inductance = 77.5e-9,
                                integration_loop_output_inductance = 200e-12,
                                integration_loop_time_constant = 250e-9)
       
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
                                    
            actual_data__drive = np.vstack((neuron_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].direct_connections['input_dendritic_drive'].drive_signal[:])) 
            error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                                            
            actual_data = np.vstack((input_1.time_vec[:],1e-6*neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
            error__signal = chi_squared_error(target_data,actual_data)
            
            plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)

            
#%% linear ramp

dt = 1e-9
tf = 50e-9


num_jjs = 4
if num_jjs == 2:
    I_drive_vec = np.asarray([6.2,17])*1e-6
elif num_jjs == 4:
    I_drive_vec = np.asarray([8.6,17])*1e-6
L_di_vec = np.asarray([77.5,775,7750])*1e-9
tau_di_vec = np.asarray([77.5e-3,250e-9]) 

for jj in range(len(L_di_vec)):
    L_di = L_di_vec[jj]
    for kk in range(len(tau_di_vec)):
        if kk == 0:
            tau_di = L_di/1e-6
        else:
            tau_di = tau_di_vec[kk]
        
        print('\n\nii = {} of {}; jj = {} of {}; kk = {} of {}'.format(ii+1,len(I_drive_vec),jj+1,len(L_di_vec),kk+1,len(tau_di_vec)))
        
        # load WR data
        directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
        if tau_di > 1e-5:
            file_name = 'dend_lin_ramp_{:d}jj_Idrv{:05.2f}uA{:05.2f}uA_Ide72.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ms_dt01.0ps.dat'.format(num_jjs,I_drive_vec[0]*1e6,I_drive_vec[1]*1e6,L_di*1e9,tau_di*1e3)
        if tau_di < 1e-5:
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

        # setup soen sim for linear ramp            
        pwl_drive = [[0e-9,0e-6],[1e-9,0e-6],[2e-9,I_drive_vec[0]],[42e-9,I_drive_vec[1]]]
        input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)            
    
        sim_params = dict()
        sim_params['dt'] = dt
        sim_params['tf'] = tf
        dendrite_1 = dendrite('dendrite_under_test', num_jjs = num_jjs,
                                inhibitory_or_excitatory = 'excitatory', circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12], 
                                input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[20e-12,1]],
                                thresholding_junction_critical_current = 40e-6, bias_currents = [72e-6,36e-6,35e-6],
                                integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
                                integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
                                dendrite_model_params = sim_params)

        time_params = dict()
        time_params['dt'] = dt
        time_params['tf'] = tf
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
                          time_params = time_params)           
        
        neuron_1.run_sim()
                                
        actual_data__drive = np.vstack((neuron_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].direct_connections['input_dendritic_drive'].drive_signal[:])) 
        error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                                        
        actual_data = np.vstack((input_1.time_vec[:],1e-6*neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
        error__signal = chi_squared_error(target_data,actual_data)
        
        plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)            

#%% constant drive
# file_name = 'dend_cnst_drv_Idrv20.00uA_Ldi0077.50nH_taudi0775ms_tsim0050ns_dt01.0ps.dat'
# data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
# target_data = np.vstack((data_dict['time'],data_dict['L9#branch']))
# target_data__drive = np.vstack((data_dict['time'],data_dict['L4#branch']))

# # setup soen sim for constant drive
# L_di = 77.5e-9
# tau_di = 7.75e-3
# dt = 0.1e-9
# tf = 50e-9

# pwl_drive = [[0e-9,0e-6],[1e-9,0e-6],[2e-9,20e-6]]

# input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
#                                     time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)

# dendritic_drive = dendritic_drive__piecewise_linear(input_1.time_vec,pwl_drive)
# actual_data__drive = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
# error__drive = chi_squared_error(target_data__drive,actual_data__drive)
        
# sim_params = dict()
# sim_params['dt'] = dt
# sim_params['tf'] = tf
# dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
#                                       input_synaptic_connections = [], input_synaptic_inductances = [[]], 
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

# #%% linear ramp
# # file_name = 'dend_lin_ramp_Idrv18.0-30.0uA_Ldi0077.50nH_taudi0010.0ns_tsim50ns_dt01.0ps.dat'
# file_name = 'dend_lin_ramp_Idrv18.0-30.0uA_Ldi0077.50nH_taudi0100.0ns_tsim50ns_dt01.0ps.dat'
# # file_name = 'dend_lin_ramp_Idrv18.0-30.0uA_Ldi0077.50nH_taudi1000.0ns_tsim50ns_dt01.0ps.dat'
# data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
# target_data = np.vstack((data_dict['time'],data_dict['L9#branch']))
# target_data__drive = np.vstack((data_dict['time'],data_dict['L4#branch']))

# # setup soen sim for linear ramp
# L_di = 77.5e-9
# tau_di = 100e-9
# dt = 0.1e-9
# tf = 50e-9

# pwl_drive = [[0e-9,0e-6],[1e-9,0e-6],[2e-9,18e-6],[42e-9,30e-6]]

# input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
#                                     time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)

# dendritic_drive = dendritic_drive__piecewise_linear(input_1.time_vec,pwl_drive)
# actual_data__drive = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
# error__drive = chi_squared_error(target_data__drive,actual_data__drive)
        
# sim_params = dict()
# sim_params['dt'] = dt
# sim_params['tf'] = tf
# dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
#                                       input_synaptic_connections = [], input_synaptic_inductances = [[]], 
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

# #%% sq pls seq

# # file_name = 'dend_sq_pls_seq_Idrv23.00uA_Ldi0077.50nH_taudi0100.0ns_tsim0100ns_dt01.0ps.dat'
# # file_name = 'dend_sq_pls_seq_Idrv23.00uA_Ldi0077.50nH_taudi0010.0ns_tsim0100ns_dt01.0ps.dat'
# # file_name = 'dend_sq_pls_seq_Idrv23.00uA_Ldi0077.50nH_taudi1000.0ns_tsim0100ns_dt01.0ps.dat'
# file_name = 'dend_sq_pls_seq_Idrv23.81uA_Ldi0077.50nH_taudi1000.0ns_tsim0100ns_dt01.0ps.dat'
# data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
# target_data = np.vstack((data_dict['time'],data_dict['L9#branch']))
# target_data__drive = np.vstack((data_dict['time'],data_dict['L4#branch']))

# # setup soen sim for sq pulse seq
# L_di = 77.5e-9
# tau_di = 1000e-9

# dt = 0.1e-9
# tf = 100e-9

# sq_pls_trn_params = dict()
# sq_pls_trn_params['t_start'] = 5e-9
# sq_pls_trn_params['t_rise'] = 1e-9
# sq_pls_trn_params['t_pulse'] = 5e-9
# sq_pls_trn_params['t_fall'] = 1e-9
# sq_pls_trn_params['t_period'] = 20e-9
# sq_pls_trn_params['value_off'] = 0
# sq_pls_trn_params['value_on'] = 23.81e-6

# input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
#                                     time_vec = np.arange(0,tf+dt,dt), square_pulse_train = sq_pls_trn_params)

# dendritic_drive = dendritic_drive__square_pulse_train(input_1.time_vec,sq_pls_trn_params)
# actual_data__drive = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
# error__drive = chi_squared_error(target_data__drive,actual_data__drive)
        
# sim_params = dict()
# sim_params['dt'] = dt
# sim_params['tf'] = tf
# dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
#                                       input_synaptic_connections = [], input_synaptic_inductances = [[]], 
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




# #%%

# if 1 == 2:
    
#     with open('../master_rate_matrix.soen', 'rb') as data_file:         
#             data_array_imported = pickle.load(data_file)
            
#     master_rate_matrix = data_array_imported['master_rate_matrix']
