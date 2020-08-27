import numpy as np
from matplotlib import pyplot as plt
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__dend_drive_and_response
from _functions import read_wr_data, chi_squared_error, dendritic_drive__piecewise_linear, dendritic_drive__exp_pls_train__LR, dendritic_drive__square_pulse_train
from soen_sim import input_signal, synapse, dendrite, neuron

plt.close('all')

#%% constant drive
file_name = 'dend_cnst_drv_Idrv20.00uA_Ldi0077.50nH_taudi0775ms_tsim0050ns_dt01.0ps.dat'
data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
target_data = np.vstack((data_dict['time'],data_dict['L9#branch']))
target_data__drive = np.vstack((data_dict['time'],data_dict['L4#branch']))

# setup soen sim for constant drive
L_di = 77.5e-9
tau_di = 7.75e-3
dt = 0.1e-9
tf = 50e-9

pwl_drive = [[0e-9,0e-6],[1e-9,0e-6],[2e-9,20e-6]]

input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)

dendritic_drive = dendritic_drive__piecewise_linear(input_1.time_vec,pwl_drive)
actual_data__drive = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
error__drive = chi_squared_error(target_data__drive,actual_data__drive)
        
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf
dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                                      input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                      input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                      input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                                      thresholding_junction_critical_current = 40e-6, bias_currents = [71.5e-6,36e-6,35e-6],
                                      integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
                                      integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
                                      dendrite_model_params = sim_params)

dendrite_1.run_sim()
                                
actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))    
error__signal = chi_squared_error(target_data,actual_data)

plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)

#%% linear ramp
# file_name = 'dend_lin_ramp_Idrv18.0-30.0uA_Ldi0077.50nH_taudi0010.0ns_tsim50ns_dt01.0ps.dat'
file_name = 'dend_lin_ramp_Idrv18.0-30.0uA_Ldi0077.50nH_taudi0100.0ns_tsim50ns_dt01.0ps.dat'
# file_name = 'dend_lin_ramp_Idrv18.0-30.0uA_Ldi0077.50nH_taudi1000.0ns_tsim50ns_dt01.0ps.dat'
data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
target_data = np.vstack((data_dict['time'],data_dict['L9#branch']))
target_data__drive = np.vstack((data_dict['time'],data_dict['L4#branch']))

# setup soen sim for linear ramp
L_di = 77.5e-9
tau_di = 100e-9
dt = 0.1e-9
tf = 50e-9

pwl_drive = [[0e-9,0e-6],[1e-9,0e-6],[2e-9,18e-6],[42e-9,30e-6]]

input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)

dendritic_drive = dendritic_drive__piecewise_linear(input_1.time_vec,pwl_drive)
actual_data__drive = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
error__drive = chi_squared_error(target_data__drive,actual_data__drive)
        
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf
dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                                      input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                      input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                      input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                                      thresholding_junction_critical_current = 40e-6, bias_currents = [71.5e-6,36e-6,35e-6],
                                      integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
                                      integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
                                      dendrite_model_params = sim_params)

dendrite_1.run_sim()
                                
actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))    
error__signal = chi_squared_error(target_data,actual_data)

plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)

#%% sq pls seq

# file_name = 'dend_sq_pls_seq_Idrv23.00uA_Ldi0077.50nH_taudi0100.0ns_tsim0100ns_dt01.0ps.dat'
# file_name = 'dend_sq_pls_seq_Idrv23.00uA_Ldi0077.50nH_taudi0010.0ns_tsim0100ns_dt01.0ps.dat'
# file_name = 'dend_sq_pls_seq_Idrv23.00uA_Ldi0077.50nH_taudi1000.0ns_tsim0100ns_dt01.0ps.dat'
file_name = 'dend_sq_pls_seq_Idrv23.81uA_Ldi0077.50nH_taudi1000.0ns_tsim0100ns_dt01.0ps.dat'
data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
target_data = np.vstack((data_dict['time'],data_dict['L9#branch']))
target_data__drive = np.vstack((data_dict['time'],data_dict['L4#branch']))

# setup soen sim for sq pulse seq
L_di = 77.5e-9
tau_di = 1000e-9

dt = 0.1e-9
tf = 100e-9

sq_pls_trn_params = dict()
sq_pls_trn_params['t_start'] = 5e-9
sq_pls_trn_params['t_rise'] = 1e-9
sq_pls_trn_params['t_pulse'] = 5e-9
sq_pls_trn_params['t_fall'] = 1e-9
sq_pls_trn_params['t_period'] = 20e-9
sq_pls_trn_params['value_off'] = 0
sq_pls_trn_params['value_on'] = 23.81e-6

input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), square_pulse_train = sq_pls_trn_params)

dendritic_drive = dendritic_drive__square_pulse_train(input_1.time_vec,sq_pls_trn_params)
actual_data__drive = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
error__drive = chi_squared_error(target_data__drive,actual_data__drive)
        
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf
dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                                      input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                      input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                      input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                                      thresholding_junction_critical_current = 40e-6, bias_currents = [71.5e-6,36e-6,35e-6],
                                      integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
                                      integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
                                      dendrite_model_params = sim_params)

dendrite_1.run_sim()
                                
actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))    
error__signal = chi_squared_error(target_data,actual_data)

plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)


#%% exp pulse seq
# file_name = 'dend_exp_pls_seq_Idrv23.81uA_Ldi0077.50nH_taudi0010.0ns_tsim0200ns_dt01.0ps.dat'
# file_name = 'dend_exp_pls_seq_Idrv23.81uA_Ldi0077.50nH_taudi0100.0ns_tsim0200ns_dt01.0ps.dat'
# file_name = 'dend_exp_pls_seq_Idrv23.81uA_Ldi0077.50nH_taudi1000.0ns_tsim0200ns_dt01.0ps.dat'
file_name = 'dend_exp_pls_seq_Idrv28.13uA_Ldi0775.0nH_taudi0010.0ns_tsim0500ns_dt01.0ps.dat'
# file_name = 'dend_exp_pls_seq_Idrv28.13uA_Ldi0775.0nH_taudi0100.0ns_tsim0500ns_dt01.0ps.dat'
# file_name = 'dend_exp_pls_seq_Idrv28.13uA_Ldi0775.0nH_taudi1000.0ns_tsim0500ns_dt01.0ps.dat'
data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
target_data = np.vstack((data_dict['time'],data_dict['L10#branch']))
target_data__drive = np.vstack((data_dict['time'],data_dict['L5#branch']))

# setup soen sim for exp pulse seq
# L_di = 77.5e-9
# tau_di = 7.75e-3

L_di = 775e-9
tau_di = 10e-9

dt = 0.1e-9
tf = 500e-9

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

input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), exponential_pulse_train = exp_pls_trn_params)

dendritic_drive = dendritic_drive__exp_pls_train__LR(input_1.time_vec,exp_pls_trn_params)
actual_data__drive = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
error__drive = chi_squared_error(target_data__drive,actual_data__drive)
        
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf
dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                                      input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                      input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                      input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                                      thresholding_junction_critical_current = 40e-6, bias_currents = [71.5e-6,36e-6,35e-6],
                                      integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
                                      integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
                                      dendrite_model_params = sim_params)

dendrite_1.run_sim()
                                
actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))    
error__signal = chi_squared_error(target_data,actual_data)

plot_wr_comparison__dend_drive_and_response(file_name,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)

#%%

if 1 == 2:
    
    with open('../master_rate_matrix.soen', 'rb') as data_file:         
            data_array_imported = pickle.load(data_file)
            
    master_rate_matrix = data_array_imported['master_rate_matrix']
