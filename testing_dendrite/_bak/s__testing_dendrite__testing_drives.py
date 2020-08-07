#%%
import numpy as np
import time

from soen_sim import input_signal, synapse, dendrite, neuron
from _functions import dendritic_drive__piecewise_linear, dendritic_drive__square_pulse_train, dendritic_drive__exponential, read_wr_data, chi_squared_error, dendritic_drive__exp_pls_train__LR
from _plotting import plot_dendritic_drive, plot_wr_comparison

#%%
dt = 0.01e-9
tf = 200e-9

#%% piecewise linear

t_on = 1.99e-9
t_rise = 10e-12
t_pulse = 10e-9
t_fall = 100e-12
t_period = 20e-9
value_off = 0
value_on = 20e-6
pwl = [[0,value_off],[t_on,value_off],[t_on+t_rise,value_on]]

input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                       time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl)  

dendritic_drive = dendritic_drive__piecewise_linear(input_1.time_vec,pwl)

# plot_dendritic_drive(input_1.time_vec, dendritic_drive)

file_name = 'dend__cnst_drv__Idrv20uA_Ldi7.75nH_taudi7.75ms_tsim200ns.dat'
data_dict = read_wr_data('wrspice_data/constant_drive/'+file_name)
target_data = np.vstack((data_dict['time'],data_dict['@I0[c]']))
actual_data = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
error = chi_squared_error(target_data,actual_data)
plot_wr_comparison(target_data,actual_data,'WR comparison, drive signals','{}_error={:1.4f}'.format(file_name,error),'$I_{flux}$ [$\mu$A]')
        
#%% square pulse train
sq_pls_trn_params = dict()
sq_pls_trn_params['t_start'] = 5e-9
sq_pls_trn_params['t_rise'] = 10e-12
sq_pls_trn_params['t_pulse'] = 10e-9
sq_pls_trn_params['t_fall'] = 10e-12
sq_pls_trn_params['t_period'] = 20e-9
sq_pls_trn_params['value_off'] = 0
sq_pls_trn_params['value_on'] = 20e-6

input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                       time_vec = np.arange(0,tf+dt,dt), square_pulse_train = sq_pls_trn_params)  

dendritic_drive = dendritic_drive__square_pulse_train(input_1.time_vec,sq_pls_trn_params)

# plot_dendritic_drive(input_1.time_vec, dendritic_drive) 

file_name = 'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi50ns_tsim_200ns.dat'
data_dict = read_wr_data('wrspice_data/square_pulse_sequence/'+file_name)
target_data = np.vstack((data_dict['time'],data_dict['@I0[c]']))
actual_data = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
error = chi_squared_error(target_data,actual_data)
plot_wr_comparison(target_data,actual_data,'WR comparison, drive signals','{}_error={:1.4f}'.format(file_name,error),'$I_{flux}$ [$\mu$A]')

#%% exponential
tf = 1e-6

exp_params = dict()
exp_params['t_rise'] = 5e-9
exp_params['t_fall'] = 5.4e-9
exp_params['tau_rise'] = 4.999e-11
exp_params['tau_fall'] = 50e-9
exp_params['value_on'] = 20e-6
exp_params['value_off'] = 0
  
input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                       time_vec = np.arange(0,tf+dt,dt), exponential = exp_params)  

dendritic_drive = dendritic_drive__exponential(input_1.time_vec,exp_params)

# plot_dendritic_drive(input_1.time_vec, dendritic_drive) 

file_name = 'dend__exp_pls_seq__amp20uA_tauin50ns_per1000ns_Ldi7.75nH_taudi100ns.dat'
data_dict = read_wr_data('wrspice_data/exponential_pulse_sequence/'+file_name)
target_data = np.vstack((data_dict['time'],data_dict['L5#branch']))
actual_data = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
error = chi_squared_error(target_data,actual_data)
plot_wr_comparison(target_data,actual_data,'WR comparison, drive signals','{}_error={:1.4f}'.format(file_name,error),'$I_{flux}$ [$\mu$A]')   

#%% exponential pulse train
tf = 1e-6

exp_pls_trn_params = dict()
exp_pls_trn_params['t_r1_start'] = 5e-9
exp_pls_trn_params['t_r1_rise'] = 100e-12
exp_pls_trn_params['t_r1_pulse'] = 200e-12
exp_pls_trn_params['t_r1_fall'] = 100e-12
exp_pls_trn_params['t_r1_period'] = 100e-9
exp_pls_trn_params['value_r1_off'] = 0
exp_pls_trn_params['value_r1_on'] = 5e3
exp_pls_trn_params['r2'] = 5.004
exp_pls_trn_params['L1'] = 250e-9
exp_pls_trn_params['L2'] = 200e-12
exp_pls_trn_params['Ib'] = 20e-6
  
input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                       time_vec = np.arange(0,tf+dt,dt), exponential_pls_train = exp_pls_trn_params)  

dendritic_drive = dendritic_drive__exp_pls_train__LR(input_1.time_vec,exp_pls_trn_params)

# plot_dendritic_drive(input_1.time_vec, dendritic_drive) 

file_name = 'dend__exp_pls_seq__amp20uA_tauin50ns_per100ns_Ldi7.75nH_taudi100ns.dat'
data_dict = read_wr_data('wrspice_data/exponential_pulse_sequence/'+file_name)
target_data = np.vstack((data_dict['time'],data_dict['L5#branch']))
actual_data = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
error = chi_squared_error(target_data,actual_data)
plot_wr_comparison(target_data,actual_data,'WR comparison, drive signals','{}_error={:1.4f}'.format(file_name,error),'$I_{flux}$ [$\mu$A]')   