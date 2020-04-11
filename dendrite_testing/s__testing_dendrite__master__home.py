#%%
import numpy as np
from matplotlib import pyplot as plt
import time

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_error_mat
from _functions__more import dendrite_model__parameter_sweep

plt.close('all')

#%% set up parameters that apply to all
    
num_loop = 7

amp_vec = np.linspace(15,15,1) # np.logspace(np.log10(7),np.log10(200),num_loop)#np.linspace(25,25,1)#np.linspace(25,25,1)#
mu1_vec = np.linspace(1,2.5,num_loop)
mu2_vec = np.linspace(0.25,1,num_loop)
mu3_vec = np.linspace(0.1,2.5,num_loop)
mu4_vec = np.linspace(0.1,2.5,num_loop)

#%% constant drive, no leak, vary L_di

# data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi7.75nH_taudi7.75ms_tsim200ns']  
                  
# data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi7.75nH_taudi7.75ms_tsim200ns',                                    
#                   'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi77.5ms_tsim200ns',
#                   'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi775ms_tsim200ns',
#                   'dend__cnst_drv__Idrv20uA_Ldi7.75uH_taudi7.75s_tsim200ns',
#                   'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi775ms_tsim1000ns',
#                   'dend__cnst_drv__Idrv20uA_Ldi7.75uH_taudi7.75s_tsim1000ns']

data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi7.75nH_taudi7.75ms_tsim200ns',                                    
                  'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi77.5ms_tsim200ns',
                  'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi775ms_tsim200ns',
                  'dend__cnst_drv__Idrv20uA_Ldi7.75uH_taudi7.75s_tsim200ns']

drive_info = dict()
drive_info['drive_type'] = 'piecewise_linear'
drive_info['pwl_drive'] = [[0e-9,0e-6],[1.9e-9,0e-6],[2e-9,20e-6]]

L_di_vec = [7.75e-9,77.5e-9,775e-9,7.75e-6,775e-9,7.75e-6]
tau_di_vec = [7.75e-3,77.5e-3,775e-3,7.75,0.775,7.75]
dt_vec = [0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9]
tf_vec = [200e-9,200e-9,200e-9,200e-9,1000e-9,1000e-9]

master_error_plot_name = 'no_leak__vary_L_di'
 
#call main sweep function 
t_tot = time.time()        
best_params__no_leak__vary_Ldi, error_mat_master__mu1_mu2__no_leak__vary_Ldi, error_mat_master__mu3_mu4__no_leak__vary_Ldi = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for no leak, vary L_di')

#%% constant drive, vary tau_di
# data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi50ns_tsim200ns',                                    
#                   'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi100ns_tsim200ns',
#                   'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi200ns_tsim200ns',
#                   'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi1000ns_tsim200ns',
#                   'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi50ns_tsim1000ns',                                    
#                   'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi100ns_tsim1000ns',
#                   'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi200ns_tsim1000ns',
#                   'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi1000ns_tsim1000ns']

# data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi50ns_tsim200ns']

data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi50ns_tsim200ns',                                    
                  'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi100ns_tsim200ns',
                  'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi200ns_tsim200ns',
                  'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi1000ns_tsim200ns']

drive_info = dict()
drive_info['drive_type'] = 'piecewise_linear'
drive_info['pwl_drive'] = [[0e-9,0e-6],[1.9e-9,0e-6],[2e-9,20e-6]]

L_di_vec = [77.5e-9,77.5e-9,77.5e-9,77.5e-9,775e-9,775e-9,775e-9,775e-9]
tau_di_vec = [50e-9,100e-9,200e-9,1000e-9,50e-9,100e-9,200e-9,1000e-9]
dt_vec = [0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9]
tf_vec = [200e-9,200e-9,200e-9,200e-9,1000e-9,1000e-9,1000e-9,1000e-9]
 
master_error_plot_name = 'vary_tau_di'
 
#call main sweep function 
t_tot = time.time()        
best_params__vary_taudi, error_mat_master__mu1_mu2__vary_taudi, error_mat_master__mu3_mu4__vary_taudi = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary tau_di')

#%% constant drive, vary I_drive

# data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi77.5ms_tsim200ns']

data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi77.5ms_tsim200ns',                                    
                  'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi775ms_tsim200ns']

drive_info = dict()
drive_info['drive_type'] = 'piecewise_linear'
drive_info['pwl_drive'] = [[0e-9,0e-6],[1.9e-9,0e-6],[2e-9,20e-6]]

L_di_vec = [77.5e-9,775e-9]
tau_di_vec = [77.5e-3,775e-3]
dt_vec = [0.1e-9,0.1e-9]
tf_vec = [200e-9,200e-9]
 
master_error_plot_name = 'vary_I_drive'
 
#call main sweep function 
t_tot = time.time()        
best_params__vary_Idrive_20uA, error_mat_master__mu1_mu2__vary_Idrive_20uA, error_mat_master__mu3_mu4__vary_Idrive_20uA = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary I_drive_20uA')

# data_file_list = ['dend__cnst_drv__Idrv25uA_Ldi77.5nH_taudi77.5ms_tsim200ns']

data_file_list = ['dend__cnst_drv__Idrv25uA_Ldi77.5nH_taudi77.5ms_tsim200ns',
                  'dend__cnst_drv__Idrv25uA_Ldi775nH_taudi775ms_tsim200ns']

drive_info['pwl_drive'] = [[0e-9,0e-6],[1.9e-9,0e-6],[2e-9,25e-6]]
 
#call main sweep function 
t_tot = time.time()        
best_params__vary_Idrive_25uA, error_mat_master__mu1_mu2__vary_Idrive_25uA, error_mat_master__mu3_mu4__vary_Idrive_25uA = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary I_drive_25uA')

data_file_list = ['dend__cnst_drv__Idrv30uA_Ldi77.5nH_taudi77.5ms_tsim200ns',
                  'dend__cnst_drv__Idrv30uA_Ldi775nH_taudi775ms_tsim200ns']

drive_info['pwl_drive'] = [[0e-9,0e-6],[1.9e-9,0e-6],[2e-9,30e-6]]
 
#call main sweep function 
t_tot = time.time()        
best_params__vary_Idrive_30uA, error_mat_master__mu1_mu2__vary_Idrive_30uA, error_mat_master__mu3_mu4__vary_Idrive_30uA = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary I_drive_30uA')

#%% linear ramp

# data_file_list = ['dend__lin_ramp__Lsi7.75nH']

data_file_list = ['dend__lin_ramp__Lsi7.75nH',                                    
                  'dend__lin_ramp__Lsi77.5nH',
                  'dend__lin_ramp__Lsi775nH',
                  'dend__lin_ramp__Lsi7.75uH']

drive_info = dict()
drive_info['drive_type'] = 'linear_ramp'
drive_info['pwl_drive'] = [[0e-9,15e-6],[100e-9,30e-6]]

L_di_vec = [7.75e-9,77.5e-9,775e-9,7.75e-6]
tau_di_vec = []
for L in L_di_vec:
    tau_di_vec.append(L/1e-6)
dt_vec = [0.1e-9,0.1e-9,0.1e-9,0.1e-9]
tf_vec = [200e-9,200e-9,200e-9,200e-9]
 
master_error_plot_name = 'lin_ramp'
 
#call main sweep function 
t_tot = time.time()        
best_params__lin_ramp, error_mat_master__mu1_mu2__lin_ramp, error_mat_master__mu3_mu4__lin_ramp = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for lin_ramp')

#%% square pulse sequence

# data_file_list = ['dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi50ns_tsim_200ns']

data_file_list = ['dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi50ns_tsim_200ns',
                  'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi100ns_tsim_200ns',
                  'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi200ns_tsim_200ns',
                  'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi1000ns_tsim_200ns',
                  'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi77.5nH_taudi50ns_tsim_200ns',
                  'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi77.5nH_taudi100ns_tsim_200ns',
                  'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi77.5nH_taudi200ns_tsim_200ns',
                  'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi77.5nH_taudi1000ns_tsim_200ns']

# data_file_list = ['dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi50ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi100ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi200ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75nH_taudi1000ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi77.5nH_taudi50ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi77.5nH_taudi100ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi77.5nH_taudi200ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi77.5nH_taudi1000ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi775nH_taudi50ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi775nH_taudi100ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi775nH_taudi200ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi775nH_taudi1000ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75uH_taudi50ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75uH_taudi100ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75uH_taudi200ns_tsim_200ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per20ns_Ldi7.75uH_taudi1000ns_tsim_200ns']

drive_info = dict()
drive_info['drive_type'] = 'sq_pls_trn'
sq_pls_trn_params = dict()
sq_pls_trn_params['t_start'] = 5e-9
sq_pls_trn_params['t_rise'] = 10e-12
sq_pls_trn_params['t_pulse'] = 10e-9
sq_pls_trn_params['t_fall'] = 10e-12
sq_pls_trn_params['t_period'] = 20e-9
sq_pls_trn_params['value_off'] = 0
sq_pls_trn_params['value_on'] = 20e-6
drive_info['sq_pls_trn_params'] = sq_pls_trn_params

L_di_vec = [7.75e-9,7.75e-9,7.75e-9,7.75e-9,77.5e-9,77.5e-9,77.5e-9,77.5e-9,775e-9,775e-9,775e-9,775e-9,7.75e-6,7.75e-6,7.75e-6,7.75e-6]
tau_di_vec = [50e-9,100e-9,200e-9,1000e-9,50e-9,100e-9,200e-9,1000e-9,50e-9,100e-9,200e-9,1000e-9,50e-9,100e-9,200e-9,1000e-9]
dt_vec = [0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9,0.1e-9]
# dt_vec = [0.01e-9,0.01e-9,0.01e-9,0.01e-9,0.01e-9,0.01e-9,0.01e-9,0.01e-9]
tf_vec = [200e-9,200e-9,200e-9,200e-9,200e-9,200e-9,200e-9,200e-9]
 
master_error_plot_name = 'sq_pls_trn_per20ns'
 
#call main sweep function 
t_tot = time.time()        
best_params__sq_pls_trn_per20ns, error_mat_master__mu1_mu2__sq_pls_trn_per20ns, error_mat_master__mu3_mu4__sq_pls_trn_per20ns = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for sq_pls_trn_per20ns')

# data_file_list = ['dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi7.75nH_taudi50ns_tsim_1000ns']

# data_file_list = ['dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi7.75nH_taudi50ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi7.75nH_taudi100ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi7.75nH_taudi200ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi7.75nH_taudi1000ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi77.5nH_taudi50ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi77.5nH_taudi100ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi77.5nH_taudi200ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi77.5nH_taudi1000ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi775nH_taudi50ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi775nH_taudi100ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi775nH_taudi200ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi775nH_taudi1000ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi7.75uH_taudi50ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi7.75uH_taudi100ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi7.75uH_taudi200ns_tsim_1000ns',
#                   'dend__sq_pls_seq__amp20uA_dur10ns_per100ns_Ldi7.75uH_taudi1000ns_tsim_1000ns']

# drive_info = dict()
# drive_info['drive_type'] = 'sq_pls_trn'
# sq_pls_trn_params = dict()
# sq_pls_trn_params['t_start'] = 5e-9
# sq_pls_trn_params['t_rise'] = 10e-12
# sq_pls_trn_params['t_pulse'] = 10e-9
# sq_pls_trn_params['t_fall'] = 10e-12
# sq_pls_trn_params['t_period'] = 100e-9
# sq_pls_trn_params['value_off'] = 0
# sq_pls_trn_params['value_on'] = 20e-6
# drive_info['sq_pls_trn_params'] = sq_pls_trn_params

# L_di_vec = [7.75e-9,7.75e-9,7.75e-9,7.75e-9,77.5e-9,77.5e-9,77.5e-9,77.5e-9,775e-9,775e-9,775e-9,775e-9,7.75e-6,7.75e-6,7.75e-6,7.75e-6]
# tau_di_vec = [50e-9,100e-9,200e-9,1000e-9,50e-9,100e-9,200e-9,1000e-9,50e-9,100e-9,200e-9,1000e-9,50e-9,100e-9,200e-9,1000e-9]
# dt_vec = [0.01e-9,0.01e-9,0.01e-9,0.01e-9,0.01e-9,0.01e-9,0.01e-9,0.01e-9]
# tf_vec = [1000e-9,1000e-9,1000e-9,1000e-9,1000e-9,1000e-9,1000e-9,1000e-9]
 
# master_error_plot_name = 'sq_pls_trn_per100ns'
 
# #call main sweep function 
# t_tot = time.time()        
# best_params__sq_pls_trn_per100ns, error_mat_master__mu1_mu2__sq_pls_trn_per100ns, error_mat_master__mu3_mu4__sq_pls_trn_per100ns = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
# elapsed = time.time() - t_tot
# print('soen_sim duration = '+str(elapsed)+' s for sq_pls_trn_per100ns')

#%% exponential pulse sequence

# data_file_list = ['dend__exp_pls_seq__amp20uA_tauin50ns_per100ns_Ldi7.75nH_taudi100ns']

# data_file_list = ['dend__exp_pls_seq__amp20uA_tauin50ns_per100ns_Ldi7.75nH_taudi100ns',
#                   'dend__exp_pls_seq__amp20uA_tauin50ns_per100ns_Ldi7.75nH_taudi1000ns',
#                   'dend__exp_pls_seq__amp20uA_tauin50ns_per100ns_Ldi77.5nH_taudi100ns',
#                   'dend__exp_pls_seq__amp20uA_tauin50ns_per100ns_Ldi77.5nH_taudi1000ns']

data_file_list = ['dend__exp_pls_seq__amp20uA_tauin50ns_per100ns_Ldi7.75nH_taudi100ns',
                  'dend__exp_pls_seq__amp20uA_tauin50ns_per100ns_Ldi7.75nH_taudi1000ns']

drive_info = dict()
drive_info['drive_type'] = 'exp_pls_trn'
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
drive_info['exp_pls_trn_params'] = exp_pls_trn_params

L_di_vec = [7.75e-9,7.75e-9,77.5e-9,77.5e-9]
tau_di_vec = [100e-9,1000e-9,100e-9,1000e-9]
# dt_vec = [0.1e-9,0.1e-9,0.1e-9,0.1e-9]
dt_vec = [0.01e-9,0.01e-9,0.01e-9,0.01e-9]
tf_vec = [1000e-9,1000e-9,1000e-9,1000e-9]
 
master_error_plot_name = 'exp_pls_trn_per100ns'
 
#call main sweep function 
t_tot = time.time()        
best_params__exp_pls_trn_per100ns, error_mat_master__mu1_mu2__exp_pls_trn_per100ns, error_mat_master__mu3_mu4__exp_pls_trn_per100ns = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for exp_pls_trn_per100ns')

# # data_file_list = ['dend__exp_pls_seq__amp20uA_tauin50ns_per1000ns_Ldi7.75nH_taudi100ns']

# data_file_list = ['dend__exp_pls_seq__amp20uA_tauin50ns_per1000ns_Ldi7.75nH_taudi100ns',
#                   'dend__exp_pls_seq__amp20uA_tauin50ns_per1000ns_Ldi7.75nH_taudi1000ns',
#                   'dend__exp_pls_seq__amp20uA_tauin50ns_per1000ns_Ldi77.5nH_taudi100ns',
#                   'dend__exp_pls_seq__amp20uA_tauin50ns_per1000ns_Ldi77.5nH_taudi1000ns']

# exp_pls_trn_params['t_r1_period'] = 1000e-9
# drive_info['exp_pls_trn_params'] = exp_pls_trn_params
 
# master_error_plot_name = 'exp_pls_trn_per1000ns'
 
# #call main sweep function 
# t_tot = time.time()        
# best_params__exp_pls_trn_per1000ns, error_mat_master__mu1_mu2__exp_pls_trn_per1000ns, error_mat_master__mu3_mu4__exp_pls_trn_per1000ns = dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
# elapsed = time.time() - t_tot
# print('soen_sim duration = '+str(elapsed)+' s for exp_pls_trn_per1000ns')

#%% assemble and plot error_mat_masters

error_mat_master__mu1_mu2 = ( error_mat_master__mu1_mu2__no_leak__vary_Ldi
                             +error_mat_master__mu1_mu2__vary_taudi
                             +error_mat_master__mu1_mu2__vary_Idrive_20uA
                             +error_mat_master__mu1_mu2__vary_Idrive_25uA
                             +error_mat_master__mu1_mu2__vary_Idrive_30uA
                             +error_mat_master__mu1_mu2__sq_pls_trn_per20ns
                             +error_mat_master__mu1_mu2__exp_pls_trn_per100ns )

error_mat_master__mu3_mu4 = ( error_mat_master__mu3_mu4__no_leak__vary_Ldi
                             +error_mat_master__mu3_mu4__vary_taudi
                             +error_mat_master__mu3_mu4__vary_Idrive_20uA
                             +error_mat_master__mu3_mu4__vary_Idrive_25uA
                             +error_mat_master__mu3_mu4__vary_Idrive_30uA
                             +error_mat_master__mu3_mu4__sq_pls_trn_per20ns
                             +error_mat_master__mu3_mu4__exp_pls_trn_per100ns )

# error_mat_master__mu1_mu2 = ( error_mat_master__mu1_mu2__no_leak__vary_Ldi
#                              +error_mat_master__mu1_mu2__vary_taudi
#                              +error_mat_master__mu1_mu2__vary_Idrive_20uA
#                              +error_mat_master__mu1_mu2__vary_Idrive_25uA
#                              +error_mat_master__mu1_mu2__vary_Idrive_30uA
#                              +error_mat_master__mu1_mu2__sq_pls_trn_per20ns
#                              +error_mat_master__mu1_mu2__sq_pls_trn_per100ns
#                              +error_mat_master__mu1_mu2__exp_pls_trn_per100ns
#                              +error_mat_master__mu1_mu2__exp_pls_trn_per1000ns )

# error_mat_master__mu3_mu4 = ( error_mat_master__mu3_mu4__no_leak__vary_Ldi
#                              +error_mat_master__mu3_mu4__vary_taudi
#                              +error_mat_master__mu3_mu4__vary_Idrive_20uA
#                              +error_mat_master__mu3_mu4__vary_Idrive_25uA
#                              +error_mat_master__mu3_mu4__vary_Idrive_30uA
#                              +error_mat_master__mu3_mu4__sq_pls_trn_per20ns
#                              +error_mat_master__mu3_mu4__sq_pls_trn_per100ns
#                              +error_mat_master__mu3_mu4__exp_pls_trn_per100ns
#                              +error_mat_master__mu3_mu4__exp_pls_trn_per1000ns )

ind_best = np.where(error_mat_master__mu1_mu2 == np.amin(error_mat_master__mu1_mu2))#error_mat.argmin()
amp_best_mu12 = amp_vec[ind_best[0]][0]
mu1_best = mu1_vec[ind_best[1]][0]
mu2_best = mu2_vec[ind_best[2]][0]
ind_best = np.where(error_mat_master__mu3_mu4 == np.amin(error_mat_master__mu3_mu4))#error_mat.argmin()
amp_best_mu34 = amp_vec[ind_best[0]][0]
mu3_best = mu3_vec[ind_best[1]][0]
mu4_best = mu4_vec[ind_best[2]][0]
print('\n\namp_best_mu12 = {}'.format(amp_best_mu12))
print('mu1_best = {}'.format(mu1_best))
print('mu2_best = {}\n\n'.format(mu2_best))
print('\n\namp_best_mu34 = {}'.format(amp_best_mu34))
print('mu3_best = {}'.format(mu3_best))
print('mu4_best = {}\n\n'.format(mu4_best))

pre_str = 'dend_fit_master'
title_string = '{}; amp_best_mu12 = {:2.2f}, amp_best_mu34 = {:2.2f}, mu1_best = {:1.2f}, mu2_best = {:1.2f}, mu3_best = {:1.2f}, mu4_best = {:1.2f}'.format(pre_str,amp_best_mu12,amp_best_mu34,mu1_best,mu2_best,mu3_best,mu4_best)    
for aa in range(len(amp_vec)):
    save_str_1 = '{}__master_error__mu1_mu2__amp_{:2.2f}'.format(pre_str,amp_vec[aa])
    save_str_2 = '{}__master_error__mu3_mu4__amp_{:2.2f}'.format(pre_str,amp_vec[aa])
    plot_error_mat(error_mat_master__mu1_mu2[aa,:,:],mu1_vec,mu2_vec,'mu1','mu2','amp = {}'.format(amp_vec[aa]),title_string,save_str_1)
    plot_error_mat(error_mat_master__mu3_mu4[aa,:,:],mu3_vec,mu4_vec,'mu3','mu4','amp = {}'.format(amp_vec[aa]),title_string,save_str_2)

#%% load data
# load_string = 'session_data__wr_fits__no_leak__vary_L_di__finding_amp_mu1_mu2+mu3_mu4__2020-04-08_14-29-15.dat'
# data_array_imported = load_session_data(load_string)

