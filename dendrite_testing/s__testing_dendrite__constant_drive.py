#%%
import numpy as np
from matplotlib import pyplot as plt
import time

# from soen_sim import input_signal, synapse, dendrite, neuron
# from _plotting import plot_dendritic_integration_loop_current, plot_wr_data, plot_wr_comparison, plot_dendritic_drive, plot_error_mat
from _functions__more import dendrite_model__parameter_sweep

plt.close('all')

#%% set up parameters that apply to all

dt = 0.1e-9
tf = 200e-9
    
num_loop = 10

amp_vec = np.logspace(np.log10(7),np.log10(200),num_loop)#np.linspace(25,25,1)#np.linspace(25,25,1)#
mu1_vec = np.linspace(0.25,2.5,num_loop)
mu2_vec = np.linspace(0.25,2.5,num_loop)
mu3_vec = np.linspace(0.25,2.5,num_loop)
mu4_vec = np.linspace(0.25,2.5,num_loop)

#%% no leak, vary L_di


# data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi7.75nH_taudi7.75ms_tsim200ns']  
                  
data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi7.75nH_taudi7.75ms_tsim200ns',                                    
                  'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi77.5ms_tsim200ns',
                  'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi775ms_tsim200ns',
                  'dend__cnst_drv__Idrv20uA_Ldi7.75uH_taudi7.75s_tsim200ns']

                  # 'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi775ms_tsim1000ns',
                  # 'dend__cnst_drv__Idrv20uA_Ldi7.75uH_taudi7.75s_tsim1000ns'

drive_info = dict()
drive_info['drive_type'] = 'piecewise_linear'
drive_info['pwl_drive'] = [[0e-9,0e-6],[1.9e-9,0e-6],[2e-9,20e-6]]

L_di_vec = [7.75e-9,77.5e-9,775e-9,7.75e-6]
tau_di_vec = [7.75e-3,77.5e-3,775e-3,7.75]
 
master_error_plot_name = 'no_leak__vary_L_di'
 
#call main sweep function 
t_tot = time.time()        
dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt,tf,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for no leak, vary L_di')

#%% load data
# load_string = 'session_data__wr_fits__no_leak__vary_L_di__finding_amp_mu1_mu2+mu3_mu4__2020-04-08_14-29-15.dat'
# data_array_imported = load_session_data(load_string)