#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages, plot_wr_comparison__synapse, plot_error_mat
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3, chi_squared_error
from _functions__more import synapse_model__parameter_sweep
from util import physical_constants
from soen_sim import input_signal, synapse
p = physical_constants()

# plt.close('all')

#%%
    
gamma1_vec = np.linspace(0.05,2.5,20)
gamma2_vec = np.linspace(0.05,2.5,20)
gamma3_vec = np.linspace(0.1,1,20)

spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]

dt = 1e-9
tf = 1e-6

#%% vary Isy
I_sy_vec = [23e-6,28e-6,33e-6,38e-6]
L_si = 77.5e-9
tau_si = 250e-9

master_error_plot_name = 'vary_I_sy'

data_file_list = []
num_files = len(I_sy_vec)
for ii in range(num_files):
    data_file_list.append('syn_Ispd20.00uA_Isy{:04.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii]*1e6,L_si*1e9,tau_si*1e9))

#call main sweep function 
t_tot = time.time()
best_params__vary_Isy, error_mat_master__gamma1_gamma2__vary_Isy, error_mat_master__gamma3__vary_Isy = synapse_model__parameter_sweep(data_file_list,I_sy_vec,L_si*np.ones([len(I_sy_vec)]),tau_si*np.ones([len(I_sy_vec)]),dt,tf,spike_times,gamma1_vec,gamma2_vec,gamma3_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary I_sy')


#%% vary Lsi
I_sy = 28e-6
L_si_vec = [7.75e-9,77.5e-9,775e-9,7.75e-6]
tau_si = 250e-9

master_error_plot_name = 'vary_L_si'

data_file_list = []
num_files = len(L_si_vec)
for ii in range(num_files):
    data_file_list.append('syn_Ispd20.00uA_Isy{:05.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy*1e6,L_si_vec[ii]*1e9,tau_si*1e9))

#call main sweep function 
t_tot = time.time()
best_params__vary_Lsi, error_mat_master__gamma1_gamma2__vary_Lsi, error_mat_master__gamma3__vary_Lsi = synapse_model__parameter_sweep(data_file_list,I_sy*np.ones([len(L_si_vec)]),L_si_vec,tau_si*np.ones([len(L_si_vec)]),dt,tf,spike_times,gamma1_vec,gamma2_vec,gamma3_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary L_si')

#%%vary tau_si
I_sy = 33e-6
L_si = 775e-9
tau_si_vec = [10e-9,50e-9,250e-9,1.25e-6]

master_error_plot_name = 'vary_tau_si'

data_file_list = []
num_files = len(L_si_vec)
for ii in range(num_files):
    data_file_list.append('syn_Ispd20.00uA_Isy{:05.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy*1e6,L_si*1e9,tau_si_vec[ii]*1e9))

#call main sweep function 
t_tot = time.time()
best_params__vary_tausi, error_mat_master__gamma1_gamma2__vary_tausi, error_mat_master__gamma3__vary_tausi = synapse_model__parameter_sweep(data_file_list,I_sy*np.ones([len(tau_si_vec)]),L_si*np.ones([len(tau_si_vec)]),tau_si_vec,dt,tf,spike_times,gamma1_vec,gamma2_vec,gamma3_vec,master_error_plot_name)
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary tau_si')

#%% assemble and plot error_mat_masters

error_mat_master__gamma1_gamma2 = ( error_mat_master__gamma1_gamma2__vary_Isy
                                  + error_mat_master__gamma1_gamma2__vary_Lsi
                                  + error_mat_master__gamma1_gamma2__vary_tausi)

error_mat_master__gamma3 = ( error_mat_master__gamma3__vary_Isy
                                  + error_mat_master__gamma3__vary_Lsi
                                  + error_mat_master__gamma3__vary_tausi)

ind_best = np.where(error_mat_master__gamma1_gamma2 == np.amin(error_mat_master__gamma1_gamma2))#error_mat.argmin()
gamma1_best = gamma1_vec[ind_best[0]][0]
gamma2_best = gamma2_vec[ind_best[1]][0]

ind_best = error_mat_master__gamma3.argmin()
gamma3_best = gamma3_vec[ind_best]
        
title_string = 'Master Error; gamma1_best = {:7.4f}, gamma2_best = {:7.4f}, gamma3_best = {:7.4f}'.format(gamma1_best,gamma2_best,gamma3_best)    
save_str_1 = 'sy__master_error__gamma1_gamma2'
save_str_2 = 'sy__master_error__gamma3'

plot_error_mat(error_mat_master__gamma1_gamma2[:,:],gamma1_vec,gamma2_vec,'gamma1','gamma2',title_string,save_str_1)

fig, ax = plt.subplots(1,1)
fig.suptitle(title_string)
ax.plot(gamma3_vec,error_mat_master__gamma3)    
ax.set_xlabel(r'$\gamma_3$')
ax.set_ylabel(r'Error')
plt.show()
fig.savefig('figures/'+save_str_2+'.png') 
