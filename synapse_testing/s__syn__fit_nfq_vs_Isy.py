#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3
from util import physical_constants
p = physical_constants()

plt.close('all')

#%% load wr data, determine quantities of interest

dI = 1
I_sy_vec = np.arange(22,39+dI,dI)
L_si = 775e-9

data_file_list = []
num_files = len(I_sy_vec)
for ii in range(num_files):
    data_file_list.append('syn_Ispd20.00uA_Isy{:2.2f}uA_Lsi0775.00nH_tausi0775ms_dt00.2ps_tsim0035ns.dat'.format(I_sy_vec[ii]))

master_data__I_di = np.zeros([num_files])
master_data__n_fq = np.zeros([num_files])
I_si_vec = np.zeros([num_files])
n_fq_vec = np.zeros([num_files])
for ii in range(num_files):
    
    print('ii = {:d} of {:d}'.format(ii+1,num_files))
    
    directory = 'wrspice_data/fitting_data'
    file_name = data_file_list[ii]
    data_dict = read_wr_data(directory+'/'+file_name)
    
    #plot wr time traces
    data_to_plot = ['L0#branch','L3#branch','v(7)']
    plot_save_string = file_name
    plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)
    
    # find peaks for each jj
    time_vec = data_dict['time']    
    initial_ind = (np.abs(time_vec-4e-9)).argmin()
    final_ind = (np.abs(time_vec-34e-9)).argmin()
    time_vec = time_vec[initial_ind:final_ind]
    
    I_si = data_dict['L3#branch']
    I_si = I_si[initial_ind:final_ind]
    I_si_vec[ii] = I_si[-1]
    
    phi_sf = data_dict['v(14)']
    phi_sf = phi_sf[initial_ind:final_ind]
    phi_jtl = data_dict['v(14)']
    phi_sf = phi_sf[initial_ind:final_ind]
    phi_si = data_dict['v(14)']
    phi_si = phi_si[initial_ind:final_ind]
    n_fq_vec[ii] = np.floor(phi_si[-1]/(2*np.pi))
    
    if 1 == 2:
        
        j_sf = data_dict['v(5)']
        j_sf = j_sf[initial_ind:final_ind]
        j_jtl = data_dict['v(6)']
        j_jtl = j_jtl[initial_ind:final_ind]
        j_si = data_dict['v(7)']
        j_si = j_si[initial_ind:final_ind]
        
        j_sf_peaks, _ = find_peaks(j_sf, height = 100e-6)
        j_jtl_peaks, _ = find_peaks(j_jtl, height = 100e-6)
        j_si_peaks, _ = find_peaks(j_si, height = 100e-6)
    
        fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)
        fig.suptitle(file_name)   
        ax[0].plot(time_vec*1e9,j_sf*1e3, '-', label = '$J_{sf}$')      
        ax[0].plot(time_vec[j_sf_peaks]*1e9,j_sf[j_sf_peaks]*1e3, 'x')
        ax[0].set_xlabel(r'Time [ns]')
        ax[0].set_ylabel(r'Voltage [mV]')
        ax[0].legend()
        ax[1].plot(time_vec*1e9,j_jtl*1e3, '-', label = '$J_{jtl}$')             
        ax[1].plot(time_vec[j_jtl_peaks]*1e9,j_jtl[j_jtl_peaks]*1e3, 'x')
        ax[1].set_xlabel(r'Time [ns]')
        ax[1].set_ylabel(r'Voltage [mV]')
        ax[1].legend()
        ax[2].plot(time_vec*1e9,j_si*1e3, '-', label = '$J_{si}$')             
        ax[2].plot(time_vec[j_si_peaks]*1e9,j_si[j_si_peaks]*1e3, 'x')
        ax[2].set_xlabel(r'Time [ns]')
        ax[2].set_ylabel(r'Voltage [mV]')
        ax[2].legend()
        plt.show()
        
        # find inter-fluxon intervals and fluxon generation rates for each JJ
        j_sf_ifi = np.diff(time_vec[j_sf_peaks])
        j_jtl_ifi = np.diff(time_vec[j_jtl_peaks])
        j_si_ifi = np.diff(time_vec[j_si_peaks])
        
        j_sf_rate = 1/j_sf_ifi
        j_jtl_rate = 1/j_jtl_ifi
        j_si_rate = 1/j_si_ifi
        
#%%

I_sy_vec_dense = np.linspace(I_sy_vec[0],I_sy_vec[-1],1000)

I_si_fit = np.polyfit(I_sy_vec,I_si_vec,2)
I_si_vec_dense = np.polyval(I_si_fit,I_sy_vec_dense)

n_fq_fit = np.polyfit(I_sy_vec,n_fq_vec,2)
n_fq_vec_dense = np.polyval(n_fq_fit,I_sy_vec_dense)

fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)
fig.suptitle('Effect of varying the synaptic weight') 
# plt.title(r) 
 
ax[0].plot(I_sy_vec,I_si_vec*1e6, 'o-', label = 'WR data')      
ax[0].plot(I_sy_vec_dense,I_si_vec_dense*1e6, '-', label = 'Polyfit')
ax[0].set_xlabel(r'$I_{sy}$ [$\mu$A]')
ax[0].set_ylabel(r'$I_{si}$ [$\mu$A]')
ax[0].set_title('Current added to SI loop after one synapse event; Lsi = {:3.1f}nH'.format(L_si))
ax[0].legend()

ax[1].plot(I_sy_vec,n_fq_vec, 'o-', label = 'WR data')             
ax[1].plot(I_sy_vec_dense,n_fq_vec_dense, '-', label = 'Polyfit')
ax[1].set_xlabel(r'$I_{sy}$ [$\mu$A]')
ax[1].set_ylabel(r'$n_{fq}$ [#]')
title_string = 'nfq = {:f}Isy^2 + {:f}Isy + {:f}; nfq_min = {}; nfq_max = {}'.format(n_fq_fit[0],n_fq_fit[1],n_fq_fit[2],n_fq_vec[0],n_fq_vec[-1])
ax[1].set_title('Number of flux quanta added to SI loop after one synapse event; '+format(title_string))
ax[1].legend()

plt.show()
fig.savefig('figures/n_fq_fit.png') 
    
#%% save data
# save_string = 'master_rate_matrix'
# data_array = dict()
# data_array['master_rate_matrix'] = master_rate_matrix
# data_array['I_drive_vec'] = I_drive_vec*1e-6
# data_array['I_di_list'] = I_di_list
# print('\n\nsaving session data ...')
# save_session_data(data_array,save_string)

#%% load test
# data_array_imported = load_session_data('session_data__master_rate_matrix__2020-04-17_13-11-03.dat')
# I_di_list__imported = data_array_imported['I_di_list']
# I_drive_vec__imported = data_array_imported['I_drive_vec']
# master_rate_matrix__imported = data_array_imported['master_rate_matrix']

# I_drive_sought = 23.45e-6
# I_drive_sought_ind = (np.abs(np.asarray(I_drive_vec__imported)-I_drive_sought)).argmin()
# I_di_sought = 14.552e-6
# I_di_sought_ind = (np.abs(np.asarray(I_di_list__imported[I_drive_sought_ind])-I_di_sought)).argmin()
# rate_obtained = master_rate_matrix__imported[I_drive_sought_ind][I_di_sought_ind]

# print('I_drive_sought = {:2.2f}uA, I_drive_sought_ind = {:d}\nI_di_sought = {:2.2f}uA, I_di_sought_ind = {:d}\nrate_obtained = {:3.2f}GHz'.format(I_drive_sought*1e6,I_drive_sought_ind,I_di_sought*1e6,I_di_sought_ind,rate_obtained*1e-9))
