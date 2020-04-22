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
I_sy_vec = np.arange(22,39+dI,dI)#39
L_si = 775e-9
t_spike = 5e-9

data_file_list = []
num_files = len(I_sy_vec)
for ii in range(num_files):
    data_file_list.append('syn_Ispd20.00uA_Isy{:2.2f}uA_Lsi0775.00nH_tausi0775ms_dt00.2ps_tsim0035ns.dat'.format(I_sy_vec[ii]))

I_si_vec = np.zeros([num_files])
t_peak_vec = np.zeros([num_files])
for ii in range(num_files):
    
    print('ii = {:d} of {:d}'.format(ii+1,num_files))
    
    directory = 'wrspice_data/fitting_data'
    file_name = data_file_list[ii]
    data_dict = read_wr_data(directory+'/'+file_name)
    
    #plot wr time traces
    data_to_plot = ['L0#branch','L3#branch','v(7)']
    # plot_save_string = file_name
    # plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)
    
    # find peak of I_si
    time_vec = data_dict['time']    
    initial_ind = (np.abs(time_vec-4e-9)).argmin()
    final_ind = (np.abs(time_vec-34e-9)).argmin()
    time_vec = time_vec[initial_ind:final_ind]-t_spike
    
    I_si = data_dict['L3#branch']
    I_si = I_si[initial_ind:final_ind]
        
    I_si_peak, _ = find_peaks(I_si, distance = len(I_si))#, height = 10e-9
    I_si_vec[ii] = I_si[I_si_peak]
    t_peak_vec[ii] = time_vec[I_si_peak]
    
    # fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
    # fig.suptitle(file_name)   
    # ax.plot(time_vec*1e9,I_si*1e3, '-', label = '$I_{si}$')      
    # ax.plot(time_vec[I_si_peak]*1e9,I_si[I_si_peak]*1e3, 'x')
    # ax.set_xlabel(r'Time [ns]')
    # ax.set_ylabel(r'Current [$\mu$A]')
    # ax.legend()
    # plt.show()
        
#%%

I_sy_vec_dense = np.linspace(I_sy_vec[0],I_sy_vec[-1],1000)

t_peak_fit = np.polyfit(I_sy_vec,t_peak_vec*1e9,2)
t_peak_vec_dense = np.polyval(t_peak_fit,I_sy_vec_dense)


fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
fig.suptitle('Rise time versus synaptic bias current') 
# plt.title(r) 
 
ax.plot(I_sy_vec,t_peak_vec*1e9, 'o-', label = 'WR data')      
ax.plot(I_sy_vec_dense,t_peak_vec_dense, '-', label = 'Polyfit')
ax.set_xlabel(r'$I_{sy}$ [$\mu$A]')
ax.set_ylabel(r'$t_{rise}$ [ns]')
ax.set_title('tau_rise vs I_sy; tau_rise = {:f}Isy^2+{:f}Isy+{:f} (I_0, Isy in uA)'.format(t_peak_fit[0],t_peak_fit[1],t_peak_fit[2]))
ax.legend()

plt.show()
fig.savefig('figures/t_peak_fit.png') 
    