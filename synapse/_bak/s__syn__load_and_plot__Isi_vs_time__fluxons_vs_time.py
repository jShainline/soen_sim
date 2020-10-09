#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages, plot_wr_comparison__synapse, plot_fq_peaks__three_jjs, plot_wr_data, plot_fq_rate__three_jjs, plot_fq_delay__three_jjs, plot_fq_isi_vs_I_si, plot_fq_ifi_and_rate_vs_time, plot_fq_rate_and_delay__three_jjs
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3, chi_squared_error
from util import physical_constants
from soen_sim import input_signal, synapse
p = physical_constants()

plt.close('all')

#%%
I_sy = 40#uA

I_drive_list = ( [1.76,1.77,1.78,1.79,1.8,#1
                  1.9,2,2.1,2.2,2.3,#2
                  2.4,2.5,2.75,3,3.5,#3
                  4,4.25,4.5,4.75,4.8,#4
                  4.85,4.9,4.95,5,5.25,#5
                  5.5,5.75,6,6.25,6.5,#6
                  6.75,7,7.25,7.5,7.75,8,8.25,8.5,8.75,#7
                  9,9.25,9.5,9.75,10,10.25,10.5,10.75,11,#8
                  11.5,12,12.5,13,13.5,#9
                  14,14.5,15,15.5,16,#10
                  16.5,17,17.5,18,18.5,#11
                  19,19.25,19.5,19.6,19.7,#12
                  19.8,19.9,20] )#13; units of uA

t_sim_list = ( [10,10,13,16,20,#1
                40,60,82,72,58,#2
                51,47,41,37,33,#3
                30,30,29,27,30,#4
                38,49,45,41,35,#5
                32,32,30,29,29,#6
                27,27,27,26,25,25,25,24,24,#7
                24,23,23,23,23,22,23,22,22,#8
                22,22,22,21,21,#9
                21,21,21,21,21,#10
                21,20,20,20,20,#11
                20,20,20,20,20,#12
                20,20,20] )#13; units of ns

#%% load wr data
I_drive = 11
t_sim = 22
     
directory = 'wrspice_data/fitting_data'
file_name = 'syn_cnst_drv_Isy{:5.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0077.5ms_tsim{:04.0f}ns_dt00.1ps.dat'.format(I_sy,I_drive,t_sim)
data_dict = read_wr_data(directory+'/'+file_name)


#%% plot wr time traces

data_to_plot = ['L0#branch','L3#branch','v(3)','v(4)','v(5)']
plot_save_string = file_name
# plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)

data_to_plot = ['L0#branch','L3#branch']
data_dict['file_name'] = file_name
# plot_wr_data(data_dict,data_to_plot,False)


#%% find and plot peaks for each jj

time_vec = data_dict['time']
j_sf = data_dict['v(3)']
j_jtl = data_dict['v(4)']
j_si = data_dict['v(5)']

  
j_sf_peaks, _ = find_peaks(j_sf, height = 200e-6)
j_jtl_peaks, _ = find_peaks(j_jtl, height = 200e-6)
j_si_peaks, _ = find_peaks(j_si, height = 200e-6)

t_lims = [[0.09,0.46],[10.09,10.46],[20.09,20.46]]
I_si = data_dict['L3#branch']
plot_fq_peaks__three_jjs(time_vec,t_lims,j_sf,j_sf_peaks,j_jtl,j_jtl_peaks,j_si,j_si_peaks,I_si,file_name)


#%% find and plot inter-fluxon intervals and fluxon generation rates for each JJ
I_si = data_dict['L3#branch']
# I_si = I_si[initial_ind:final_ind]
I_drive = data_dict['L0#branch']
# I_drive = I_drive[initial_ind:final_ind]            

j_sf_ifi = np.diff(time_vec[j_sf_peaks])
j_jtl_ifi = np.diff(time_vec[j_jtl_peaks])
j_si_ifi = np.diff(time_vec[j_si_peaks])

j_sf_rate = 1/j_sf_ifi
j_jtl_rate = 1/j_jtl_ifi
j_si_rate = 1/j_si_ifi

dt_sf_jtl = np.zeros([len(j_si_ifi)])
dt_jtl_si = np.zeros([len(j_si_ifi)])
dt_si_sf = np.zeros([len(j_si_ifi)])

for ii in range(len(j_si_ifi)):
    dt_sf_jtl[ii] = time_vec[j_jtl_peaks[ii]] - time_vec[j_sf_peaks[ii]]
    dt_jtl_si[ii] = time_vec[j_si_peaks[ii]] - time_vec[j_jtl_peaks[ii]]
    dt_si_sf[ii] = time_vec[j_sf_peaks[ii+1]] - time_vec[j_si_peaks[ii]]
    
plot_fq_rate_and_delay__three_jjs(time_vec,I_si,j_si_ifi,j_si_rate,j_si_peaks,j_sf_peaks,j_jtl_peaks,dt_sf_jtl,dt_jtl_si,dt_si_sf,file_name)
    