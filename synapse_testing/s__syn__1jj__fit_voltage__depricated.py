#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pickle
import numpy.matlib

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks_and_average_voltage_vs_time__1jj, plot_fq_peaks_and_average_voltage_vs_Isf__1jj, plot_Vsf_vs_Isf_fit, plot_Vsf_vs_Isf, plot_rate_vs_Isf
from _functions import save_session_data, load_session_data, read_wr_data, syn_1jj_Vsf_vs_Isf_fit
from util import physical_constants
p = physical_constants()

# plt.close('all')

#%% load wr data, find peaks, find rates

I_sy = 40#uA

dI = 0.25
I_drive_list = np.concatenate([np.array([0.01,0.1]),np.arange(0.5,20+dI,dI)])

#%%
    
j_sf_ifi_array = [] # array of inter-fluxon intervals at the synaptic firing junction
j_sf_rate_array = [] # array of fluxon production rate at the synaptic firing junction
j_jtl_ifi_array = [] # array of inter-fluxon intervals at the jtl junction
j_jtl_rate_array = [] # array of fluxon production rate at the jtl junction
j_si_ifi_array = [] # array of inter-fluxon intervals at the synaptic integration loop junction
j_si_rate_array = [] # array of fluxon production rate at the synaptic integration loop junction

I_si_array = [] # array of values of I_si
I_sf_array = [] # array of values of I_sf
# V_sf_avg_array = [] # array of values of V_sf
# V_sf_avg_time_vec_array = [] # array of time vectors associated with average voltages

num_drives = len(I_drive_list) 
directory = 'wrspice_data/fitting_data/1jj'
mu1_vec = np.zeros([num_drives])
mu2_vec = np.zeros([num_drives])
V0_vec = np.zeros([num_drives])
Vsf_array = []
Isf_array = []
for ii in range(num_drives): # range(10): # 
    
    print('ii = {:d} of {:d}'.format(ii+1,num_drives))        
    
    file_name = 'syn_1jj_cnst_drv_Isy{:5.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0077.5ms_dt00.01ps.dat'.format(I_sy,I_drive_list[ii])
    data_dict = read_wr_data(directory+'/'+file_name)
    
    # find peaks for each jj
    time_vec = data_dict['time']
    V_sf = data_dict['v(3)']
    
    # initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
    # final_ind = (np.abs(time_vec-t_sim_list[ii]*1e-9)).argmin()

    # time_vec = time_vec[initial_ind:final_ind]
    # j_sf = j_sf[initial_ind:final_ind]
    # j_jtl = j_jtl[initial_ind:final_ind]
    # j_si = j_si[initial_ind:final_ind]
  
    j_sf_peaks, _ = find_peaks(V_sf, height = 200e-6)
   
    # find inter-fluxon intervals and fluxon generation rates for each JJ
    I_si = data_dict['L1#branch']
    # I_si = I_si[initial_ind:final_ind]
    I_drive = data_dict['L0#branch']
    # I_drive = I_drive[initial_ind:final_ind]     
          
    j_si_ifi = np.diff(time_vec[j_sf_peaks])
    
    j_si_rate = 1/j_si_ifi

    j_si_ifi_array.append(j_si_ifi)
    j_si_rate_array.append(j_si_rate)
    
    I_si_array.append(I_si[:])
    I_sf = I_sy+1e6*I_drive[:]-1e6*I_si[:]
    I_sf_array.append(I_sf)
    
    V_sf_avg = np.zeros([len(j_sf_peaks)-1])
    I_sf_avg = np.zeros([len(j_sf_peaks)-1])
    time_vec_avg = np.zeros([len(j_sf_peaks)-1])
    for jj in range(len(j_sf_peaks)-1):
        ind_vec = np.arange(j_sf_peaks[jj],j_sf_peaks[jj+1],1)
        V_sf_avg[jj] = 1e6*np.sum(V_sf[ind_vec])/len(ind_vec) 
        I_sf_avg[jj] = np.sum(I_sf[ind_vec])/len(ind_vec)
        time_vec_avg[jj] = 1e9*np.sum(time_vec[ind_vec])/len(ind_vec)
    # V_sf_avg_array.append(V_sf_avg[:])
    # V_sf_avg_time_vec_array.append(time_vec_avg[:])
    Vsf_array.append(V_sf_avg)
    Isf_array.append(I_sf_avg)
    
    # plot fq peaks and average voltage 
    # plot_fq_peaks_and_average_voltage_vs_time__1jj(time_vec,V_sf,j_sf_peaks,time_vec_avg,V_sf_avg,file_name)
    # plot_fq_peaks_and_average_voltage_vs_Isf__1jj(I_sf,V_sf,j_sf_peaks,I_sf_avg,V_sf_avg,file_name)
    
    # fit average V_sf vs average I_sf
    # popt, pcov = curve_fit(syn_1jj_Vsf_vs_Isf_fit, I_sf_avg, V_sf_avg, bounds = ([1.0, 0.1, 10], [3.0, 1.0, 1000]))
    # mu1_vec[ii] = popt[0]
    # mu2_vec[ii] = popt[1]
    # V0_vec[ii] = popt[2]
    # mu1_vec[ii] = 2
    # mu2_vec[ii] = 0.5
    
    # plot fits
    # V_sf_fit = syn_1jj_Vsf_vs_Isf_fit(I_sf_avg,mu1_vec[ii],mu2_vec[ii],V0_vec[ii])
    # Ic = 40
    # Rn = 4.125
    # V_sf_fixed = syn_1jj_Vsf_vs_Isf_fit(I_sf_avg,2,0.5,Ic*Rn)
    # plot_Vsf_vs_Isf_fit(I_sf,V_sf,j_sf_peaks,I_sf_avg,V_sf_avg,V_sf_fit,V_sf_fixed,file_name,mu1_vec[ii],mu2_vec[ii],V0_vec[ii])

#%% plot all

popt, pcov = curve_fit(syn_1jj_Vsf_vs_Isf_fit, Isf_array[-1], Vsf_array[-1], bounds = ([0.1,0.1,10], [10,10.0,1000]))
mu1 = popt[0]
mu2 = popt[1]
V0 = popt[2]

V_sf_fit = syn_1jj_Vsf_vs_Isf_fit(Isf_array[-1],mu1,mu2,V0)
plot_Vsf_vs_Isf(Vsf_array,Isf_array,V_sf_fit,mu1,mu2,V0)
plot_rate_vs_Isf(j_si_rate_array,Isf_array,V_sf_fit)



