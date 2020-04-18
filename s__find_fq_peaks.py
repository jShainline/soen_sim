#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks

from util import physical_constants, color_dictionary
from _functions import read_wr_data, t_fq, V_fq
from _plotting import plot_wr_data, plot_wr_data__currents_and_voltages, plot_fq_peaks, plot_fq_peaks__dt_vs_bias

p = physical_constants()
colors = color_dictionary()

#%%
directory = 'wrspice_data'
file_name = 'dend__cnst_drv__Idrv19.0uA_Ib35.0uA_Ldi0077.5nH_taudi0077.5ms_tsim_100ns.dat'
data_to_plot = ['L3#branch','L4#branch','L8#branch','v(3)','v(4)','v(5)']#'L0#branch','L1#branch','L2#branch',
plot_save_string = False

data_dict = read_wr_data(directory+'/'+file_name)
data_dict['file_name'] = file_name
plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)

#%% plot fq peaks
time_vec = data_dict['time']

V_j_df = data_dict['v(3)']
V_j_jtl = data_dict['v(4)']
V_j_di = data_dict['v(5)']

I_j_df = data_dict['L6#branch']
I_j_jtl = data_dict['L7#branch']
I_j_di = data_dict['L8#branch']

peaks_j_df, _ = find_peaks(V_j_df, height = 1e-4, distance = 20)
peaks_j_jtl, _ = find_peaks(data_dict['v(4)'], height = 1e-4, distance = 20)
peaks_j_di, _ = find_peaks(data_dict['v(5)'], height = 1e-4, distance = 20)
# peaks_jdf, _ = find_peaks(x, distance = 20)
plot_fq_peaks(time_vec,data_dict['v(3)'],peaks_j_df)
plot_fq_peaks(time_vec,data_dict['v(4)'],peaks_j_jtl)
plot_fq_peaks(time_vec,data_dict['v(5)'],peaks_j_di)

#%% calculate time between peaks

dt_fq_df = np.diff(time_vec[peaks_j_df])
dt_fq_jtl = np.diff(time_vec[peaks_j_jtl])
dt_fq_di = np.diff(time_vec[peaks_j_di])

fig, ax = plt.subplots(1,1)
ax.plot(time_vec[peaks_j_df[0:-1]]*1e9,dt_fq_df*1e12, '-') #, label = 'data'   
ax.set_xlabel(r'Time [ns]')
ax.set_ylabel(r'$\delta t$ [ps]')
plt.show()


#%% plot dt vs current

Ic = 40e-6
I_j_df__avg = np.zeros([len(peaks_j_df)-1,1])
time_vec__avg = np.zeros([len(peaks_j_df)-1,1])
for ii in range(len(peaks_j_df)-1):
    I_j_df__avg[ii] = np.sum(I_j_df[peaks_j_df[ii]:peaks_j_df[ii+1]])/(peaks_j_df[ii+1]-peaks_j_df[ii])
    time_vec__avg[ii] = np.sum(time_vec[peaks_j_df[ii]:peaks_j_df[ii+1]])/(peaks_j_df[ii+1]-peaks_j_df[ii])
    
fig, ax = plt.subplots(1,1)
ax.plot(time_vec[peaks_j_df[0:-1]]*1e9,I_j_df__avg*1e6, '-') #, label = 'data'   
ax.set_xlabel(r'Time [ns]')
ax.set_ylabel(r'$\bar{I}$ [$\mu$A]')
plt.show()
    
#%%
fig, ax = plt.subplots(1,1)
ax.plot(I_j_df__avg*1e6,dt_fq_df*1e12, '-') #, label = 'data'   
ax.set_xlabel(r'$\bar{I}$ [$\mu$A]')
ax.set_ylabel(r'$\delta t$ [ps]')
plt.show()

#%%
R = 4.125
pf = (p['Phi0']/(Ic*R))
mu1_guess = 1
mu2_guess = 0.5

dI = 100e-9
current_vec = np.arange(Ic+dI,Ic+10e-6+100e-9,dI)
dt_vec = pf*((current_vec/Ic)**mu1_guess-1)**(-mu2_guess)

fig, ax = plt.subplots(1,1)
ax.plot(current_vec*1e6,dt_vec*1e12, '-') #, label = 'data'   
ax.set_xlabel(r'$I$ [$\mu$A]')
ax.set_ylabel(r'$\delta t$ [ps]')
plt.show()

#%% plot expected functional form
Ic = 40e-6
R = 4.125
current_vec = np.arange(Ic+dI,Ic+10e-6+100e-9,dI)
mu1_vec = [1,2]
mu2_vec = [0.5,1]

# color_list = ['blue_1','blue_2','blue_3','red_1','red_2','red_3','green_1','green_2','green_3']
color_list = ['blue_3','red_3','green_3','yellow_3']

fig, ax = plt.subplots(1,1)
for ii in range(len(mu1_vec)):
    for jj in range(len(mu2_vec)):
        ax.plot(current_vec*1e6,t_fq(current_vec,Ic,R,mu1_vec[ii],mu2_vec[jj])*1e12, '-', color = colors[color_list[ii*len(mu2_vec)+jj]], label = '$\mu_1$ = {}, $\mu_2$ = {}'.format(mu1_vec[ii],mu2_vec[jj])   ) #
ax.set_xlabel(r'Current [$\mu$A]')
ax.set_ylabel(r'$\delta t$ [ps]')
ax.legend()
plt.show()

fig, ax = plt.subplots(1,1)
for ii in range(len(mu1_vec)):
    for jj in range(len(mu2_vec)):
        ax.plot(current_vec*1e6,V_fq(current_vec,Ic,R,mu1_vec[ii],mu2_vec[jj])*1e3, '-', color = colors[color_list[ii*len(mu2_vec)+jj]], label = '$\mu_1$ = {}, $\mu_2$ = {}'.format(mu1_vec[ii],mu2_vec[jj])   ) #
ax.set_xlabel(r'Current [$\mu$A]')
ax.set_ylabel(r'Voltage [mV]')
ax.legend()
plt.show()


t_fq_test = t_fq(current_vec,Ic,R,mu1_vec[ii],mu2_vec[jj])
# plot_fq_peaks__dt_vs_bias(I_j_df__avg,dt_fq_df,Ic)



# p = physical_constants()
# R = 4.125
# pf = (p['Phi0']/(Ic*R))
# 
# t_fq_guess = pf*
