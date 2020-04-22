#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages
from _functions import read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3
from util import physical_constants
p = physical_constants()

plt.close('all')

#%% load wr data
read_wr = True
if read_wr == True:
    directory = 'wrspice_data'
    file_name = 'dend_lin_ramp_Idrv18.0-30.0uA_Ldi0077.50nH_taudi00775ms_tsim43ns_dt00.1ps.dat'
    data_dict = read_wr_data(directory+'/'+file_name)
    data_dict['file_name'] = file_name

#%% plot wr data
data_to_plot = ['L4#branch','L3#branch','v(3)','v(4)','v(5)']#'L0#branch','L1#branch','L2#branch',
plot_save_string = False
plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)

#%% find peaks for each jj
time_vec = data_dict['time']
j_df = data_dict['v(3)']
j_jtl = data_dict['v(4)']
j_di = data_dict['v(5)']

initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
final_ind = (np.abs(time_vec-42e-9)).argmin()

time_vec = time_vec[initial_ind:final_ind]
j_df = j_df[initial_ind:final_ind]
j_jtl = j_jtl[initial_ind:final_ind]
j_di = j_di[initial_ind:final_ind]

# fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
# ax.plot(time_vec*1e9,j_df*1e3, '-', )             
# ax.set_xlabel(r'Time [ns]')
# ax.set_ylabel(r'Voltage [mV]')
# plt.show()
  
j_df_peaks, _ = find_peaks(j_df, height = [140e-6,175e-6])
j_jtl_peaks, _ = find_peaks(j_jtl, height = 200e-6)
j_di_peaks, _ = find_peaks(j_di, height = 200e-6)

# fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
# ax.plot(time_vec*1e9,j_df*1e3, '-', label = 'data')             
# ax.plot(time_vec[j_df_peaks]*1e9,j_df[j_df_peaks]*1e3, 'x', label = 'peaks')
# ax.set_xlabel(r'Time [ns]')
# ax.set_ylabel(r'Voltage [mV]')
# ax.legend()
# plt.show()

fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)
fig.suptitle(file_name)   
ax[0].plot(time_vec*1e9,j_df*1e3, '-', label = '$J_{df}$')             
ax[0].plot(time_vec[j_df_peaks]*1e9,j_df[j_df_peaks]*1e3, 'x')
ax[0].set_xlabel(r'Time [ns]')
ax[0].set_ylabel(r'Voltage [mV]')
ax[0].legend()
ax[1].plot(time_vec*1e9,j_jtl*1e3, '-', label = '$J_{jtl}$')             
ax[1].plot(time_vec[j_jtl_peaks]*1e9,j_jtl[j_jtl_peaks]*1e3, 'x')
ax[1].set_xlabel(r'Time [ns]')
ax[1].set_ylabel(r'Voltage [mV]')
ax[1].legend()
ax[2].plot(time_vec*1e9,j_di*1e3, '-', label = '$J_{di}$')             
ax[2].plot(time_vec[j_di_peaks]*1e9,j_di[j_di_peaks]*1e3, 'x')
ax[2].set_xlabel(r'Time [ns]')
ax[2].set_ylabel(r'Voltage [mV]')
ax[2].legend()
plt.show()

#%% find inter-fluxon intervals and fluxon generation rates for each JJ

I_di = data_dict['L3#branch']
I_di = I_di[initial_ind:final_ind]
I_drive = data_dict['L4#branch']
I_drive = I_drive[initial_ind:final_ind]

j_df_ifi = np.diff(time_vec[j_df_peaks])
j_jtl_ifi = np.diff(time_vec[j_jtl_peaks])
j_di_ifi = np.diff(time_vec[j_di_peaks])

j_df_rate = 1/j_df_ifi
j_jtl_rate = 1/j_jtl_ifi
j_di_rate = 1/j_di_ifi

fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False) 
fig.suptitle(file_name)
ax.plot(time_vec*1e9,I_drive*1e6, '-', label = '$I_{drive}$')              
ax.plot(time_vec*1e9,I_di*1e6, '-', label = '$I_{di}$') 
ax.set_xlabel(r'Time [ns]')
ax.set_ylabel(r'Current [$\mu$A]')
ax.legend()
plt.show()

fig, ax = plt.subplots(nrows = 2, ncols = 3, sharex = False, sharey = False) 
fig.suptitle(file_name)   

ax[0,0].plot(time_vec[j_df_peaks[0:-1]]*1e9,j_df_ifi*1e12, '-', label = '$J_{df}$ IFI')             
ax[0,0].plot(time_vec[j_jtl_peaks[0:-1]]*1e9,j_jtl_ifi*1e12, '-', label = '$J_{jtl}$ IFI')             
ax[0,0].plot(time_vec[j_di_peaks[0:-1]]*1e9,j_di_ifi*1e12, '-', label = '$J_{di}$ IFI')             
ax[0,0].set_xlabel(r'Time [ns]')
ax[0,0].set_ylabel(r'Inter-fluxon interval [ps]')
ax[0,0].legend()
   
ax[0,1].plot(I_drive[j_df_peaks[0:-1]]*1e6,j_df_ifi*1e12, '-', label = '$J_{df}$ IFI')             
ax[0,1].plot(I_drive[j_jtl_peaks[0:-1]]*1e6,j_jtl_ifi*1e12, '-', label = '$J_{jtl}$ IFI')             
ax[0,1].plot(I_drive[j_di_peaks[0:-1]]*1e6,j_di_ifi*1e12, '-', label = '$J_{di}$ IFI')             
ax[0,1].set_xlabel(r'$I_{drive}$ [$\mu$A]')
ax[0,1].legend()
   
ax[0,2].plot(I_di[j_df_peaks[0:-1]]*1e6,j_df_ifi*1e12, '-', label = '$J_{df}$ IFI')             
ax[0,2].plot(I_di[j_jtl_peaks[0:-1]]*1e6,j_jtl_ifi*1e12, '-', label = '$J_{jtl}$ IFI')             
ax[0,2].plot(I_di[j_di_peaks[0:-1]]*1e6,j_di_ifi*1e12, '-', label = '$J_{di}$ IFI')             
ax[0,2].set_xlabel(r'$I_{di}$ [$\mu$A]')
ax[0,2].legend()
ax[0,2].set_title(file_name)

  
ax[1,0].plot(time_vec[j_df_peaks[0:-1]]*1e9,j_df_rate*1e-9, '-', label = '$J_{df}$ rate')             
ax[1,0].plot(time_vec[j_jtl_peaks[0:-1]]*1e9,j_jtl_rate*1e-9, '-', label = '$J_{jtl}$ rate')             
ax[1,0].plot(time_vec[j_di_peaks[0:-1]]*1e9,j_di_rate*1e-9, '-', label = '$J_{di}$ rate')             
ax[1,0].set_xlabel(r'Time [ns]')
ax[1,0].set_ylabel(r'Fluxon generation rate [GHz]')
ax[1,0].legend()
   
ax[1,1].plot(I_drive[j_df_peaks[0:-1]]*1e6,j_df_rate*1e-9, '-', label = '$J_{df}$ rate')             
ax[1,1].plot(I_drive[j_jtl_peaks[0:-1]]*1e6,j_jtl_rate*1e-9, '-', label = '$J_{jtl}$ rate')             
ax[1,1].plot(I_drive[j_di_peaks[0:-1]]*1e6,j_di_rate*1e-9, '-', label = '$J_{di}$ rate')             
ax[1,1].set_xlabel(r'$I_{drive}$ [$\mu$A]')
ax[1,1].legend()

ax[1,2].plot(I_di[j_df_peaks[0:-1]]*1e6,j_df_rate*1e-9, '-', label = '$J_{df}$ rate')             
ax[1,2].plot(I_di[j_jtl_peaks[0:-1]]*1e6,j_jtl_rate*1e-9, '-', label = '$J_{jtl}$ rate')             
ax[1,2].plot(I_di[j_di_peaks[0:-1]]*1e6,j_di_rate*1e-9, '-', label = '$J_{di}$ rate')             
ax[1,2].set_xlabel(r'$I_{di}$ [$\mu$A]')
ax[1,2].legend()


     
# fig, ax = plt.subplots(1,1)
# error = ax.imshow(np.log10(np.transpose(error_mat[:,:])), cmap = plt.cm.viridis, interpolation='none', extent=[vec1[0],vec1[-1],vec2[0],vec2[-1]], aspect = 'auto', origin = 'lower')
# cbar = fig.colorbar(error, extend='both')
# cbar.minorticks_on()     
# fig.suptitle('log10(Error) versus {} and {}'.format(x_label,y_label))
# plt.title(title_string)
# ax.set_xlabel(r'{}'.format(x_label))
# ax.set_ylabel(r'{}'.format(y_label))   
# plt.show()      
# fig.savefig('figures/'+save_str+'__log.png') 

#%% find delay between junctions

delay_df_jtl = np.zeros([len(j_jtl_peaks)])
delay_jtl_di = np.zeros([len(j_di_peaks)])
delay_di_df = np.zeros([len(j_di_peaks)])

for ii in range(len(j_jtl_peaks)):
    delay_df_jtl[ii] = time_vec[j_jtl_peaks[ii]]-time_vec[j_df_peaks[ii]]
for ii in range(len(j_di_peaks)):
    delay_jtl_di[ii] = time_vec[j_di_peaks[ii]]-time_vec[j_jtl_peaks[ii]]
for ii in range(len(j_di_peaks)):
    delay_di_df[ii] = time_vec[j_df_peaks[ii+1]]-time_vec[j_di_peaks[ii]]


fig, ax = plt.subplots(nrows = 1, ncols = 3, sharex = False, sharey = True)
fig.suptitle(file_name)

ax[0].plot(time_vec[j_jtl_peaks]*1e9,delay_df_jtl*1e12, '-', label = '$\Delta t_{jtl-df}$')            
ax[0].plot(time_vec[j_di_peaks]*1e9,delay_jtl_di*1e12, '-', label = '$\Delta t_{di-jtl}$')
ax[0].plot(time_vec[j_di_peaks]*1e9,delay_di_df*1e12, '-', label = '$\Delta t_{di-df}$')
ax[0].set_xlabel(r'Time [ns]')
ax[0].set_ylabel(r'Delay between fluxons from adjacent JJs [ps]')
ax[0].legend()

ax[1].plot(I_drive[j_jtl_peaks]*1e6,delay_df_jtl*1e12, '-', label = '$\Delta t_{jtl-df}$')            
ax[1].plot(I_drive[j_di_peaks]*1e6,delay_jtl_di*1e12, '-', label = '$\Delta t_{di-jtl}$')
ax[1].plot(I_drive[j_di_peaks]*1e6,delay_di_df*1e12, '-', label = '$\Delta t_{di-df}$')
ax[1].set_xlabel(r'$I_{drive}$ [$\mu$A]')
ax[1].legend()

ax[2].plot(I_di[j_jtl_peaks]*1e6,delay_df_jtl*1e12, '-', label = '$\Delta t_{jtl-df}$')            
ax[2].plot(I_di[j_di_peaks]*1e6,delay_jtl_di*1e12, '-', label = '$\Delta t_{di-jtl}$')
ax[2].plot(I_di[j_di_peaks]*1e6,delay_di_df*1e12, '-', label = '$\Delta t_{di-df}$')
ax[2].set_xlabel(r'$I_{di}$ [$\mu$A]')
ax[2].legend()

plt.show()

#%% fit total delay to functional form

# popt, pcov = curve_fit(inter_fluxon_interval__fit_2, I_di[j_df_peaks[0:-1]], j_df_ifi, p0 = [50e-12,19e-6,2.8,0.5,105e-6], bounds = ([10e-12,10e-6, 0, 0, 10e-6], [100e-12,30e-6, 3., 1., 200e-6]))

# fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
# ax.plot(I_di[j_df_peaks[0:-1]]*1e6,j_df_ifi*1e12, '-', label = 'WR')             
# ax.plot(I_di*1e6,inter_fluxon_interval__fit_2(I_di,*popt)*1e12, '-', label = 'Fit: t0 = {:3.2f}ps, $I_f$ = {:2.2f}uA, mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(popt[0]*1e12,popt[1]*1e6,popt[2],popt[3],popt[4]*1e6))             
# ax.set_xlabel(r'Current [$\mu$A]')
# ax.set_ylabel(r'Inter-fluxon interval [ps]')
# ax.legend()
# ax.set_ylim([0,200])
# plt.show()

# popt, pcov = curve_fit(inter_fluxon_interval__fit_3, I_di[j_df_peaks[0:-1]], j_df_ifi, bounds = ([40.1e-6,40.1e-6], [70e-6,70e-6]))

# fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
# ax.plot(I_di[j_df_peaks[0:-1]]*1e6,j_df_ifi*1e12, '-', label = 'WR')             
# ax.plot(I_di*1e6,inter_fluxon_interval__fit_3(I_di,*popt)*1e12, '-', label = 'Fit: I_bar_1 = {:2.4f}uA, I_bar_2 = {:2.4f}uA'.format(popt[0]*1e6,popt[1]*1e6))             
# ax.set_xlabel(r'Current [$\mu$A]')
# ax.set_ylabel(r'Inter-fluxon interval [ps]')
# ax.legend()
# ax.set_ylim([0,200])
# plt.show()

#%%
# abcde = inter_fluxon_interval__fit_3(I_di,*popt)*1e12 

#%% fit inter-fluxon interval versus current bias

# plt.close('all')

# inter_fluxon_interval = np.diff(time_vec__part[fluxon_peaks])

# # popt, pcov = curve_fit(inter_fluxon_interval__fit, current_vec__part[fluxon_peaks[0:-1]], inter_fluxon_interval, bounds = ([0, 0, 0], [3., 1., 200e-6]))

# fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
# ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,inter_fluxon_interval*1e12, '-', label = 'WR')                         
# # ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,inter_fluxon_interval__fit(current_vec__part[fluxon_peaks[0:-1]],2.8,0.58,100e-6)*1e12, '-', label = 'Fit: mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(2.8,0.58,100e-6*1e6)) 
# ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,inter_fluxon_interval__fit(current_vec__part[fluxon_peaks[0:-1]],*popt)*1e12, '-', label = 'Fit: mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(popt[0],popt[1],popt[2]*1e6)) 
# ax.set_xlabel(r'Current [$\mu$A]')
# ax.set_ylabel(r'Inter-fluxon Interval [ps]')
# ax.legend()
# plt.show()

# fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
# ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,1/inter_fluxon_interval*1e-9, '-', label = 'WR')                         
# # ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,inter_fluxon_interval__fit(current_vec__part[fluxon_peaks[0:-1]],2.8,0.58,100e-6)*1e12, '-', label = 'Fit: mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(2.8,0.58,100e-6*1e6)) 
# ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,1/inter_fluxon_interval__fit(current_vec__part[fluxon_peaks[0:-1]],*popt)*1e-9, '-', label = 'Fit: mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(popt[0],popt[1],popt[2]*1e6)) 
# ax.set_xlabel(r'Current [$\mu$A]')
# ax.set_ylabel(r'Fluxon Generation Rate [GHz]')
# ax.legend()
# plt.show()

#%%
# xqp = inter_fluxon_interval(40e-6)