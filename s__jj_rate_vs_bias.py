#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages
from _functions import read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval
from util import physical_constants
p = physical_constants()

#%% load wr data

directory = 'wrspice_data'
file_name = 'jj_rate_vs_bias.dat'
data_dict = read_wr_data(directory+'/'+file_name)
data_dict['file_name'] = file_name

#%% plot wr data
data_to_plot = ['@I0[c]','v(1)']#'L0#branch','L1#branch','L2#branch',
plot_save_string = False
plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)

#%% fit average voltage versus current bias
time_vec = data_dict['time']
current_vec = data_dict['@I0[c]']
voltage_vec = data_dict['v(1)']

initial_ind = (np.abs(time_vec-1e-9)).argmin()
final_ind = (np.abs(time_vec-41e-9)).argmin()

time_vec__part = time_vec[initial_ind:final_ind]
current_vec__part = current_vec[initial_ind:final_ind]
voltage_vec__part = voltage_vec[initial_ind:final_ind]

# fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
# ax.plot(current_vec__part*1e6,voltage_vec__part*1e3, '-', )             
# ax.set_xlabel(r'Current [$\mu$A]')
# ax.set_ylabel(r'Voltage [mV]')
# plt.show()


# num_time_samples_avg = 10
# voltage_vec__avg = np.zeros([len(voltage_vec__part)-num_time_samples_avg])
# current_vec__avg = np.zeros([len(current_vec__part)-num_time_samples_avg])
# for ii in range(len(voltage_vec__part)-num_time_samples_avg):
#     voltage_vec__avg[ii] = np.sum(voltage_vec__part[ii:ii+num_time_samples_avg])/num_time_samples_avg
#     current_vec__avg[ii] = np.sum(current_vec__part[ii:ii+num_time_samples_avg])/num_time_samples_avg
    
fluxon_peaks, _ = find_peaks(voltage_vec__part, height = 100e-6)
num_fluxons_avg = 20
voltage_vec__avg = np.zeros([len(fluxon_peaks)-num_fluxons_avg])
current_vec__avg = np.zeros([len(fluxon_peaks)-num_fluxons_avg])
for ii in range(len(fluxon_peaks)-num_fluxons_avg):
    voltage_vec__avg[ii] = np.sum(voltage_vec__part[fluxon_peaks[ii]:fluxon_peaks[ii+num_fluxons_avg]-1])/(fluxon_peaks[ii+num_fluxons_avg]-fluxon_peaks[ii]-1)
    current_vec__avg[ii] = np.sum(current_vec__part[fluxon_peaks[ii]:fluxon_peaks[ii+num_fluxons_avg]-1])/(fluxon_peaks[ii+num_fluxons_avg]-fluxon_peaks[ii]-1)

# fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
# ax.plot(voltage_vec__avg*1e3,current_vec__avg*1e6, '-', label = 'WR')             
# ax.plot(V_fq__for_fit(current_vec__avg,2,0.5,110e-6)*1e3,current_vec__avg*1e6, '-', label = 'Approx')             
# ax.set_ylabel(r'Current [$\mu$A]')
# ax.set_xlabel(r'Voltage [mV]')
# ax.legend()
# plt.show()

# current_vec_avg = np.asarray(current_vec__avg, dtype = float)
# V_test = V_fq__for_fit(current_vec__avg,2,0.5,110e-6)

popt, pcov = curve_fit(V_fq__fit, current_vec__avg, voltage_vec__avg, bounds = ([0, 0, 10e-6], [3., 1., 200e-6]))

fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
ax.plot(voltage_vec__avg*1e3,current_vec__avg*1e6, '-', label = 'WR')             
ax.plot(V_fq__fit(current_vec__avg,*popt)*1e3,current_vec__avg*1e6, '-', label = 'Fit: mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(popt[0],popt[1],popt[2]*1e6))             
ax.set_ylabel(r'Current [$\mu$A]')
ax.set_xlabel(r'Voltage [mV]')
ax.legend()
plt.show()


#%% fit inter-fluxon interval versus current bias

# plt.close('all')

inter_fluxon_interval = np.diff(time_vec__part[fluxon_peaks])

# popt, pcov = curve_fit(inter_fluxon_interval__fit, current_vec__part[fluxon_peaks[0:-1]], inter_fluxon_interval, bounds = ([0, 0, 0], [3., 1., 200e-6]))

fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,inter_fluxon_interval*1e12, '-', label = 'WR')                         
# ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,inter_fluxon_interval__fit(current_vec__part[fluxon_peaks[0:-1]],2.8,0.58,100e-6)*1e12, '-', label = 'Fit: mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(2.8,0.58,100e-6*1e6)) 
ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,inter_fluxon_interval__fit(current_vec__part[fluxon_peaks[0:-1]],*popt)*1e12, '-', label = 'Fit: mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(popt[0],popt[1],popt[2]*1e6)) 
ax.set_xlabel(r'Current [$\mu$A]')
ax.set_ylabel(r'Inter-fluxon Interval [ps]')
ax.legend()
plt.show()

fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,1/inter_fluxon_interval*1e-9, '-', label = 'WR')                         
# ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,inter_fluxon_interval__fit(current_vec__part[fluxon_peaks[0:-1]],2.8,0.58,100e-6)*1e12, '-', label = 'Fit: mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(2.8,0.58,100e-6*1e6)) 
ax.plot(current_vec__part[fluxon_peaks[0:-1]]*1e6,1/inter_fluxon_interval__fit(current_vec__part[fluxon_peaks[0:-1]],*popt)*1e-9, '-', label = 'Fit: mu1 = {:1.3f}, mu2 = {:1.3f}, V0 = {:3.3f}uV'.format(popt[0],popt[1],popt[2]*1e6)) 
ax.set_xlabel(r'Current [$\mu$A]')
ax.set_ylabel(r'Fluxon Generation Rate [GHz]')
ax.legend()
plt.show()

#%%
# xqp = inter_fluxon_interval(40e-6)