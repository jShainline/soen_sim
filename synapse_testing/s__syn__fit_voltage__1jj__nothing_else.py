#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pickle
import numpy.matlib

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks_and_average_voltage__isolated_JJ
from _functions import read_wr_data, syn_isolatedjj_voltage_fit
from util import physical_constants
p = physical_constants()

# plt.close('all')

#%% load wr data

Ic = 40#uA

data_dict = read_wr_data('wrspice_data/fitting_data/jj_rate_vs_bias.dat')


#%% find peaks, find rates

time_vec = data_dict['time']
I_bias = data_dict['@I0[c]']  
V_fq = data_dict['v(1)']
j_peaks, _ = find_peaks(V_fq, height = 100e-6)
   
ind_max = V_fq[j_peaks].argmax()
j_peaks = j_peaks[ind_max:]

# find inter-fluxon intervals and fluxon generation rates versus bias current  
j_ifi = np.diff(time_vec[j_peaks])
j_rate = 1/j_ifi

# calculate average currents, voltages, and times
V_avg = np.zeros([len(j_peaks)-1])
I_avg = np.zeros([len(j_peaks)-1])
time_avg = np.zeros([len(j_peaks)-1])

# for jj in range(len(j_peaks)-1):
#     ind_vec = np.arange(j_peaks[jj],j_peaks[jj+1],1)
#     V_avg[jj] = 1e6*np.sum(V_fq[ind_vec])/len(ind_vec) 
#     I_avg[jj] = 1e6*np.sum(I_bias[ind_vec])/len(ind_vec)
#     time_avg[jj] = 1e9*np.sum(time_vec[ind_vec])/len(ind_vec)
    
for jj in range(len(j_peaks)-1):
    ind_vec = np.arange(j_peaks[jj],j_peaks[jj+1],1)
    V_avg[jj] = np.sum(V_fq[ind_vec])/len(ind_vec) 
    I_avg[jj] = np.sum(I_bias[ind_vec])/len(ind_vec)
    time_avg[jj] = np.sum(time_vec[ind_vec])/len(ind_vec)

#%% fit voltage versus current bias

# popt, pcov = curve_fit(syn_isolatedjj_voltage_fit, I_avg, V_avg, bounds = ([100, 3, 0.2, 1], [400, 4, 0.6, 1.2]))

popt, pcov = curve_fit(syn_isolatedjj_voltage_fit, I_avg, V_avg, bounds = ([100e-6, 3, 0.2, 1e-6], [400e-6, 4, 0.6, 1.2e-6]))

V0 = popt[0]
mu1 = popt[1]
mu2 = popt[2]
Ir = popt[3]

# V_fit = syn_isolatedjj_voltage_fit(I_avg,V0,mu1,mu2,Ir)
# print('V0 = {}uV, mu1 = {}, mu2 = {}, Ir = {}uA'.format(V0,mu1,mu2,Ir))

V_fit = syn_isolatedjj_voltage_fit(I_avg,V0,mu1,mu2,Ir)
print('V0 = {}uV, mu1 = {}, mu2 = {}, Ir = {}uA'.format(V0*1e6,mu1,mu2,Ir*1e6))

# popt, pcov = curve_fit(syn_isolatedjj_voltage_fit, I_avg, V_avg, bounds = ([200, 3, 1], [300, 4.5, 1.3]))
# V0 = popt[0]
# mu = popt[1]
# Ir = popt[2]


# V_fit = syn_isolatedjj_voltage_fit(I_avg,V0,mu,Ir)

# print('V0 = {}uV, mu = {}, Ir = {}uA'.format(V0,mu,Ir))


#%% plot
    
plot_fq_peaks_and_average_voltage__isolated_JJ(time_vec,I_bias,V_fq,j_peaks,time_avg,I_avg,V_avg,V_fit,V0,mu1,mu2,Ir)
# plot_fq_peaks_and_average_voltage__isolated_JJ(time_vec,I_bias,V_fq,j_peaks,time_avg,I_avg,V_avg,V_fit,V0,mu,Ir)
    
