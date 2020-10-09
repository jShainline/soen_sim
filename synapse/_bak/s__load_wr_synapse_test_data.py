#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages, plot_wr_comparison__synapse
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3, chi_squared_error
from _function_more import synapse_model__parameter_sweep
from util import physical_constants
from soen_sim import input_signal, synapse
p = physical_constants()

# plt.close('all')

#%% load wr data, determine quantities of interest

I_sy_vec = [23,28,33,38,28,28,28,28,33,33,33,33]#uA
L_si_vec = [77.5,77.5,77.5,77.5,7.75,77.5,775,7750,775,775,775,775]#nH
tau_si_vec = [250,250,250,250,250,250,250,250,10,50,250,1250]#ns
 
data_file_list = []
num_files = len(I_sy_vec)
for ii in range(num_files):
    data_file_list.append('syn_Ispd20.00uA_Isy{:04.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii],L_si_vec[ii],tau_si_vec[ii]))
    
for ii in range(num_files):
    
    print('ii = {:d} of {:d}'.format(ii+1,num_files))
    
    directory = 'wrspice_data/fitting_data'
    file_name = data_file_list[ii]
    data_dict = read_wr_data(directory+'/'+file_name)
    
    #plot wr time traces
    data_to_plot = ['L0#branch','L3#branch','v(2)']
    plot_save_string = file_name
    plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)