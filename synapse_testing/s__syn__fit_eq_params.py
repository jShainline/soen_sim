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
from soen_sim import input_signal, synapse
p = physical_constants()

plt.close('all')

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
    
#%% vary Isy
I_sy_vec = [23*1e-6,28*1e-6,33*1e-6,38*1e-6]#uA
L_si = 77.5e-9
tau_si = 250e-9
 
data_file_list = []
num_files = len(I_sy_vec)
for ii in range(num_files):
    data_file_list.append('syn_Ispd20.00uA_Isy{:04.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii]*1e6,L_si_vec[ii]*1e9,tau_si_vec[ii]*1e9))
    
for ii in range(num_files):
    
    print('ii = {:d} of {:d}'.format(ii+1,num_files))
    
    directory = 'wrspice_data/fitting_data'
    file_name = data_file_list[ii]
    data_dict = read_wr_data(directory+'/'+file_name)
    target_data = np.vstack((data_dict['time'],data_dict['L3#branch']))
    target_data__drive = np.vstack((data_dict['time'],data_dict['L4#branch']))
    
    # initialize input signal
    name__i = 'in'
    input_1 = input_signal(name__i, input_temporal_form = 'single_spike', spike_times = np.array([100e-9]))#stochasticity = 'gaussian', jitter_params = jitter_params   
            # print(input_1.spike_times)
            
    # initialize synapses
    name_s = 'sy'
    synapse_1 = synapse(name_s, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si, 
                        integration_loop_self_inductance = L_si, integration_loop_output_inductance = 0e-12, 
                        synaptic_bias_current = I_sy_vec[ii], integration_loop_bias_current = 35e-6,
                        input_signal_name = 'in')
    
    # dendritic_drive = dendritic_drive__piecewise_linear(input_1.time_vec,pwl_drive)
    # actual_data__drive = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
    # error__drive = chi_squared_error(target_data__drive,actual_data__drive)
            
    # sim_params = dict()
    # sim_params['dt'] = dt
    # sim_params['tf'] = tf
    # dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
    #                                       input_synaptic_connections = [], input_synaptic_inductances = [[]], 
    #                                       input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
    #                                       input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
    #                                       thresholding_junction_critical_current = 40e-6, bias_currents = [71.5e-6,36e-6,35e-6],
    #                                       integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
    #                                       integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
    #                                       dendrite_model_params = sim_params)
    
    # dendrite_1.run_sim()
                                    
    # actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))    
    # error__signal = chi_squared_error(target_data,actual_data)
    
    # plot_wr_comparison__drive_and_response('test',target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)
    

#%% vary Lsi

#%%vary tau_si
