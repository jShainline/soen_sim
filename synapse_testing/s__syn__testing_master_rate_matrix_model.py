#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages, plot_wr_comparison__synapse, plot_error_mat
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3, chi_squared_error
from _functions__more import synapse_model__parameter_sweep
from util import physical_constants
from soen_sim import input_signal, synapse
p = physical_constants()

plt.close('all')

#%%
spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]

dt = 1e-9
tf = 1e-6
                    
# create sim_params dictionary
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf


#%% vary Isy
I_sy_vec = [23e-6,28e-6,33e-6,38e-6]
L_si = 77.5e-9
tau_si = 250e-9

num_files = len(I_sy_vec)
t_tot = time.time()
for ii in range(num_files):
    
    #load WR data
    file_name = 'syn_Ispd20.00uA_Isy{:04.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii]*1e6,L_si*1e9,tau_si*1e9)
    data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
    wr_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
    target_data = np.vstack((data_dict['time'],data_dict['L3#branch']))

    # initialize input signal
    name__i = 'in'
    input_1 = input_signal(name__i, input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse
    name_s = 'sy'
    synapse_1 = synapse(name_s, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si, 
                        integration_loop_self_inductance = L_si, integration_loop_output_inductance = 0e-12, 
                        synaptic_bias_current = I_sy_vec[ii], integration_loop_bias_current = 35e-6,
                        input_signal_name = 'in', synapse_model_params = sim_params)
    
    synapse_1.run_sim()    
    actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:]))    
    error_signal = chi_squared_error(target_data,actual_data)
    plot_wr_comparison__synapse(file_name,spike_times,wr_drive,target_data,actual_data,file_name,error_signal)    

elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary I_sy')


#%% vary Lsi
I_sy = 28e-6
L_si_vec = [7.75e-9,77.5e-9,775e-9,7.75e-6]
tau_si = 250e-9

num_files = len(L_si_vec)
t_tot = time.time()
for ii in range(num_files):
    
    #load WR data
    file_name = 'syn_Ispd20.00uA_Isy{:05.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy*1e6,L_si_vec[ii]*1e9,tau_si*1e9)
    data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
    wr_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
    target_data = np.vstack((data_dict['time'],data_dict['L3#branch']))

    # initialize input signal
    name__i = 'in'
    input_1 = input_signal(name__i, input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse
    name_s = 'sy'
    synapse_1 = synapse(name_s, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si, 
                        integration_loop_self_inductance = L_si_vec[ii], integration_loop_output_inductance = 0e-12, 
                        synaptic_bias_current = I_sy, integration_loop_bias_current = 35e-6,
                        input_signal_name = 'in', synapse_model_params = sim_params)
    
    synapse_1.run_sim()    
    actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:]))    
    error_signal = chi_squared_error(target_data,actual_data)
    plot_wr_comparison__synapse(file_name,spike_times,wr_drive,target_data,actual_data,file_name,error_signal) 
    
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary L_si')


#%%vary tau_si
I_sy = 33e-6
L_si = 775e-9
tau_si_vec = [10e-9,50e-9,250e-9,1.25e-6]

num_files = len(tau_si_vec)
t_tot = time.time()
for ii in range(num_files):
    
    #load WR data
    file_name = 'syn_Ispd20.00uA_Isy{:05.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy*1e6,L_si*1e9,tau_si_vec[ii]*1e9)
    data_dict = read_wr_data('wrspice_data/test_data/'+file_name)
    wr_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
    target_data = np.vstack((data_dict['time'],data_dict['L3#branch']))

    # initialize input signal
    name__i = 'in'
    input_1 = input_signal(name__i, input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse
    name_s = 'sy'
    synapse_1 = synapse(name_s, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si_vec[ii], 
                        integration_loop_self_inductance = L_si, integration_loop_output_inductance = 0e-12, 
                        synaptic_bias_current = I_sy, integration_loop_bias_current = 35e-6,
                        input_signal_name = 'in', synapse_model_params = sim_params)
    
    synapse_1.run_sim()    
    actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:]))    
    error_signal = chi_squared_error(target_data,actual_data)
    plot_wr_comparison__synapse(file_name,spike_times,wr_drive,target_data,actual_data,file_name,error_signal)

elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary tau_si')

