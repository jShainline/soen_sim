#%%
import numpy as np
from matplotlib import pyplot as plt
import time

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__synapse__Isi_and_Isf, plot_wr_comparison__synapse, plot_wr_comparison__synapse__tiles, plot_wr_comparison__synapse__vary_Isy
from _functions import read_wr_data, chi_squared_error
from util import physical_constants
from soen_sim import input_signal, synapse
p = physical_constants()

plt.close('all')

#%%
spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]

dt = 0.1e-9
tf = 1e-6
                    
# create sim_params dictionary
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf

target_data_array = []
actual_data_array = []
error_array = []

#%% vary Isy
I_sy_vec = [21e-6,27e-6,33e-6,39e-6]
L_si = 77.5e-9
tau_si = 250e-9

num_files = len(I_sy_vec)
t_tot = time.time()
for ii in range(num_files): # range(1): # 
    
    print('\nvary Isy, ii = {} of {}\n'.format(ii+1,num_files))
    
    #load WR data
    file_name = 'syn_1jj_Ispd20.00uA_trep50ns_Isy{:04.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii]*1e6,L_si*1e9,tau_si*1e9)
    data_dict = read_wr_data('wrspice_data/test_data/1jj/'+file_name)
    target_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
    target_data = np.vstack((data_dict['time'],data_dict['L1#branch']))
    target_data_array.append(target_data)

    # initialize input signal
    input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse    
    synapse_1 = synapse('sy', num_jjs = 1, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si, 
                        integration_loop_self_inductance = L_si, integration_loop_output_inductance = 0e-12, 
                        synaptic_bias_current = I_sy_vec[ii], integration_loop_bias_current = 35e-6,
                        input_signal_name = 'in', synapse_model_params = sim_params)
    
    synapse_1.run_sim() 
    actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
    actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:])) 
    sf_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_sf[:]))
    actual_data_array.append(actual_data)
    error_drive = chi_squared_error(target_drive,actual_drive)
    error_signal = chi_squared_error(target_data,actual_data)
    error_array.append(error_signal)
    plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,error_drive,error_signal)    
    # plot_wr_comparison__synapse__Isi_and_Isf('bias_lower all; J_sf criterion',spike_times,target_drive,actual_drive,target_data,actual_data,sf_data,synapse_1.I_c,synapse_1.I_reset,file_name,error_drive,error_signal)    

elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary I_sy')

#%% vary Lsi
I_sy = 28e-6
L_si_vec = [7.75e-9,77.5e-9,775e-9,7.75e-6]
tau_si = 250e-9

num_files = len(L_si_vec)
t_tot = time.time()
for ii in range(num_files):
    
    print('\nvary Lsi, ii = {} of {}\n'.format(ii+1,num_files))
    
    #load WR data
    file_name = 'syn_1jj_Ispd20.00uA_trep50ns_Isy{:05.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy*1e6,L_si_vec[ii]*1e9,tau_si*1e9)
    data_dict = read_wr_data('wrspice_data/test_data/1jj/'+file_name)
    target_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
    target_data = np.vstack((data_dict['time'],data_dict['L1#branch']))
    target_data_array.append(target_data)

    # initialize input signal
    input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse
    synapse_1 = synapse('sy', num_jjs = 1, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si, 
                        integration_loop_self_inductance = L_si_vec[ii], integration_loop_output_inductance = 0e-12, 
                        synaptic_bias_current = I_sy, integration_loop_bias_current = 35e-6,
                        input_signal_name = 'in', synapse_model_params = sim_params)
    
    synapse_1.run_sim()    
    actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
    actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:]))   
    actual_data_array.append(actual_data)
    sf_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_sf[:]))
    error_drive = chi_squared_error(target_drive,actual_drive)
    error_signal = chi_squared_error(target_data,actual_data)
    error_array.append(error_signal)
    plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,error_drive,error_signal)    
    
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary L_si')


#%%vary tau_si
I_sy = 32e-6
L_si = 775e-9
tau_si_vec = [10e-9,50e-9,250e-9,1.25e-6]

num_files = len(tau_si_vec)
t_tot = time.time()
for ii in range(num_files):
    
    print('\nvary tau_si, ii = {} of {}\n'.format(ii+1,num_files))
    
    #load WR data
    file_name = 'syn_1jj_Ispd20.00uA_trep50ns_Isy{:05.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy*1e6,L_si*1e9,tau_si_vec[ii]*1e9)
    data_dict = read_wr_data('wrspice_data/test_data/1jj/'+file_name)
    target_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
    target_data = np.vstack((data_dict['time'],data_dict['L1#branch']))
    target_data_array.append(target_data)

    # initialize input signal
    input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse
    synapse_1 = synapse('sy', num_jjs = 1, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si_vec[ii], 
                        integration_loop_self_inductance = L_si, integration_loop_output_inductance = 0e-12, 
                        synaptic_bias_current = I_sy, integration_loop_bias_current = 35e-6,
                        input_signal_name = 'in', synapse_model_params = sim_params)
    
    synapse_1.run_sim()    
    actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
    actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:]))   
    actual_data_array.append(actual_data)
    sf_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_sf[:]))
    error_drive = chi_squared_error(target_drive,actual_drive)
    error_signal = chi_squared_error(target_data,actual_data)
    error_array.append(error_signal)
    plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,error_drive,error_signal) 

elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary tau_si')

#%%

plot_wr_comparison__synapse__tiles(target_data_array,actual_data_array,spike_times,error_array)

#%% vary Isy more
dI = 1
I_sy_vec = np.arange(21,39+dI,dI)
L_si = 77.5e-9
tau_si = 250e-9

num_files = len(I_sy_vec)
t_tot = time.time()

target_data_array = []
actual_data_array = []
for ii in range(num_files): # range(1): # 
    
    print('\nvary Isy, ii = {} of {}\n'.format(ii+1,num_files))
    
    #load WR data
    file_name = 'syn_1jj_Ispd20.00uA_trep50ns_Isy{:04.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii],L_si*1e9,tau_si*1e9)
    data_dict = read_wr_data('wrspice_data/test_data/1jj/'+file_name)
    target_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
    target_data = np.vstack((data_dict['time'],data_dict['L1#branch']))
    target_data_array.append(target_data)

    # initialize input signal
    input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse    
    synapse_1 = synapse('sy', num_jjs = 1, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si, 
                        integration_loop_self_inductance = L_si, integration_loop_output_inductance = 0e-12, 
                        synaptic_bias_current = 1e-6*I_sy_vec[ii], integration_loop_bias_current = 35e-6,
                        input_signal_name = 'in', synapse_model_params = sim_params)
    
    synapse_1.run_sim() 
    actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
    actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:])) 
    sf_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_sf[:]))
    actual_data_array.append(actual_data)
    
    # error_drive = chi_squared_error(target_drive,actual_drive)
    # error_signal = chi_squared_error(target_data,actual_data)
    # error_array.append(error_signal)
    
    # plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,error_drive,error_signal)    
    # plot_wr_comparison__synapse__Isi_and_Isf('bias_lower all; J_sf criterion',spike_times,target_drive,actual_drive,target_data,actual_data,sf_data,synapse_1.I_c,synapse_1.I_reset,file_name,error_drive,error_signal)    

elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary I_sy')

#%%
plot_wr_comparison__synapse__vary_Isy(I_sy_vec,target_data_array,actual_data_array)
