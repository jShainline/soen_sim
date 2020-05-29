#%%
import numpy as np
from matplotlib import pyplot as plt
import time

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__synapse__Isi_and_Isf, plot_wr_comparison__synapse, plot_wr_comparison__synapse__tiles, plot_wr_comparison__synapse__vary_Isy, plot_wr_comparison__synapse__tiles__with_drive
from _functions import read_wr_data, chi_squared_error
from util import physical_constants
from soen_sim import input_signal, synapse
p = physical_constants()

plt.close('all')

#%%
I_spd = 20e-6

spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]    
I_sy_vec = [21e-6,27e-6,33e-6,39e-6,28e-6,28e-6,28e-6,28e-6,32e-6,32e-6,32e-6,32e-6]
L_si_vec = [77.5e-9,77.5e-9,77.5e-9,77.5e-9,7.75e-9,77.5e-9,775e-9,7.75e-6,775e-9,775e-9,775e-9,775e-9]
tau_si_vec = [250e-9,250e-9,250e-9,250e-9,250e-9,250e-9,250e-9,250e-9,10e-9,50e-9,250e-9,1.25e-6]

dt = 0.1e-9
tf = 1e-6
                    
# create sim_params dictionary
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf
sim_params['synapse_model'] = 'lookup_table' # __spd_delta

target_data_array = []
target_drive_array = []

actual_data_array = []
actual_drive_array = []

error_array_drive = []
error_array_signal = []

calculate_chi_squared = True
plot_each = False

#%%

num_files = len(I_sy_vec)
t_tot = time.time()
for ii in range(num_files): # range(1): # 
    
    print('\nii = {} of {}\n'.format(ii+1,num_files))
    
    #load WR data
    file_name = 'syn_1jj_Ispd20.00uA_trep50ns_Isy{:04.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii]*1e6,L_si_vec[ii]*1e9,tau_si_vec[ii]*1e9)
    data_dict = read_wr_data('wrspice_data/test_data/1jj/'+file_name)
    target_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
    target_drive_array.append(target_drive)
    target_data = np.vstack((data_dict['time'],data_dict['L1#branch']))    
    target_data_array.append(target_data)

    # initialize input signal
    input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse    
    synapse_1 = synapse('sy', num_jjs = 1, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si_vec[ii], 
                        integration_loop_self_inductance = L_si_vec[ii], integration_loop_output_inductance = 0e-12, 
                        synaptic_bias_currents = [I_spd,I_sy_vec[ii]],
                        input_signal_name = 'in', synapse_model_params = sim_params)
    
    synapse_1.run_sim() 
    
    actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
    actual_drive_array.append(actual_drive)
    actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:])) 
    sf_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_sf[:]))
    actual_data_array.append(actual_data)
    
    if calculate_chi_squared == True:
        error_drive = chi_squared_error(target_drive,actual_drive)
        error_array_drive.append(error_drive)
        error_signal = chi_squared_error(target_data,actual_data)
        error_array_signal.append(error_signal)
        if plot_each == True:
            plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,error_drive,error_signal)     
    else:
        if plot_each == True:
            plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,1,1)    

elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+'s')

#%%

if calculate_chi_squared == True:
    legend_strings = [ ['Isy = {:5.2f}uA'.format(I_sy_vec[0]*1e6),
                        'Isy = {:5.2f}uA'.format(I_sy_vec[1]*1e6),
                        'Isy = {:5.2f}uA'.format(I_sy_vec[2]*1e6),
                        'Isy = {:5.2f}uA'.format(I_sy_vec[3]*1e6)],
                       ['Lsi = {:7.2f}nH'.format(L_si_vec[4]*1e9),
                        'Lsi = {:7.2f}nH'.format(L_si_vec[5]*1e9),
                        'Lsi = {:7.2f}nH'.format(L_si_vec[6]*1e9),
                        'Lsi = {:7.2f}nH'.format(L_si_vec[7]*1e9)],
                       ['tau_si = {:7.2f}ns'.format(tau_si_vec[8]*1e9),
                        'tau_si = {:7.2f}ns'.format(tau_si_vec[9]*1e9),
                        'tau_si = {:7.2f}ns'.format(tau_si_vec[10]*1e9),
                        'tau_si = {:7.2f}ns'.format(tau_si_vec[11]*1e9)] ]
    plot_wr_comparison__synapse__tiles__with_drive(target_drive_array,actual_drive_array,target_data_array,actual_data_array,spike_times,error_array_drive,error_array_signal,legend_strings)


#%% vary Isy more

if 1 == 2:
    
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
                            synaptic_bias_currents = [I_spd,1e-6*I_sy_vec[ii]],
                            input_signal_name = 'in', synapse_model_params = sim_params)
        
        synapse_1.run_sim() 
        
        actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
        actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:])) 
        sf_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_sf[:]))
        actual_data_array.append(actual_data)
            
    if calculate_chi_squared == True:
        error_drive = chi_squared_error(target_drive,actual_drive)
        error_array_drive.append(error_drive)
        error_signal = chi_squared_error(target_data,actual_data)
        error_array_signal.append(error_signal)
        if plot_each == True:
            plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,error_drive,error_signal)     
    else:
        if plot_each == True:
            plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,1,1)
    
    elapsed = time.time() - t_tot
    print('soen_sim duration = '+str(elapsed)+' s for vary I_sy')
    
    plot_wr_comparison__synapse__vary_Isy(I_sy_vec,target_data_array,actual_data_array)
