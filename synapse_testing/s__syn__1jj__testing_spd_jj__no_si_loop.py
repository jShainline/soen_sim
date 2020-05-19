#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import time

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__synapse__spd_jj_test
from _functions import read_wr_data, chi_squared_error
from util import physical_constants
from soen_sim import input_signal, synapse
p = physical_constants()

plt.close('all')

#%%
# spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]
spike_times = [5e-9]

dt = 0.01e-9
tf = 300e-9 # 1e-6
                    
# create sim_params dictionary
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf

#%%
target_data_array = []
actual_data_array = []
error_array = []

#%% vary Isy
I_sy_vec = [39e-6] # [21e-6,39e-6] # [21e-6,27e-6,33e-6,39e-6]

num_files = len(I_sy_vec)
t_tot = time.time()
load_wr = True
for ii in range(num_files): # range(1): # 
    
    print('\nvary Isy, ii = {} of {}\n'.format(ii+1,num_files))
    
    #load WR data
    if load_wr == True:
        # file_name = 'syn_1jj_Ispd20.00uA_trep50ns_Isy{:04.2f}uA_noSI_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii]*1e6)
        file_name = 'syn_1jj_Ispd20.00uA_Isy{:04.2f}uA_noSI_dt00.2ps_tsim1000ns.dat'.format(I_sy_vec[ii]*1e6)
        data_dict = read_wr_data('wrspice_data/test_data/1jj/'+file_name)
        target_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
        target_data = np.vstack((data_dict['time'],data_dict['L2#branch']))
        target_data_array.append(target_data)
    
    # find fluxon peaks
    time_vec = data_dict['time']
    I_sf_wr = data_dict['L2#branch']  
    V_sf_wr = data_dict['v(4)']
    j_peaks, _ = find_peaks(V_sf_wr, height = 100e-6)
    
    # find inter-fluxon intervals and fluxon generation rates
    j_ifi = np.diff(time_vec[j_peaks])
    j_rate = 1/j_ifi
    
    # calculate average currents, voltages, and times
    V_sf_wr_avg = np.zeros([len(j_peaks)-1])
    I_sf_wr_avg = np.zeros([len(j_peaks)-1])
    time_avg = np.zeros([len(j_peaks)-1])
    
    for jj in range(len(j_peaks)-1):
        ind_vec = np.arange(j_peaks[jj],j_peaks[jj+1],1)
        V_sf_wr_avg[jj] = np.sum(V_sf_wr[ind_vec])/len(ind_vec) 
        I_sf_wr_avg[jj] = np.sum(I_sf_wr[ind_vec])/len(ind_vec)
        time_avg[jj] = np.sum(time_vec[ind_vec])/len(ind_vec)

    # initialize input signal
    input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse    
    synapse_1 = synapse('sy', num_jjs = 1, synaptic_bias_currents = [I_sy_vec[ii]], input_signal_name = 'in', synapse_model_params = sim_params)
    
    synapse_1.run_sim() 
    
    I_sf = synapse_1.I_sf
    V_sf = synapse_1.V_sf
    r_spd1 = synapse_1.r_spd1
    I_spd2 = I_sf - I_sy_vec[ii]
    j_sf_state = synapse_1.j_sf_state
    
    actual_drive = np.vstack((synapse_1.time_vec[:],I_spd2[:]))
    actual_data = np.vstack((synapse_1.time_vec[:],I_sf[:])) 
    actual_data_array.append(actual_data)
    
    # error_drive = chi_squared_error(target_drive,actual_drive)
    # error_signal = chi_squared_error(target_data,actual_data)
    # error_array.append(error_signal)
    
    plot_wr_comparison__synapse__spd_jj_test(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,V_sf_wr,j_peaks,V_sf_wr_avg,time_avg,V_sf)    
    # plot_wr_comparison__synapse__spd_jj_test(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,error_drive,error_signal,V_sf_wr,j_peaks,V_sf_wr_avg,time_avg,V_sf)    
    
    # plot_wr_comparison__synapse__Isi_and_Isf('bias_lower all; J_sf criterion',spike_times,target_drive,actual_drive,target_data,actual_data,sf_data,synapse_1.I_c,synapse_1.I_reset,file_name,error_drive,error_signal)    

elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s for vary I_sy')
