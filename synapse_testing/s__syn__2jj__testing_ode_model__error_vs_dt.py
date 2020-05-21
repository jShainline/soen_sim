#%%
import numpy as np
from matplotlib import pyplot as plt
import time

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__synapse__Isi_and_Isf, plot_wr_comparison__synapse, plot_wr_comparison__synapse__tiles, plot_wr_comparison__synapse__vary_Isy, plot__syn__error_vs_dt
from _functions import read_wr_data, chi_squared_error
from util import physical_constants
from soen_sim import input_signal, synapse
p = physical_constants()

plt.close('all')

#%%
# dt_vec = np.concatenate([np.arange(0.01e-9,0.11e-9,0.01e-9),np.arange(0.2e-9,1.1e-9,0.1e-9),np.arange(2e-9,11e-9,1e-9)])
dt_vec = [0.01e-9,0.1e-9,1e-9]

wr_dt = 10e-12;

I_spd = 20e-6
I_sc = 35e-6
spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]

tf = 1e-6
                    
target_data_array = []
actual_data_array = []
error_array = []

data_file_list = []

spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]    
I_sy_vec = [23e-6,27e-6,33e-6,38e-6,29e-6,29e-6,29e-6,29e-6,34e-6,34e-6,34e-6,34e-6]
L_si_vec = [77.5e-9,77.5e-9,77.5e-9,77.5e-9,7.75e-9,77.5e-9,775e-9,7.75e-6,775e-9,775e-9,775e-9,775e-9]
tau_si_vec = [250e-9,250e-9,250e-9,250e-9,250e-9,250e-9,250e-9,250e-9,10e-9,50e-9,250e-9,1.25e-6]
for ii in range(len(I_sy_vec)):
    data_file_list.append('syn_2jj_Ispd20.00uA_trep50ns_Isy{:04.2f}uA_Isc35.00uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt{:4.1f}ps_tsim1000ns.dat'.format(I_sy_vec[ii]*1e6,L_si_vec[ii]*1e9,tau_si_vec[ii]*1e9,wr_dt*1e12)) 

num_files = len(data_file_list)    
error_mat = np.zeros([num_files,len(dt_vec)])
error_drive_mat = np.zeros([num_files,len(dt_vec)])

for qq in range(len(dt_vec)): 
    
    sim_params = dict()
    sim_params['dt'] = dt_vec[qq]
    sim_params['tf'] = tf
    sim_params['synapse_model'] = 'ode'

    for ii in range(num_files):
                
        print('qq = {} of {}, ii = {} of {}'.format(qq+1,len(dt_vec),ii+1,num_files))
    
        #load WR data
        data_dict = read_wr_data('wrspice_data/test_data/2jj/'+data_file_list[ii])
        target_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
        target_data = np.vstack((data_dict['time'],data_dict['L2#branch']))
    
        # initialize input signal
        input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
            
        # initialize synapse
        synapse_1 = synapse('sy', num_jjs = 2, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si_vec[ii], 
                            integration_loop_self_inductance = L_si_vec[ii], integration_loop_output_inductance = 0e-12, 
                            synaptic_bias_current = I_sy_vec[ii], integration_loop_bias_current = I_sc,
                            input_signal_name = 'in', synapse_model_params = sim_params)
        
        synapse_1.run_sim()   
        
        actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
        actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:]))
        
        error_drive = chi_squared_error(target_drive,actual_drive)
        error_drive_mat[ii,qq] = error_drive
        
        error_signal = chi_squared_error(target_data,actual_data)
        error_mat[ii,qq] = error_signal
    
#%%

plot__syn__error_vs_dt(np.asarray(dt_vec),error_mat,error_drive_mat)