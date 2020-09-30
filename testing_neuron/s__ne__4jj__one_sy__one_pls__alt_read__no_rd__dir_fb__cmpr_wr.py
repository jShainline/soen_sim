import numpy as np
from matplotlib import pyplot as plt

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_neuronal_response__single_synaptic_pulse, plot_num_in_burst, plot_phase_portrait, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_two, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_one, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_four, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_three
from _functions import read_wr_data
from soen_sim import input_signal, synapse, neuron

plt.close('all')

#%% sim params
dt = 0.1 # ns
spike_times = [5] # ns
  
I_sy_vec = np.asarray([72]) # np.asarray([72,76,80])*1e-6 # [80e-6] # 
I_ne_vec = np.asarray([72]) # np.asarray([72,76,80])
tau_si_vec = np.asarray([500,1000,2000]) # np.asarray([500,1000,2000]) # [1e-6] # 

num_jjs__syn = 4
num_jjs__ne = 4
    
L_si = 77.5e3 # pH # [10e-9,50e-9] #  np.linspace(100e-9,1e-6,num_L_si) # [7.75e-9,77.5e-9,775e-9,7.75e-6]
L_msi = 130 # pH

L_ni = 7.75e3 # pH # 10e-9 # 
L_mni = 400 # pH
tau_nf = 50 # ns

tf = 1e3 # ns

k__nr_nf = 1

#%%

critical_current = 40 # uA
current = 35
norm_current = np.max([np.min([current/critical_current,1]),1e-9])
L_jj = (3.2910596281416393e2/critical_current)*np.arcsin(norm_current)/(norm_current)
r_nf = (30+L_jj)/(50)

directory_name = 'wrspice_data/{:d}jj/single_pulse/'.format(num_jjs__syn)
file_format = 'ne_4jj_1pls_alt_read_no_rd_dir_fb_I_sy{:05.2f}uA_I_nf{:05.2f}uA_tau_si{:07.2f}ns_knrnf{:03.2f}.dat'
for ii in range(len(I_ne_vec)):
    I_ne = I_ne_vec[ii]
    for jj in range(len(I_sy_vec)):
        I_sy = I_sy_vec[jj]
        for kk in range(len(tau_si_vec)):
            tau_si = tau_si_vec[kk]            
            # r_si = (L_si+L_jj)/(tau_si*1e-9)
    
            print('ii = {} of {}; jj = {} of {}; kk = {} of {};'.format(ii+1,len(I_ne_vec),jj+1,len(I_sy_vec),kk+1,len(tau_si_vec)))
            
            file_name = file_format.format(I_sy,I_ne,tau_si,k__nr_nf)
            data_dict = read_wr_data('{}{}'.format(directory_name,file_name))            
            
            input_1 = input_signal(name = 'in', 
                                    input_temporal_form = 'single_spike', # 'single_spike' or 'constant_rate' or 'arbitrary_spike_train'
                                    spike_times = spike_times)            
                    
            sy = synapse(name = 'sy',
                                synaptic_circuit_inductors = [100e3,100e3,400], # pH
                                synaptic_circuit_resistors = [5e6,4.008e3], # mOhm # [5e3,4.005],
                                synaptic_hotspot_duration = 0.2, # ns
                                synaptic_spd_current = 10, # uA
                                input_direct_connections = ['in'],
                                num_jjs = num_jjs__syn,
                                inhibitory_or_excitatory = 'excitatory',
                                synaptic_dendrite_circuit_inductances = [0,20,200,77.5], # pH
                                synaptic_dendrite_input_synaptic_inductance = [20,1], # pH
                                junction_critical_current = 40,
                                bias_currents = [I_sy, 36, 35],
                                integration_loop_self_inductance = L_si,
                                integration_loop_output_inductance = L_msi,
                                integration_loop_time_constant = tau_si)
            
            ne = neuron(name = 'ne', num_jjs = num_jjs__ne,
                              input_synaptic_connections = ['sy'],
                              input_synaptic_inductances = [[20,1]], # pH
                              junction_critical_current = 40, # uA
                              bias_currents = [I_ne,36,35], # uA
                              circuit_inductances = [0,0,200,77.5], # pH
                              integration_loop_self_inductance = L_ni,
                              integration_loop_time_constant = tau_nf,
                              integration_loop_output_inductance = [400,1], # pH # inductor going into MI feeding threshold circuit [L,k]  
                              neuronal_receiving_input_refractory_inductance = [20,1], # pH
                              threshold_circuit_inductances = [10,0,20], # pH
                              threshold_circuit_resistance = 0.8, # mOhm
                              threshold_circuit_bias_current = 35, # uA
                              threshold_junction_critical_current = 40, # uA
                              time_params = dict([['dt',dt],['tf',tf]])) 
                          
            ne.run_sim()
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare(ne,data_dict) 
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_four(ne,data_dict)
            plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_three(ne,data_dict) 
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_two(ne,data_dict) 
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_one(ne,data_dict) 
            
            # actual_data = np.vstack((input_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
            # error__signal = chi_squared_error(target_data,actual_data)
