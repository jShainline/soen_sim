import numpy as np
from matplotlib import pyplot as plt

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_neuronal_response__single_synaptic_pulse, plot_num_in_burst, plot_phase_portrait, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_two, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_one
from _functions import read_wr_data
from soen_sim import input_signal, synapse, neuron

plt.close('all')

#%% sim params
dt = 100e-12
spike_times = [5e-9]
  
I_sy_vec = np.asarray([76,80])*1e-6 # np.asarray([72,76,80])*1e-6 # [80e-6] # 
I_ne_vec = np.asarray([72,76,80])*1e-6 # [72e-6] # 
tau_si_vec = np.asarray([500,1000,2000])*1e-9 # [1e-6] # 

num_jjs__syn = 4
num_jjs__ne = 4
    
L_si = 77.5e-9 # [10e-9,50e-9] #  np.linspace(100e-9,1e-6,num_L_si) # [7.75e-9,77.5e-9,775e-9,7.75e-6]
L_msi = 100e-12

L_ni = 7.75e-9 # 10e-9 # 
L_mni = 400e-12
tau_nf = 50e-9

tf = 1e-6

#%%

critical_current = 40e-6
current = 0
norm_current = np.max([np.min([current/critical_current,1]),1e-9])
L_jj = (3.2910596281416393e-16/critical_current)*np.arcsin(norm_current)/(norm_current)
r_nf = (30e-12+L_jj)/(50e-9)

directory_name = 'wrspice_data/{:d}jj/single_pulse/'.format(num_jjs__syn)
file_format = 'ne_4jj_1pls_alt_read_no_rd_dir_fb_I_sy{:05.2f}uA_I_nf{:05.2f}uA_tau_si{:07.2f}ns.dat'
for ii in range(len(I_ne_vec)):
    I_ne = I_ne_vec[ii]
    for jj in range(len(I_sy_vec)):
        I_sy = I_sy_vec[jj]
        for kk in range(len(tau_si_vec)):
            tau_si = tau_si_vec[kk]            
            # r_si = (L_si+L_jj)/(tau_si*1e-9)
    
            print('ii = {} of {}; jj = {} of {}; kk = {} of {};'.format(ii+1,len(I_ne_vec),jj+1,len(I_sy_vec),kk+1,len(tau_si_vec)))
            
            file_name = file_format.format(I_sy*1e6,I_ne*1e6,tau_si*1e9)
            data_dict = read_wr_data('{}{}'.format(directory_name,file_name))            
            
            input_1 = input_signal(name = 'in', 
                                    input_temporal_form = 'single_spike', # 'single_spike' or 'constant_rate' or 'arbitrary_spike_train'
                                    spike_times = spike_times)            
                    
            sy = synapse(name = 'sy',
                                synaptic_circuit_inductors = [100e-9,100e-9,250e-12],
                                synaptic_circuit_resistors = [5e3,4.005],
                                synaptic_hotspot_duration = 200e-12,
                                synaptic_spd_current = 10e-6,
                                input_direct_connections = ['in'],
                                num_jjs = num_jjs__syn,
                                inhibitory_or_excitatory = 'excitatory',
                                synaptic_dendrite_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                                synaptic_dendrite_input_synaptic_inductance = [20e-12,1],
                                junction_critical_current = 40e-6,
                                bias_currents = [I_sy, 36e-6, 35e-6],
                                integration_loop_self_inductance = L_si,
                                integration_loop_output_inductance = L_msi,
                                integration_loop_time_constant = tau_si)
            
            ne = neuron(name = 'ne', num_jjs = num_jjs__ne,
                              input_synaptic_connections = ['sy'],
                              input_synaptic_inductances = [[20e-12,1]],
                              junction_critical_current = 40e-6,
                              bias_currents = [I_ne,36e-6,35e-6],
                              circuit_inductances = [0e-12,0e-12,200e-12,77.5e-12],
                              integration_loop_self_inductance = L_ni,
                              integration_loop_time_constant = tau_nf,
                              integration_loop_output_inductance = [400e-12,1], # inductor going into MI feeding threshold circuit [L,k]  
                              neuronal_receiving_input_refractory_inductance = [20e-12,1],
                              threshold_circuit_inductances = [10e-12,0,20e-12],
                              threshold_circuit_resistance = 8e-4,
                              threshold_circuit_bias_current = 35e-6,
                              threshold_junction_critical_current = 40e-6,
                              time_params = dict([['dt',dt],['tf',tf]])) 
                          
            ne.run_sim()
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare(ne,data_dict) 
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_two(ne,data_dict) 
            plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_one(ne,data_dict) 
