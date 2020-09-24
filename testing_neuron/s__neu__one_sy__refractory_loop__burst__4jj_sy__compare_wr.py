import numpy as np
from matplotlib import pyplot as plt

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_neuronal_response__single_synaptic_pulse, plot_num_in_burst, plot_phase_portrait, plot_neuronal_response__single_synaptic_pulse__wr_compare
from _functions import read_wr_data
from soen_sim import input_signal, synapse, neuron

plt.close('all')

#%% sim params
dt = 100e-12
spike_times = [5e-9]

num_jjs__syn = 4
num_jjs__ne = 4
    
L_si_vec = [50e-9] # [10e-9,50e-9] #  np.linspace(100e-9,1e-6,num_L_si) # [7.75e-9,77.5e-9,775e-9,7.75e-6]
L_msi = 200e-12
tau_si_vec = [0.5e-6] # [1e-6,2e-6,4e-6] # np.linspace(2100e-9,2500e-9,num_tau_si)
I_sy_vec = [72e-6] # np.arange(70,79,1)*1e-6 # [73e-6,77e-6,81e-6] # np.arange(73e-6,85e-6,1e-6) # [28e-6,30e-6,32e-6,34e-6,36e-6]

I_nf = 76e-6
L_nf = 57.5e-12 # 10e-9 # 
L_mnf = 20e-12
tau_nf = 50e-9

I_rf = 73e-6
L_ri = 40e-9 # 10e-9 # 
L_mri = 20e-12
tau_ri = 60e-9

tf = 300e-9    

# neu_4jj_Isy72uA_Lsi0050.00nH_Lmsi200pH_tausi0500ns_Inf76uA_Lnf57.50pH_Lmnf20pH_taunf50ns_Irf73uA_Lri0010.00nH_Lmri20pH_tauri50ns.dat
#%% run loops
num_L_si = len(L_si_vec)
num_I_sy = len(I_sy_vec)
num_tau_si = len(tau_si_vec)
counter = 0
for ii in range(num_L_si):
    L_si = L_si_vec[ii]
    for jj in range(num_tau_si):
        tau_si = tau_si_vec[jj]
        for kk in range(num_I_sy):
            I_sy = I_sy_vec[kk]
            
            counter += 1
            print('ii = {:d} of {:d} (L_si); jj = {:d} of {:d} (tau_si); kk = {:d} of {:d} (I_sy); {:d} of {:d} total'.format(ii+1,num_L_si,jj+1,num_tau_si,kk+1,num_I_sy,counter,num_L_si*num_tau_si*num_I_sy))
            
            # wr
            directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs__syn)
            file_name = 'neu_{:d}jj_Isy{:02.0f}uA_Lsi{:07.2f}nH_Lmsi{:03.0f}pH_tausi{:04.0f}ns_Inf{:02.0f}uA_Lnf{:05.2f}pH_Lmnf{:02.0f}pH_taunf{:02.0f}ns_Irf{:02.0f}uA_Lri{:07.2f}nH_Lmri{:02.0f}pH_tauri{:02.0f}ns.dat'.format(num_jjs__syn,I_sy*1e6,L_si*1e9,L_msi*1e12,tau_si*1e9,I_nf*1e6,L_nf*1e12,L_mnf*1e12,tau_nf*1e9,I_rf*1e6,L_ri*1e9,L_mri*1e12,tau_ri*1e9)
            data_dict = read_wr_data('{}{}'.format(directory_name,file_name))            
            
            input_1 = input_signal(name = 'in', 
                                    input_temporal_form = 'single_spike', # 'single_spike' or 'constant_rate' or 'arbitrary_spike_train'
                                    spike_times = spike_times)            
                    
            synapse_1 = synapse(name = 'sy',
                                synaptic_circuit_inductors = [100e-9,100e-9,250e-12],
                                synaptic_circuit_resistors = [5e3,4.008],
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

            neuron_1 = neuron(name = 'ne', num_jjs = num_jjs__ne,
                              input_synaptic_connections = ['sy'],
                              input_synaptic_inductances = [[20e-12,1]],
                              junction_critical_current = 40e-6,
                              bias_currents = [I_nf,32e-6,35e-6],
                              circuit_inductances = [0e-12,0e-12,77.5e-12,77.5e-12],
                              integration_loop_self_inductance = L_nf,
                              integration_loop_time_constant = tau_nf,
                              integration_loop_output_inductances = [[0e-12,1],[20e-12,1]], # first is to drive latching JJ, second is to drive refractory dendrite; both are of the form [L_self,k]  
                              refractory_dendrite_num_jjs = 4,
                              refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                              refractory_time_constant = tau_ri,
                              refractory_junction_critical_current = 40e-6,
                              refractory_loop_self_inductance = L_ri,
                              refractory_loop_output_inductance = 20e-12,
                              refractory_bias_currents = [I_rf,36e-6,35e-6], # I_rf
                              refractory_receiving_input_inductance = [20e-12,1],
                              neuronal_receiving_input_refractory_inductance = [20e-12,1],
                              time_params = dict([['dt',dt],['tf',tf]])) 
                          
            neuron_1.run_sim()
            plot_neuronal_response__single_synaptic_pulse__wr_compare(neuron_1,data_dict) 
                
                
