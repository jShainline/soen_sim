import numpy as np
from matplotlib import pyplot as plt

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_neuronal_response__single_synaptic_pulse, plot_num_in_burst, plot_phase_portrait
from soen_sim import input_signal, synapse, neuron

plt.close('all')

#%% sim params
dt = 100e-12
spike_times = [5e-9]

num_jjs__syn = 4
num_jjs__ne = 4

if num_jjs__syn == 4:
    L_si_vec = [50e-9] # [50e-9,100e-9,200e-9]
    I_sy_vec = [72e-6] # np.arange(70,85,1)*1e-6 # [73e-6,77e-6,81e-6] # np.arange(73e-6,85e-6,1e-6) # [28e-6,30e-6,32e-6,34e-6,36e-6]
    tau_si_vec = [500e-9] # [500e-9,1e-6,2e-6]
    I_nf = 73e-6
    I_rf = 73e-6
    L_nf = 57.5e-12
    L_ri = 57.5e-12
    tf = 2e-6
    

# num_L_si = 1
# num_I_sy = 1
# num_tau_si = 1
# L_si_vec = np.array([77.5e-9])
# I_sy_vec = np.array([33e-6])
# tau_si_vec = np.array([250e-9])

#%% run loops
num_L_si = len(L_si_vec)
num_I_sy = len(I_sy_vec)
num_tau_si = len(tau_si_vec)
num_in_burst__array = np.zeros([num_L_si,num_tau_si,num_I_sy])
counter = 0
for ii in range(num_L_si):
    L_si = L_si_vec[ii]
    for jj in range(num_tau_si):
        tau_si = tau_si_vec[jj]
        for kk in range(num_I_sy):
            I_sy = I_sy_vec[kk]
            
            counter += 1
            print('ii = {:d} of {:d} (L_si); jj = {:d} of {:d} (tau_si); kk = {:d} of {:d} (I_sy); {:d} of {:d} total'.format(ii+1,num_L_si,jj+1,num_tau_si,kk+1,num_I_sy,counter,num_L_si*num_tau_si*num_I_sy))
            
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
                                integration_loop_output_inductance = 200e-12,
                                integration_loop_time_constant = tau_si)

            neuron_1 = neuron(name = 'ne', num_jjs = num_jjs__ne,
                              input_synaptic_connections = ['sy'],
                              input_synaptic_inductances = [[20e-12,1]],
                              junction_critical_current = 40e-6,
                              bias_currents = [I_nf,36e-6,35e-6],
                              circuit_inductances = [0e-12,0e-12,200e-12,77.5e-12],
                              integration_loop_self_inductance = L_nf,
                              integration_loop_time_constant = 50e-9,
                              integration_loop_output_inductances = [[0e-12,1],[20e-12,1]], # first is to drive latching JJ, second is to drive refractory dendrite; both are of the form [L_self,k]  
                              refractory_dendrite_num_jjs = num_jjs__ne,
                              refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                              refractory_time_constant = 50e-9,
                              refractory_junction_critical_current = 40e-6,
                              refractory_loop_self_inductance = L_ri,
                              refractory_loop_output_inductance = 20e-12,
                              refractory_bias_currents = [I_rf,36e-6,35e-6],
                              refractory_receiving_input_inductance = [20e-12,1],
                              neuronal_receiving_input_refractory_inductance = [20e-12,1],
                              time_params = dict([['dt',dt],['tf',tf]])) 
                          
            neuron_1.run_sim()
            plot_neuronal_response__single_synaptic_pulse(neuron_1)    
            # plot_phase_portrait(neuron_1,I_sy = I_sy,L_si = L_si,tau_si = tau_si)
            # if (   (round(I_sy/1e-6) == 74 and round(L_si/1e-9) == 50 and round(tau_si/1e-9) == 500)  
            #     or (round(I_sy/1e-6) == 74 and round(L_si/1e-9) == 50 and round(tau_si/1e-9) == 1000) 
            #     or (round(I_sy/1e-6) == 74 and round(L_si/1e-9) == 50 and round(tau_si/1e-9) == 2000) 
            #     ):
            #     plot_neuronal_response__single_synaptic_pulse(neuron_1)    
            #     plot_phase_portrait(neuron_1,I_sy = I_sy,L_si = L_si,tau_si = tau_si)
                
                
            num_in_burst__array[ii,jj,kk] = len(neuron_1.spike_times)

#%%

# plot_num_in_burst(I_sy_vec,L_si_vec,tau_si_vec,num_in_burst__array,num_jjs__syn,I_nf)

