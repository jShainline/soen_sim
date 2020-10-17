import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_num_in_burst, plot_phase_portrait, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_two, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_one, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_four, plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_three
from _functions import read_wr_data, chi_squared_error__ISI
from soen_sim import input_signal, synapse, neuron
from _util import color_dictionary
colors = color_dictionary()

plt.close('all')

#%% sim params
dt = 0.1 # ns
spike_times = [5] # ns
  
I_sy_vec = np.asarray([72,76,80]) # np.asarray([72,76,80])*1e-6 # [80e-6] # 
I_ne_vec = np.asarray([73]) # np.asarray([72,73,84])
tau_si_vec = np.asarray([500,1000,2000]) # np.asarray([500,1000,2000]) # [1e-6] # 

num_jjs__syn = 4
num_jjs__ne = 4
    
L_si = 77.5e3 # pH # [10e-9,50e-9] #  np.linspace(100e-9,1e-6,num_L_si) # [7.75e-9,77.5e-9,775e-9,7.75e-6]
L_msi = 130 # pH

L_ni = 7.75e3 # pH # 10e-9 # 
L_mni = 400 # pH
tau_nf = 50 # ns

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
    
            print('\n\nii = {} of {}; jj = {} of {}; kk = {} of {};'.format(ii+1,len(I_ne_vec),jj+1,len(I_sy_vec),kk+1,len(tau_si_vec)))
            
            file_name = file_format.format(I_sy,I_ne,tau_si,k__nr_nf)
            data_dict = read_wr_data('{}{}'.format(directory_name,file_name))   
            time_vec__wr = 1e9*data_dict['time']
            dt__wr = time_vec__wr[1]-time_vec__wr[0]
            I_nf__wr = 1e6*data_dict['L14#branch']
            I_nf_peaks__wr, _ = find_peaks(I_nf__wr, height = 30, distance = 30/dt__wr)
                        
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
            
            threshold_circuit = 'hTron'
            
            if threshold_circuit == 'JJ':
                
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
                                  time_params = dict([['dt',dt],['tf',time_vec__wr[-1]]])) 
                ne.run_sim()
            
            elif threshold_circuit == 'hTron':
                
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
                                  threshold_circuit = 'hTron',
                                  threshold_circuit_inductances = [200,400,10,100e3,125], # pH
                                  threshold_circuit_time_constants = [50,50], # ns
                                  threshold_circuit_bias_current = 20, # uA
                                  threshold_circuit_hTron_resistances = [1e6,5e6], # [gate,channel] # mohm
                                  threshold_circuit_hTron_current_thresholds = [3,0.3,16,4], # [I_gate_on,I_gate_off,I_channel_on,I_channel_off] # uA
                                  time_params = dict([['dt',dt],['tf',time_vec__wr[-1]]]))
                ne.run_sim()
                
                plot_time_traces = True
                if plot_time_traces == True:
                    
                    fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)
                    # fig.suptitle('num_jjs = {}'.format(num_jjs))
                    
                    ax[0].plot(ne.time_vec,ne.I_ni_vec, '-', color = colors['red3'])
                    ax[0].set_ylabel(r'$I_{drive}$ [$\mu$A]')
                    
                    ax[1].plot(ne.time_vec,ne.threshold_I_1, '-', color = colors['blue3'])
                    ax[1].set_ylabel(r'$I_{gate}$ [$\mu$A]')
                         
                    ax[2].plot(ne.time_vec,ne.threshold_I_3, '-', color = colors['green3'])
                    ax[2].set_ylabel(r'$I_{th}$ [$\mu$A]')
                            
                    ax[2].set_xlabel(r'Time [ns]')
                    # ax[2].set_xlim([0,100])
                    
                    plt.show()
                          
            error = chi_squared_error__ISI(time_vec__wr[I_nf_peaks__wr],ne.spike_times)
            
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare(ne,data_dict,error) 
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_four(ne,data_dict,error)
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_three(ne,data_dict,error,I_nf_peaks__wr) 
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_two(ne,data_dict,error) 
            # plot_neuronal_response__single_synaptic_pulse__no_rd__direct_feedback__wr_compare__just_one(ne,data_dict,error) 
            
            # plot_phase_portrait(ne, I_sy = I_sy, L_si = L_si, tau_si = tau_si, I_ne = I_ne)
                

            
