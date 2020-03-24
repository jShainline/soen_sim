#%%
import numpy as np
from matplotlib import pyplot as plt
from pylab import *

from soen_sim import input_signal, synapse, neuron
from _functions import load_neuron_data

plt.close('all')

#%% temporal
num_tau_sim = 5
observation_duration = 20e-6
dt = 10e-9

tau_ref_vec = [100e-9]#[25e-9,50e-9,100e-9,200e-9]#
I_sy_vec = [37e-6]#np.linspace(34e-6,39e-6,6)#
tau_si_vec = [3000e-9]#np.linspace(1000e-9,4000e-9,4)#
rate_vec = [1e6]#np.linspace(2e5,20e6,25)#np.array([1e6])#np.logspace(np.log10(2e5),np.log10(2e7),10)#[9.6e6]#
jitter_params = [0,25e-9]#[gaussian center, gaussian deviation]
num_synapses_tot = 9

num_spikes_in_mat = np.zeros([len(rate_vec),len(tau_si_vec),len(I_sy_vec),len(tau_ref_vec),num_synapses_tot])
num_spikes_out_mat = np.zeros([len(rate_vec),len(tau_si_vec),len(I_sy_vec),len(tau_ref_vec),num_synapses_tot])
isi_output_mat = 1e9*np.ones([len(rate_vec),len(tau_si_vec),len(I_sy_vec),len(tau_ref_vec),num_synapses_tot])
isi_output_avg_mat = 1e9*np.ones([len(rate_vec),len(tau_si_vec),len(I_sy_vec),len(tau_ref_vec),num_synapses_tot])
# isi_input_vec = 1/rate_vec

#%% run it
num_id = 0   
for qq in range(len(tau_ref_vec)):
    print('\n\n\n\n\n\nqq = {} of {} (tau_ref)'.format(qq+1,len(tau_ref_vec)))
    tau_ref = tau_ref_vec[qq]

    for kk in range(len(I_sy_vec)):
        print('\n\n\n\nkk = {} of {} (I_sy)'.format(kk+1,len(I_sy_vec)))
        I_sy = I_sy_vec[kk]
        
        for ii in range(len(tau_si_vec)):
            print('\n\nii = {} of {} (tau_si)\n'.format(ii+1,len(tau_si_vec)))
            tau_si = tau_si_vec[ii]
            
            for jj in range(len(rate_vec)):
                print('jj = {} of {} (rate_in)\n'.format(jj+1,len(rate_vec)))
                rate = rate_vec[jj]                
                        
                for rr in range(num_synapses_tot):
                    print('rr = {} of {} (num_synapses_active)'.format(rr+1,num_synapses_tot))
                    num_synapses_active = rr+1
                    
                    num_id += 1
                    input_synapses = []
                    input_inductances = []
                    for pp in range(num_synapses_tot):
                        
                        # initialize input signals
                        name__i = 'input_signal__{:d}_{:d}'.format(num_id,pp)
                        time_last_spike = num_tau_sim*tau_si_vec[ii]+observation_duration+2*1/rate_vec[jj]
                        input_1 = input_signal(name__i, input_temporal_form = 'constant_rate', spike_times = [rate_vec[jj],time_last_spike],
                        stochasticity = 'gaussian', jitter_params = jitter_params)   
                        # print(input_1.spike_times)
                        
                        # initialize synapses
                        name__s = 'input_synapse__{:d}_{:d}'.format(num_id,pp)
                        if pp+1 <= num_synapses_active:
                            input_signal_name = name__i
                        else:
                            input_signal_name = ''
                        synapse_1 = synapse(name__s, loop_temporal_form = 'exponential', time_constant = tau_si_vec[ii], 
                                            integration_loop_self_inductance = 50e-9, integration_loop_output_inductance = 200e-12, 
                                            synaptic_bias_current = I_sy_vec[kk], loop_bias_current = 31e-6,
                                            input_signal_name = input_signal_name)
                        
                        input_synapses.append(name__s)
                        coupling_factor = 0.6#-0.0375*num_synapses_tot+0.5375
                        input_inductances.append([10e-12,coupling_factor])
                    
                    #initialize neuron
                    input_inductances.append([20e-12,1])
                    name__n = 'rate_encoding_neuron__{:d}'.format(num_id)
                    neuron_1 = neuron(name__n, input_connections = input_synapses, input_inductances = input_inductances,
                                      thresholding_junction_critical_current = 40e-6, threshold_bias_current = 36.5e-6, 
                                      refractory_temporal_form = 'exponential', refractory_time_constant = tau_ref_vec[qq], 
                                      refractory_loop_self_inductance = 10e-9, refractory_loop_output_inductance = 200e-12)
                           
                    # propagate in time                         
                    sim_params = dict()
                    sim_params['dt'] = dt
                    sim_params['pre_observation_duration'] = num_tau_sim*synapse_1.time_constant
                    sim_params['observation_duration'] = observation_duration
                    sim_params['num_tau_sim'] = num_tau_sim
                    neuron_1.sim_params = sim_params
                    neuron_1.run_sim()
                    
                    # plot temporal response
                    # plot_save_string = 'I_sy={:2.2f}uA__tau_si={:04.2f}ns__tau_ref={:04.2f}ns__rate_in={:04.2f}MHz__dt={:04.2f}ns__obs={:04.2f}us__jitter={}ns'.format(1e6*I_sy_vec[kk],1e9*tau_si_vec[ii],1e9*tau_ref_vec[qq],1e-6*rate_vec[jj],1e9*dt,observation_duration*1e6,jitter_params[1])
                    # neuron_1.plot_receiving_loop_current(plot_save_string)            
                    # neuron_1.plot_spike_train(plot_save_string)
                    
                    # fill observation matrices                
                    num_spikes_in_mat[jj,ii,kk,qq,rr] = neuron_1.synapses[0].num_spikes
                    num_spikes_out_mat[jj,ii,kk,qq,rr] = neuron_1.num_spikes
                    if neuron_1.num_spikes > 1:
                        isi_output_mat[jj,ii,kk,qq,rr] = neuron_1.isi_output__last_two
                        isi_output_avg_mat[jj,ii,kk,qq,rr] = neuron_1.isi_output__avg
                      
                    #save neuron data
                    data_save_string = 'I_sy={:2.2f}uA__tau_si={:04.2f}ns__tau_ref={:04.2f}ns__rate_in={:04.2f}MHz__dt={:04.2f}ns__obs={:04.2f}us__jitter={}ns__num_sy_tot={}__num_sy_active={}'.format(1e6*I_sy_vec[kk],1e9*tau_si_vec[ii],1e9*tau_ref_vec[qq],1e-6*rate_vec[jj],1e9*dt,observation_duration*1e6,jitter_params[1],num_synapses_tot,num_synapses_active)
                    neuron_1.save_neuron_data(data_save_string)
                    
                    #delete synapses and neurons
                    # del synapse_1, neuron_1
      
                # plot
                neuron_1.num_active_synapses_vec = np.arange(1,num_synapses_tot+1,1)
                neuron_1.rate = rate
                neuron_1.tau_si = tau_si
                neuron_1.tau_ref = tau_ref
                neuron_1.I_sy = I_sy
                neuron_1.isi_output_avg_vec = isi_output_avg_mat[jj,ii,kk,qq,:]
                plot_save_string = 'I_sy={:2.2f}-{:02.2f}uA__tau_si={:04.2f}-{:04.2f}ns__tau_ref={:04.2f}-{:04.2f}ns__rate_in={:04.2f}-{:04.2f}MHz__dt={:04.2f}ns__obs={:04.2f}us__jitter={}ns__num_sy={}'.format(1e6*I_sy_vec[0],1e6*I_sy_vec[-1],1e9*tau_si_vec[0],1e9*tau_si_vec[-1],1e9*tau_ref_vec[0],1e9*tau_ref_vec[-1],1e-6*rate_vec[0],1e-6*rate_vec[-1],1e9*dt,observation_duration*1e6,jitter_params[1],num_synapses_tot)
                neuron_1.plot_rate_vs_num_active_synapses(plot_save_string = plot_save_string)
                # neuron_1.plot_rate_transfer_function__no_lines(plot_save_string = plot_save_string)


