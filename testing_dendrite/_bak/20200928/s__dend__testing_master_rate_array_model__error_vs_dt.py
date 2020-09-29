import numpy as np
from matplotlib import pyplot as plt

from _plotting import plot__dend__error_vs_dt
from _functions import read_wr_data
from soen_sim import input_signal, dendrite, neuron #synapse, 

# plt.close('all')
# 
#%%

num_jjs = 4
# dt_vec = np.concatenate([np.arange(0.01e-9,0.11e-9,0.005e-9),np.arange(0.2e-9,1.1e-9,0.05e-9),np.arange(2e-9,11e-9,.05e-9)]) # np.arange(0.2e-9,1.1e-9,0.1e-9) # 

dt_vec = np.logspace(np.log10(0.01e-9),np.log10(10e-9),59)

#%% linear ramp

tf = 50e-9

if num_jjs == 2:
    I_drive_vec = np.asarray([6.2,16])*1e-6
elif num_jjs == 4:
    I_drive_vec = np.asarray([8.6,16])*1e-6
L_di_vec__lin_ramp = np.asarray([77.5,775,7750])*1e-9
tau_di_vec__lin_ramp = np.asarray([77.5e-3,250e-9]) 

error_drive_mat__lin_ramp = np.zeros([len(L_di_vec__lin_ramp),len(tau_di_vec__lin_ramp),len(dt_vec)])
error_mat__lin_ramp = np.zeros([len(L_di_vec__lin_ramp),len(tau_di_vec__lin_ramp),len(dt_vec)])
for jj in range(len(L_di_vec__lin_ramp)):
    for kk in range(len(tau_di_vec__lin_ramp)):
        L_di = L_di_vec__lin_ramp[jj]
        if kk == 0:
            tau_di = L_di/1e-6
        else:
            tau_di = tau_di_vec__lin_ramp[kk]
                
        # load WR data
        directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
        if tau_di > 1e-2:
            file_name = 'dend_lin_ramp_{:d}jj_Idrv{:05.2f}uA{:05.2f}uA_Ide72.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ms_dt01.0ps.dat'.format(num_jjs,I_drive_vec[0]*1e6,I_drive_vec[1]*1e6,L_di*1e9,tau_di*1e3)
        if tau_di < 1e-4:
            file_name = 'dend_lin_ramp_{:d}jj_Idrv{:05.2f}uA{:05.2f}uA_Ide72.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,I_drive_vec[0]*1e6,I_drive_vec[1]*1e6,L_di*1e9,tau_di*1e9)    
        data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
        if num_jjs == 2:
            I_di_str = 'L0#branch'
            I_drive_str = 'L1#branch'
        elif num_jjs == 4:
            I_di_str = 'L2#branch'
            I_drive_str = 'L3#branch'
        target_data = np.vstack((data_dict['time'],data_dict[I_di_str]))
        target_data__drive = np.vstack((data_dict['time'],data_dict[I_drive_str]))        
            
        print('calculating chi^2 norm ...')        
        dt_norm = target_data[0,1]-target_data[0,0]
        norm = 0
        norm__drive = 0
        for rr in range(len(target_data[0,:])):
            norm += np.abs( target_data[1,rr] )**2  
            norm__drive += np.abs( target_data__drive[1,rr] )**2            
        print('done calculating chi^2 norm.')
            
        for qq in range(len(dt_vec)):
            dt = dt_vec[qq]
            print('\n\nlin_ramp: jj = {} of {}; kk = {} of {}; qq = {} of {}'.format(jj+1,len(L_di_vec__lin_ramp),kk+1,len(tau_di_vec__lin_ramp),qq+1,len(dt_vec)))

            # setup soen sim for linear ramp            
            pwl_drive = [[0e-9,0e-6],[1e-9,0e-6],[2e-9,I_drive_vec[0]],[42e-9,I_drive_vec[1]]]
            input_1 = input_signal(name = 'input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)            
    
            dendrite_1 = dendrite('dendrite_under_test', num_jjs = num_jjs,
                                    inhibitory_or_excitatory = 'excitatory', circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12], 
                                    input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                    input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                    input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[20e-12,1]],
                                    thresholding_junction_critical_current = 40e-6, bias_currents = [72e-6,36e-6,35e-6],
                                    integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
                                    integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di)
    
            neuron_1 = neuron('dummy_neuron', input_dendritic_connections = ['dendrite_under_test'], 
                              circuit_inductances = [0e-12,0e-12,200e-12,77.5e-12],
                              input_dendritic_inductances = [[20e-12,1]], 
                              refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                              refractory_time_constant = 50e-9,
                              refractory_thresholding_junction_critical_current = 40e-6,
                              refractory_loop_self_inductance =775e-12,
                              refractory_loop_output_inductance = 100e-12,
                              refractory_bias_currents = [74e-6,36e-6,35e-6],
                              refractory_receiving_input_inductance = [20e-12,1],
                              neuronal_receiving_input_refractory_inductance = [20e-12,1],
                              time_params = dict([['dt',dt],['tf',tf]]))           
            
            neuron_1.run_sim()
                               
            print('calculating chi^2s ...')
            actual_data__drive = np.vstack((neuron_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].direct_connections['input_dendritic_drive'].drive_signal[:])) 
            # error__drive = chi_squared_error(target_data__drive,actual_data__drive)
            dt = actual_data__drive[0,1]-actual_data__drive[0,0]
            error__drive = 0
            for rr in range(len(actual_data__drive[0,:])):
                ind = (np.abs(target_data__drive[0,:]-actual_data__drive[0,rr])).argmin()        
                error__drive += np.abs( target_data__drive[1,ind]-actual_data__drive[1,rr] )**2
            error__drive = dt*error__drive/(dt_norm*norm__drive)
            
            actual_data = np.vstack((input_1.time_vec[:],1e-6*neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
            # error__signal = chi_squared_error(target_data,actual_data)
            dt = actual_data[0,1]-actual_data[0,0]
            error__signal = 0
            for rr in range(len(actual_data[0,:])):
                ind = (np.abs(target_data[0,:]-actual_data[0,rr])).argmin()        
                error__signal += np.abs( target_data[1,ind]-actual_data[1,rr] )**2
            error__signal = dt*error__signal/(dt_norm*norm)                            
            
            error_drive_mat__lin_ramp[jj,kk,qq] = error__drive
            error_mat__lin_ramp[jj,kk,qq] = error__signal
            print('done calculating chi^2s.')

            
#%% sq pls seq

L_di_vec__sq_pls_seq = np.asarray([77.5,775,7750])*1e-9
tau_di_vec__sq_pls_seq = np.asarray([10,100,1000])*1e-9 

error_drive_mat__sq_pls_seq = np.zeros([len(L_di_vec__sq_pls_seq),len(tau_di_vec__sq_pls_seq),len(dt_vec)])
error_mat__sq_pls_seq = np.zeros([len(L_di_vec__sq_pls_seq),len(tau_di_vec__sq_pls_seq),len(dt_vec)])
for jj in range(len(L_di_vec__sq_pls_seq)):
    L_di = L_di_vec__sq_pls_seq[jj]
    
    for kk in range(len(tau_di_vec__sq_pls_seq)):
        tau_di = tau_di_vec__sq_pls_seq[kk]
                
        # load WR data
        directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
        file_name = 'dend_sq_pls_seq_{:d}jj_Ide74.00uA_Ldr20pH20pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,L_di*1e9,tau_di*1e9)
        data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
        if num_jjs == 2:
            I_di_str = 'L0#branch'
            I_drive_str = 'L1#branch'
        elif num_jjs == 4:
            I_di_str = 'L2#branch'
            I_drive_str = 'L3#branch'
        tf = data_dict['time'][-1]
        target_data = np.vstack((data_dict['time'],data_dict[I_di_str]))
        target_data__drive = np.vstack((data_dict['time'],data_dict[I_drive_str]))
        
        print('calculating chi^2 norm ...')        
        dt_norm = target_data[0,1]-target_data[0,0]
        norm = 0
        norm__drive = 0
        for rr in range(len(target_data[0,:])):
            norm += np.abs( target_data[1,rr] )**2  
            norm__drive += np.abs( target_data__drive[1,rr] )**2            
        print('done calculating chi^2 norm.')

        for qq in range(len(dt_vec)):
            dt = dt_vec[qq]
            print('\n\nsq_pls_seq: jj = {} of {}; kk = {} of {}; qq = {} of {}'.format(jj+1,len(L_di_vec__sq_pls_seq),kk+1,len(tau_di_vec__sq_pls_seq),qq+1,len(dt_vec)))
            
            # setup soen sim for square pulses         
            pwl_drive = [[0e-9,0e-6],[4.9e-9,0e-6],[5e-9,7.84e-6],[15e-9,7.84e-6],[15.1e-9,0e-6],[24.9e-9,0e-6],[25e-9,13.21e-6],[35e-9,13.21e-6],[35.1e-9,0e-6],[44.9e-9,0e-6],[45e-9,9.17e-6],[55e-9,9.17e-6],[55.1e-9,0e-6],[64.9e-9,0e-6],[65e-9,15.84e-6],[75e-9,15.84e-6],[75.1e-9,0e-6],]
            input_1 = input_signal(name = 'input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)            
    
            dendrite_1 = dendrite('dendrite_under_test', num_jjs = num_jjs,
                                    inhibitory_or_excitatory = 'excitatory', circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12], 
                                    input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                    input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                    input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[20e-12,1]],
                                    thresholding_junction_critical_current = 40e-6, bias_currents = [74e-6,36e-6,35e-6],
                                    integration_loop_self_inductance = L_di, integration_loop_output_inductance = 0e-12,
                                    integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di)
    
            neuron_1 = neuron('dummy_neuron', input_dendritic_connections = ['dendrite_under_test'], 
                              circuit_inductances = [0e-12,0e-12,200e-12,77.5e-12],
                              input_dendritic_inductances = [[20e-12,1]], 
                              refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                              refractory_time_constant = 50e-9,
                              refractory_thresholding_junction_critical_current = 40e-6,
                              refractory_loop_self_inductance =775e-12,
                              refractory_loop_output_inductance = 100e-12,
                              refractory_bias_currents = [74e-6,36e-6,35e-6],
                              refractory_receiving_input_inductance = [20e-12,1],
                              neuronal_receiving_input_refractory_inductance = [20e-12,1],
                              time_params = dict([['dt',dt],['tf',tf]]))           
            
            neuron_1.run_sim()
                                    
            print('calculating chi^2s ...')
            actual_data__drive = np.vstack((neuron_1.time_vec[:],neuron_1.dendrites['dendrite_under_test'].direct_connections['input_dendritic_drive'].drive_signal[:])) 
            # error__drive = chi_squared_error(target_data__drive,actual_data__drive)
            dt = actual_data__drive[0,1]-actual_data__drive[0,0]
            error__drive = 0
            for rr in range(len(actual_data__drive[0,:])):
                ind = (np.abs(target_data__drive[0,:]-actual_data__drive[0,rr])).argmin()        
                error__drive += np.abs( target_data__drive[1,ind]-actual_data__drive[1,rr] )**2
            error__drive = dt*error__drive/(dt_norm*norm__drive)
            
            actual_data = np.vstack((input_1.time_vec[:],1e-6*neuron_1.dendrites['dendrite_under_test'].I_di_vec[:]))    
            # error__signal = chi_squared_error(target_data,actual_data)
            dt = actual_data[0,1]-actual_data[0,0]
            error__signal = 0
            for rr in range(len(actual_data[0,:])):
                ind = (np.abs(target_data[0,:]-actual_data[0,rr])).argmin()        
                error__signal += np.abs( target_data[1,ind]-actual_data[1,rr] )**2
            error__signal = dt*error__signal/(dt_norm*norm)                            
            
            error_drive_mat__sq_pls_seq[jj,kk,qq] = error__drive
            error_mat__sq_pls_seq[jj,kk,qq] = error__signal
            print('done calculating chi^2s.')    

#%%
# plt.close('all')
plot__dend__error_vs_dt(np.asarray(dt_vec),error_mat__lin_ramp,error_drive_mat__lin_ramp,L_di_vec__lin_ramp,tau_di_vec__lin_ramp,'num_jjs = {:d}, lin_ramp'.format(num_jjs))
plot__dend__error_vs_dt(np.asarray(dt_vec),error_mat__sq_pls_seq,error_drive_mat__sq_pls_seq,L_di_vec__sq_pls_seq,tau_di_vec__sq_pls_seq,'num_jjs = {:d}, sq_pls_seq'.format(num_jjs))



#%%

# sq_pls_trn_params = dict()
# sq_pls_trn_params['t_start'] = 5e-9
# sq_pls_trn_params['t_rise'] = 1e-9
# sq_pls_trn_params['t_pulse'] = 5e-9
# sq_pls_trn_params['t_fall'] = 1e-9
# sq_pls_trn_params['t_period'] = 20e-9
# sq_pls_trn_params['value_off'] = 0
# sq_pls_trn_params['value_on'] = 23.81e-6