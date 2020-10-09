import numpy as np
from matplotlib import pyplot as plt

from _plotting import plot_wr_comparison__synapse, plot_wr_comparison__synapse__n_fq_vs_I_de, plot__syn__wr_cmpr__single_pulse, plot__syn__wr_cmpr__pulse_train
from _functions import read_wr_data, chi_squared_error
from soen_sim import input_signal, synapse, neuron # dendrite, 
from util import physical_constants

p = physical_constants()

plt.close('all')

#%%
num_jjs = 4

dt = 0.1 # 0.1
I_spd = 10

#%% single pulse

spike_times = [5]

if num_jjs == 2:
    I_de_vec = np.asarray([70])
elif num_jjs == 4:
    I_de_vec = np.asarray([70])
    
L_di_vec = np.asarray([7.75,77.5,775,7750])*1e3 # np.asarray([7.75e3]) # 
tau_di_vec = np.asarray([10,50,250,1250]) # np.asarray([10]) #  available: 10,25,50,250,1250

soen_response__sp = []
soen_time__sp = []
wr_response__sp = []
wr_time__sp = []
chi_signal__sp = np.zeros([len(L_di_vec),len(tau_di_vec)])
for ii in range(len(I_de_vec)):
    I_de = I_de_vec[ii]
    for jj in range(len(L_di_vec)):
        L_di = L_di_vec[jj]
        soen_response__sp.append([])
        soen_time__sp.append([])
        wr_response__sp.append([])
        wr_time__sp.append([])
        for kk in range(len(tau_di_vec)): 
            tau_di = tau_di_vec[kk]
            print('\n\nii = {} of {}; jj = {} of {}; kk = {} of {}'.format(ii+1,len(I_de_vec),jj+1,len(L_di_vec),kk+1,len(tau_di_vec)))
            
            # load WR data
            directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
            file_name = 'syn_one_pls_{:d}jj_Ispd{:05.2f}uA_Ide{:05.2f}uA_Ldr20.0pH20.0pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,I_spd,I_de,L_di*1e-3,tau_di)
            data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
            if num_jjs == 2:
                I_di_str = 'L1#branch'
                I_drive_str = 'L3#branch'
            elif num_jjs == 4:
                I_di_str = 'L3#branch'
                I_drive_str = 'L0#branch'
            target_data = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_di_str]))
            tf = np.round(1e9*data_dict['time'][-1],0)
            target_data__drive = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_drive_str]))

            # setup soen sim for exp pulse seq
            input_1 = input_signal(name = 'in', 
                                   input_temporal_form = 'single_spike', # 'single_spike' or 'constant_rate' or 'arbitrary_spike_train'
                                   spike_times = spike_times)            
        
            sy = synapse(name = 'sy',
                                synaptic_circuit_inductors = [100e3,100e3,400],
                                synaptic_circuit_resistors = [5e6,4.008e3],
                                synaptic_hotspot_duration = 0.2,
                                synaptic_spd_current = 10,
                                input_direct_connections = ['in'],
                                num_jjs = num_jjs,
                                inhibitory_or_excitatory = 'excitatory',
                                synaptic_dendrite_circuit_inductances = [0,20,200,77.5],
                                synaptic_dendrite_input_synaptic_inductance = [20,1],
                                junction_critical_current = 40,
                                bias_currents = [I_de, 36, 35],
                                integration_loop_self_inductance = L_di,
                                integration_loop_output_inductance = 0,
                                integration_loop_time_constant = tau_di)
       
            ne = neuron('ne',
                              input_synaptic_connections = ['sy'],
                              input_synaptic_inductances = [[20,1]],
                              junction_critical_current = 40,
                              circuit_inductances = [0,0,200,77.5],                              
                              refractory_loop_circuit_inductances = [0,20,200,77.5],
                              refractory_time_constant = 50,
                              refractory_thresholding_junction_critical_current = 40,
                              refractory_loop_self_inductance = 775,
                              refractory_loop_output_inductance = 100,
                              refractory_bias_currents = [74,36,35],
                              refractory_receiving_input_inductance = [20,1],
                              neuronal_receiving_input_refractory_inductance = [20,1],
                              integration_loop_time_constant = 25,
                              time_params = dict([['dt',dt],['tf',tf]]))           
            
            ne.run_sim()
                                    
            actual_data__drive = np.vstack((ne.time_vec[:],ne.synapses['sy'].I_spd2_vec[:])) 
            if ii == 0 and jj == 0 and kk == 0:
                error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                chi_drive__sp = error__drive                               
            
            actual_data = np.vstack((ne.time_vec[:],ne.synapses['sy'].I_di_vec[:]))    
            error__signal = chi_squared_error(target_data,actual_data)
            chi_signal__sp[jj,kk] = error__signal
            
            soen_response__sp[jj].append(ne.synapses['sy'].I_di_vec[:])
            soen_time__sp[jj].append(ne.time_vec[:])
            wr_response__sp[jj].append(1e6*data_dict[I_di_str])
            wr_time__sp[jj].append(1e9*data_dict['time'])
            
            # plot_wr_comparison__synapse(file_name,spike_times,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)
            
plot__syn__wr_cmpr__single_pulse(soen_time__sp,ne.synapses['sy'].I_spd2_vec[:],soen_response__sp,wr_time__sp,1e6*data_dict[I_drive_str],wr_response__sp,L_di_vec,tau_di_vec,chi_drive__sp,chi_signal__sp,num_jjs)
            
#%% pls seq

spike_times = [5,85,285,335]

if num_jjs == 2:
    I_de_vec = np.asarray([68,78])
elif num_jjs == 4:
    I_de_vec = np.asarray([72,82])
    
L_di_vec = np.asarray([77.5,7750])*1e3 # 
tau_di_vec = np.asarray([250]) # 10,25,50,

soen_response__ps = []
soen_time__ps = []
wr_response__ps = []
wr_time__ps = []
chi_signal__ps = np.zeros([len(I_de_vec),len(L_di_vec)])
for ii in range(len(I_de_vec)):
    I_de = I_de_vec[ii]
    soen_response__ps.append([])
    soen_time__ps.append([])
    wr_response__ps.append([])
    wr_time__ps.append([])
    for jj in range(len(L_di_vec)):
        L_di = L_di_vec[jj]
        for kk in range(len(tau_di_vec)): 
            tau_di = tau_di_vec[kk]
            print('\n\nii = {} of {}; jj = {} of {}; kk = {} of {}'.format(ii+1,len(I_de_vec),jj+1,len(L_di_vec),kk+1,len(tau_di_vec)))
            
            # load WR data
            directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
            file_name = 'syn_pls_seq_{:d}jj_Ispd{:05.2f}uA_M200pH_Ide{:05.2f}uA_Ldr20.0pH20.0pH_Ldi{:07.2f}nH_taudi{:07.2f}ns_dt01.0ps.dat'.format(num_jjs,I_spd,I_de,L_di*1e-3,tau_di)
            data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
            if num_jjs == 2:
                I_di_str = 'L1#branch'
                I_drive_str = 'L3#branch'
            elif num_jjs == 4:
                I_di_str = 'L3#branch'
                I_drive_str = 'L0#branch'
            target_data = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_di_str]))
            tf = np.round(1e9*data_dict['time'][-1])
            target_data__drive = np.vstack((1e9*data_dict['time'],1e6*data_dict[I_drive_str]))

            # setup soen sim for exp pulse seq
            input_1 = input_signal(name = 'in', 
                                   input_temporal_form = 'arbitrary_spike_train', # 'single_spike' or 'constant_rate' or 'arbitrary_spike_train'
                                   spike_times = spike_times)            
        
            sy = synapse(name = 'sy',
                                synaptic_circuit_inductors = [100e3,100e3,200],
                                synaptic_circuit_resistors = [5e6,4.008*1e3],
                                synaptic_hotspot_duration = 0.2,
                                synaptic_spd_current = 10,
                                input_direct_connections = ['in'],
                                num_jjs = num_jjs,
                                inhibitory_or_excitatory = 'excitatory',
                                synaptic_dendrite_circuit_inductances = [0,20,200,77.5],
                                synaptic_dendrite_input_synaptic_inductance = [20,1],
                                junction_critical_current = 40,
                                bias_currents = [I_de, 36, 35],
                                integration_loop_self_inductance = L_di,
                                integration_loop_output_inductance = 0,
                                integration_loop_time_constant = tau_di)
       
            ne = neuron('ne',
                              input_synaptic_connections = ['sy'],
                              input_synaptic_inductances = [[20,1]],
                              junction_critical_current = 40,
                              circuit_inductances = [0,0,200,77.5],                              
                              refractory_loop_circuit_inductances = [0,20,200,77.5],
                              refractory_time_constant = 50,
                              refractory_thresholding_junction_critical_current = 40,
                              refractory_loop_self_inductance = 775,
                              refractory_loop_output_inductance = 100,
                              refractory_bias_currents = [74,36,35],
                              refractory_receiving_input_inductance = [20,1],
                              neuronal_receiving_input_refractory_inductance = [20,1],
                              integration_loop_time_constant = 50,
                              time_params = dict([['dt',dt],['tf',tf]]))           
            
            ne.run_sim()
                                    
            actual_data__drive = np.vstack((ne.time_vec[:],ne.synapses['sy'].I_spd2_vec[:])) 
            if ii == 0 and jj == 0 and kk == 0:
                error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                chi_drive__ps = error__drive
                                            
            actual_data = np.vstack((ne.time_vec[:],ne.synapses['sy'].I_di_vec[:]))    
            error__signal = chi_squared_error(target_data,actual_data)
            chi_signal__ps[ii,jj] = error__signal
            
            soen_response__ps[ii].append(ne.synapses['sy'].I_di_vec[:])
            soen_time__ps[ii].append(ne.time_vec[:])
            wr_response__ps[ii].append(1e6*data_dict[I_di_str])
            wr_time__ps[ii].append(1e9*data_dict['time'])
            
            # plot_wr_comparison__synapse(file_name,spike_times,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)
            
plot__syn__wr_cmpr__pulse_train(soen_time__ps,ne.synapses['sy'].I_spd2_vec[:],soen_response__ps,wr_time__ps,1e6*data_dict[I_drive_str],wr_response__ps,L_di_vec,I_de_vec,chi_drive__ps,chi_signal__ps,num_jjs)
            

#%% n_fq vs I_de

L_di = 7750e3
tau_di = 7750e3

spike_times = [5]

num_jjs = 4
if num_jjs == 2:
    I_de_array = [np.arange(65,80,1),np.arange(57,80,1)]
elif num_jjs == 4:
    I_de_array = [np.arange(70,84,1),np.arange(61,84,1)]

M_vec = [200,400]

n_fq_1_array = []
n_fq_2_array = []
n_fq_soen_array = []
for ii in range(len(M_vec)):
    M = M_vec[ii]
    n_fq_1_array.append([])
    n_fq_2_array.append([])
    n_fq_soen_array.append([])
    I_de_vec = I_de_array[ii]
    for jj in range(len(I_de_vec)):
        I_de = I_de_vec[jj]
       
        print('\n\nii = {} of {}; jj = {} of {}'.format(ii+1,len(M_vec),jj+1,len(I_de_vec)))
        
        # load WR data
        directory_name = 'wrspice_data/{:d}jj/'.format(num_jjs)
        file_name = 'syn_one_pls_{:d}jj_Ispd{:05.2f}uA_M{:3.0f}pH_Ide{:05.2f}uA_Ldr20.0pH20.0pH_Ldi7750.00nH_taudi7750.00ms_dt01.0ps.dat'.format(num_jjs,I_spd,M,I_de)
        data_dict = read_wr_data('{}{}'.format(directory_name,file_name))
        if num_jjs == 2:
            I_di_str = 'L1#branch'
        elif num_jjs == 4:
            I_di_str = 'L3#branch'
        time_vec = 1e9*data_dict['time']
        t0_ind = ( np.abs( time_vec[:]-4.9 ) ).argmin()
        I_di = 1e6*data_dict[I_di_str]
        I0 = I_di[t0_ind]
        If = I_di[-1]
        n_fq_1 = (If-I0)/(p['Phi0__pH_ns']/L_di)
        n_fq_1_array[ii].append(n_fq_1)
                                
        # setup soen sim for exp pulse seq
        input_1 = input_signal(name = 'in', 
                               input_temporal_form = 'single_spike', # 'single_spike' or 'constant_rate' or 'arbitrary_spike_train'
                               spike_times = spike_times)            
    
        sy = synapse(name = 'sy',
                            synaptic_circuit_inductors = [100e3,100e3,M],
                            synaptic_circuit_resistors = [5e6,4.008e3],
                            synaptic_hotspot_duration = 0.2,
                            synaptic_spd_current = 10,
                            input_direct_connections = ['in'],
                            num_jjs = num_jjs,
                            inhibitory_or_excitatory = 'excitatory',
                            synaptic_dendrite_circuit_inductances = [0,20,200,77.5],
                            synaptic_dendrite_input_synaptic_inductance = [20,1],
                            junction_critical_current = 40,
                            bias_currents = [I_de, 36, 35],
                            integration_loop_self_inductance = L_di,
                            integration_loop_output_inductance = 0,
                            integration_loop_time_constant = tau_di)
   
        ne = neuron('ne',
                          input_synaptic_connections = ['sy'],
                          input_synaptic_inductances = [[20,1]],
                          junction_critical_current = 40,
                          circuit_inductances = [0,0,200,77.5],                              
                          refractory_loop_circuit_inductances = [0,20,200,77.5],
                          refractory_time_constant = 50,
                          refractory_thresholding_junction_critical_current = 40,
                          refractory_loop_self_inductance = 775,
                          refractory_loop_output_inductance = 100,
                          refractory_bias_currents = [74,36,35],
                          refractory_receiving_input_inductance = [20,1],
                          neuronal_receiving_input_refractory_inductance = [20,1],
                          integration_loop_time_constant = 50,
                          time_params = dict([['dt',dt],['tf',tf]]))           
        
        ne.run_sim()
        I_di = ne.synapses['sy'].I_di_vec[-1]
        n_fq_soen = I_di/(p['Phi0__pH_ns']/L_di)
        n_fq_soen_array[ii].append(n_fq_soen)
        
        # actual_data__drive = np.vstack((ne.time_vec[:],1e-6*ne.synapses['synapse_under_test'].I_spd2_vec[:])) 
        # error__drive = chi_squared_error(target_data__drive,actual_data__drive)
                                        
        # actual_data = np.vstack((ne.time_vec[:],1e-6*ne.synapses['synapse_under_test'].I_di_vec[:]))    
        # error__signal = chi_squared_error(target_data,actual_data)
        
        # plot_wr_comparison__synapse(file_name,spike_times,target_data__drive,actual_data__drive,target_data,actual_data,file_name,error__drive,error__signal)
            
plot_wr_comparison__synapse__n_fq_vs_I_de(M_vec,I_de_array,n_fq_1_array,n_fq_soen_array,num_jjs)
