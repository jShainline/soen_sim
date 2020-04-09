#%%
import numpy as np
from matplotlib import pyplot as plt
import time

from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_dendritic_integration_loop_current, plot_wr_data, plot_wr_comparison, plot_dendritic_drive, plot_error_mat
from _functions import dendrite_current_splitting, Ljj, amp_fitter, mu_fitter_3_4, read_wr_data, chi_squared_error, save_session_data, load_session_data

plt.close('all')

#%% no leak, vary L_di

t_tot = time.time()

# data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi7.75nH_taudi7.75ms_tsim200ns']  
                  
data_file_list = ['dend__cnst_drv__Idrv20uA_Ldi7.75nH_taudi7.75ms_tsim200ns',                                    
                  'dend__cnst_drv__Idrv20uA_Ldi77.5nH_taudi77.5ms_tsim200ns',
                  'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi775ms_tsim200ns',
                  'dend__cnst_drv__Idrv20uA_Ldi7.75uH_taudi7.75s_tsim200ns']

                  # 'dend__cnst_drv__Idrv20uA_Ldi775nH_taudi775ms_tsim1000ns',
                  # 'dend__cnst_drv__Idrv20uA_Ldi7.75uH_taudi7.75s_tsim1000ns'

pwl_drive = [[0e-9,0e-6],[1.9e-9,0e-6],[2e-9,20e-6]]

L_di_vec = [7.75e-9,77.5e-9,775e-9,7.75e-6]
tau_di_vec = [7.75e-3,77.5e-3,775e-3,7.75]

num_sims = len(data_file_list)
error_vec = np.zeros([num_sims,1])
dt = 0.1e-9
tf = 200e-9
    
num_loop = 15
amp_vec = np.logspace(np.log10(7),np.log10(200),num_loop)#np.linspace(25,25,1)#np.linspace(25,25,1)#
mu1_vec = np.linspace(0.25,2.5,num_loop)
mu2_vec = np.linspace(0.25,2.5,num_loop)
mu3_vec = np.linspace(0.25,2.5,num_loop)
mu4_vec = np.linspace(0.25,2.5,num_loop) 
          
best_params = dict()
best_params['amp'] = []
best_params['mu1'] = []
best_params['mu2'] = []
best_params['mu3'] = []
best_params['mu4'] = []

data_array = dict()
data_array['amp_vec'] = amp_vec
data_array['mu1_vec'] = mu1_vec
data_array['mu2_vec'] = mu2_vec
data_array['mu3_vec'] = mu3_vec
data_array['mu4_vec'] = mu4_vec
    
for ii in range(len(data_file_list)):
    print('\n\ndata_file {} of {}\n'.format(ii+1,len(data_file_list)))
    plt.close('all')
    
    # WR data
    file_name = data_file_list[ii]+'.dat'
    data_dict = read_wr_data('wrspice_data/constant_drive/'+file_name)
    target_data = np.vstack((data_dict['time'],data_dict['L9#branch']))
         
    #------------------------
    # find best amp, mu1, mu2
    #------------------------
    mu3 = [1] #np.linspace([0.5,2.5,10])
    mu4 = [0.5] #np.linspace([0.25,1.5,10])
    
    error_mat_1 = np.zeros([len(amp_vec),len(mu1_vec),len(mu2_vec)])
    print('seeking amp, mu1, mu2 ...')
    for aa in range(len(amp_vec)):
        for bb in range(len(mu1_vec)):
            for cc in range(len(mu2_vec)):
                     
                print('aa = {} of {}, bb = {} of {}, cc = {} of {}'.format(aa+1,len(amp_vec),bb+1,len(mu1_vec),cc+1,len(mu2_vec)))
                
                # initialize input signal
                input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                        time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)  
                
                # create sim_params dictionary
                sim_params = dict()
                sim_params['amp'] = amp_vec[aa]
                sim_params['mu1'] = mu1_vec[bb]
                sim_params['mu2'] = mu2_vec[cc]
                sim_params['mu3'] = mu3
                sim_params['mu4'] = mu4
                sim_params['dt'] = dt
                sim_params['tf'] = tf
                               
                # initialize dendrite
                dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                                      input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                      input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                      input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                                      thresholding_junction_critical_current = 40e-6, bias_currents = [72e-6,29e-6,35e-6],
                                      integration_loop_self_inductance = L_di_vec[ii], integration_loop_output_inductance = 0e-12,
                                      integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di_vec[ii],
                                      integration_loop_saturation_current = 11.75e-6, dendrite_model_params = sim_params)
                                                                
                # Idr = dendrite_current_splitting(40e-6,0,72e-6,29e-6,35e-6,np.sqrt(200e-12*10e-12),10e-12,10e-12,26e-12,200e-12,77.5e-12,7.75e-9)
                    
                dendrite_1.run_sim()
            
                # plot_dendritic_drive(dendrite_1.time_vec, dendrite_1.dendritic_drive)
                
                actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))    
                error_mat_1[aa,bb,cc] = chi_squared_error(target_data,actual_data)
                
                # plot_wr_comparison(target_data,actual_data,'{}; amp = {}, mu1 = {}, mu2 = {}'.format(data_file_list[ii],amp_vec[aa],mu1_vec[bb],mu2_vec[cc]))
                
    ind_best = np.where(error_mat_1 == np.amin(error_mat_1))#error_mat.argmin()
    amp_best = amp_vec[ind_best[0]][0]
    mu1_best = mu1_vec[ind_best[1]][0]
    mu2_best = mu2_vec[ind_best[2]][0]
    print('\n\namp_best = {}'.format(amp_best))
    print('mu1_best = {}'.format(mu1_best))
    print('mu2_best = {}\n\n'.format(mu2_best))
    best_params['amp'].append(amp_best)
    best_params['mu1'].append(mu1_best)
    best_params['mu2'].append(mu2_best)
    data_array['error_mat__amp_mu1_mu2'] = error_mat_1
    
    #plot errors
    title_string = '{}; amp_best = {:2.2f}, mu1_best = {:1.4f}, mu2_best = {:1.4f}'.format(data_file_list[ii],amp_best,mu1_best,mu2_best)
    save_str = '{}__error__amp_mu1_mu2'.format(data_file_list[ii])
    for aa in range(len(amp_vec)):    
        plot_error_mat(error_mat_1[aa,:,:],mu1_vec,mu2_vec,'mu1','mu2','amp = {}'.format(amp_vec[aa]),title_string,save_str)
        
    #repeat best one and plot   
    input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                       time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)  
                
    sim_params = dict()
    sim_params['amp'] = amp_best
    sim_params['mu1'] = mu1_best
    sim_params['mu2'] = mu2_best
    sim_params['mu3'] = mu3
    sim_params['mu4'] = mu4
    sim_params['dt'] = dt
    sim_params['tf'] = tf
    
    dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                        input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                        input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                        input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                        thresholding_junction_critical_current = 40e-6, bias_currents = [72e-6,29e-6,35e-6],
                        integration_loop_self_inductance = L_di_vec[ii], integration_loop_output_inductance = 0e-12,
                        integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di_vec[ii],
                        integration_loop_saturation_current = 11.75e-6, dendrite_model_params = sim_params)
                                                                
    dendrite_1.run_sim()
    # plot_dendritic_integration_loop_current(dendrite_1)
    actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))
    plot_string = '{}__amp_best_{:2.2f}_mu1_best_{:1.4f}_mu2_best_{:1.4f}'.format(data_file_list[ii],amp_best,mu1_best,mu2_best)
    plot_wr_comparison(target_data,actual_data,plot_string)
    
    #------------------
    # find best mu3,mu4
    #------------------   
    error_mat_2 = np.zeros([len(mu3_vec),len(mu4_vec)])
    print('seeking mu3,mu4...')
    for aa in range(len(mu3_vec)):
        for bb in range(len(mu4_vec)):
                     
            print('aa = {} of {}, bb = {} of {}'.format(aa+1,len(mu3_vec),bb+1,len(mu4_vec)))
            
            # initialize input signal
            input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)  
            
            # create sim_params dictionary
            sim_params = dict()
            sim_params['amp'] = amp_best
            sim_params['mu1'] = mu1_best
            sim_params['mu2'] = mu2_best
            sim_params['mu3'] = mu3_vec[aa]
            sim_params['mu4'] = mu4_vec[bb]
            sim_params['dt'] = dt
            sim_params['tf'] = tf
                           
            # initialize dendrite
            dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                                  input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                  input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                  input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                                  thresholding_junction_critical_current = 40e-6, bias_currents = [72e-6,29e-6,35e-6],
                                  integration_loop_self_inductance = L_di_vec[ii], integration_loop_output_inductance = 0e-12,
                                  integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di_vec[ii],
                                  integration_loop_saturation_current = 11.75e-6, dendrite_model_params = sim_params)
                                                            
            # Idr = dendrite_current_splitting(40e-6,0,72e-6,29e-6,35e-6,np.sqrt(200e-12*10e-12),10e-12,10e-12,26e-12,200e-12,77.5e-12,7.75e-9)
                
            dendrite_1.run_sim()
        
            # plot_dendritic_drive(dendrite_1.time_vec, dendrite_1.dendritic_drive)
            
            actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))    
            error_mat_2[aa,bb] = chi_squared_error(target_data,actual_data)
            
            # plot_wr_comparison(target_data,actual_data,'{}; amp = {}, mu1 = {}, mu2 = {}'.format(data_file_list[ii],amp_vec[aa],mu1_vec[bb],mu2_vec[cc]))
                
    ind_best = np.where(error_mat_2 == np.amin(error_mat_2))#error_mat.argmin()
    mu3_best = mu3_vec[ind_best[0]][0]
    mu4_best = mu4_vec[ind_best[1]][0]
    print('\n\nmu3_best = {}'.format(mu3_best))
    print('mu4_best = {}'.format(mu4_best))
    best_params['mu3'].append(mu3_best)
    best_params['mu4'].append(mu4_best)
    data_array['error_mat__mu3_mu4'] = error_mat_2
    
    #plot errors
    title_string = '{}; amp_best = {:2.2f}, mu1_best = {:1.4f}, mu2_best = {:1.4f}, mu3_best = {:1.4f}, mu4_best = {:1.4f}'.format(data_file_list[ii],amp_best,mu1_best,mu2_best,mu3_best,mu4_best)
    save_str = '{}__error__mu3_mu4'.format(data_file_list[ii])
    plot_error_mat(error_mat_2,mu3_vec,mu4_vec,'mu3','mu4','amp, mu1, mu2 previously fit',title_string,save_str)  

    #repeat best one and plot   
    input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                       time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)  
                
    sim_params = dict()
    sim_params['amp'] = amp_best
    sim_params['mu1'] = mu1_best
    sim_params['mu2'] = mu2_best
    sim_params['mu3'] = mu3_best
    sim_params['mu4'] = mu4_best
    sim_params['dt'] = dt
    sim_params['tf'] = tf
    
    dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                        input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                        input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                        input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                        thresholding_junction_critical_current = 40e-6, bias_currents = [72e-6,29e-6,35e-6],
                        integration_loop_self_inductance = L_di_vec[ii], integration_loop_output_inductance = 0e-12,
                        integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di_vec[ii],
                        integration_loop_saturation_current = 11.75e-6, dendrite_model_params = sim_params)
                                                                
    dendrite_1.run_sim()
    # plot_dendritic_integration_loop_current(dendrite_1)
    plot_string = '{}__amp_best_{:f}_mu1_best_{:f}_mu2_best_{:f}_mu3_best_{:f}_mu4_best_{:f}'.format(data_file_list[ii],amp_best,mu1_best,mu2_best,mu3_best,mu4_best) 
    actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))
    plot_wr_comparison(target_data,actual_data,plot_string)            

# save data
save_string = 'wr_fits__no_leak__vary_L_di__finding_amp_mu1_mu2+mu3_mu4'
data_array['wr_spice_data_file_list'] = data_file_list
data_array['best_params'] = best_params
save_session_data(data_array,save_string)
   
elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+' s')

# load data
# load_string = 'session_data__wr_fits__no_leak__vary_L_di__finding_amp_mu1_mu2+mu3_mu4__2020-04-08_14-29-15.dat'
# data_array_imported = load_session_data(load_string)