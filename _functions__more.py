#%%
import numpy as np
from matplotlib import pyplot as plt

from soen_sim import input_signal, synapse, dendrite, neuron
from _functions import save_session_data, read_wr_data, chi_squared_error
from _plotting import plot_dendritic_drive, plot_wr_comparison, plot_error_mat

#%%
def dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt,tf,drive_info,amp_vec,mu1_vec,mu2_vec,mu3_vec,mu4_vec,master_error_plot_name):
    
    best_params = dict()
    best_params['amp_mu12'] = []
    best_params['amp_mu34'] = []
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
    
    num_sims = len(data_file_list)
    num_amps = len(amp_vec)
    num_mu1 = len(mu1_vec)
    num_mu2 = len(mu2_vec)
    num_mu3 = len(mu3_vec)
    num_mu4 = len(mu4_vec)
    
    error_mat_master__mu1_mu2 = np.zeros([num_amps,num_mu1,num_mu2])
    error_mat_master__mu3_mu4 = np.zeros([num_amps,num_mu3,num_mu4])
    
    if drive_info['drive_type'] == 'piecewise_linear':
        pwl_drive = drive_info['pwl_drive']      
    
    for ii in range(num_sims):
        print('\n\ndata_file {} of {}\n'.format(ii+1,num_sims))
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
                         
                    print('aa = {} of {}, bb = {} of {}, cc = {} of {}'.format(aa+1,num_amps,bb+1,num_mu1,cc+1,num_mu2))
                    
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
        
        error_mat_master__mu1_mu2 += error_mat_1
        ind_best = np.where(error_mat_1 == np.amin(error_mat_1))#error_mat.argmin()
        amp_best_mu12 = amp_vec[ind_best[0]][0]
        mu1_best = mu1_vec[ind_best[1]][0]
        mu2_best = mu2_vec[ind_best[2]][0]
        print('\n\namp_best_mu12 = {}'.format(amp_best_mu12))
        print('mu1_best = {}'.format(mu1_best))
        print('mu2_best = {}\n\n'.format(mu2_best))
        best_params['amp_mu12'].append(amp_best_mu12)
        best_params['mu1'].append(mu1_best)
        best_params['mu2'].append(mu2_best)
        data_array['error_mat__amp_mu1_mu2'] = error_mat_1
        
        #plot errors
        title_string = '{}; amp_best_mu12 = {:2.2f}, mu1_best = {:1.4f}, mu2_best = {:1.4f}'.format(data_file_list[ii],amp_best_mu12,mu1_best,mu2_best)
        save_str = '{}__error__amp_mu1_mu2'.format(data_file_list[ii])
        for aa in range(len(amp_vec)):    
            plot_error_mat(error_mat_1[aa,:,:],mu1_vec,mu2_vec,'mu1','mu2','amp = {}'.format(amp_vec[aa]),title_string,save_str)
            
        #repeat best one and plot   
        input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                           time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)  
                    
        sim_params = dict()
        sim_params['amp'] = amp_best_mu12
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
        plot_string = '{}__amp_best_{:2.2f}_mu1_best_{:1.4f}_mu2_best_{:1.4f}'.format(data_file_list[ii],amp_best_mu12,mu1_best,mu2_best)
        plot_wr_comparison(target_data,actual_data,plot_string)
        
        #-----------------------------
        # find best amp_mu34, mu3, mu4
        #-----------------------------   
        error_mat_2 = np.zeros([num_amps,num_mu3,num_mu4])
        print('seeking amp, mu3, mu4 ...')
        for aa in range(len(amp_vec)):
            for bb in range(len(mu3_vec)):
                for cc in range(len(mu4_vec)):
                             
                    print('aa = {} of {}, bb = {} of {}, cc = {} of {}'.format(aa+1,num_amps,bb+1,num_mu3,cc+1,num_mu4))
                    
                    # initialize input signal
                    input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                            time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)  
                    
                    # create sim_params dictionary
                    sim_params = dict()
                    sim_params['amp'] = amp_vec[aa]
                    sim_params['mu1'] = mu1_best
                    sim_params['mu2'] = mu2_best
                    sim_params['mu3'] = mu3_vec[bb]
                    sim_params['mu4'] = mu4_vec[cc]
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
            
        error_mat_master__mu3_mu4 += error_mat_2
        ind_best = np.where(error_mat_2 == np.amin(error_mat_2))#error_mat.argmin()
        amp_best_mu34 = amp_vec[ind_best[0]][0]
        mu3_best = mu3_vec[ind_best[1]][0]
        mu4_best = mu4_vec[ind_best[2]][0]
        print('\n\namp_best_mu34 = {}'.format(amp_best_mu34))
        print('mu3_best = {}'.format(mu3_best))
        print('mu4_best = {}'.format(mu4_best))
        best_params['mu3'].append(mu3_best)
        best_params['mu4'].append(mu4_best)
        data_array['error_mat__mu3_mu4'] = error_mat_2
        
        #plot errors
        title_string = '{}; amp_best_mu12 = {:2.2f}, amp_best_mu34 = {:2.2f}, mu1_best = {:1.2f}, mu2_best = {:1.2f}, mu3_best = {:1.2f}, mu4_best = {:1.2f}'.format(data_file_list[ii],amp_best_mu12,amp_best_mu34,mu1_best,mu2_best,mu3_best,mu4_best)
        save_str = '{}__error__mu3_mu4'.format(data_file_list[ii])            
        for aa in range(len(amp_vec)):    
            plot_error_mat(error_mat_2[aa,:,:],mu3_vec,mu4_vec,'mu3','mu4','amp = {}'.format(amp_vec[aa]),title_string,save_str)
    
        #repeat best one and plot   
        input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                           time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)  
                    
        sim_params = dict()
        sim_params['amp'] = amp_best_mu34
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
        plot_string = '{}__amp_best_mu12_{:2.2f}_amp_best_mu34_{:2.2f}_mu1_best_{:1.2f}_mu2_best_{:1.2f}_mu3_best_{:1.2f}_mu4_best_{:1.2f}'.format(data_file_list[ii],amp_best_mu12,amp_best_mu34,mu1_best,mu2_best,mu3_best,mu4_best) 
        actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))
        plot_wr_comparison(target_data,actual_data,plot_string)            
    
    # save data
    save_string = 'wr_fits__no_leak__vary_L_di__finding_amp_mu1_mu2+mu3_mu4'
    data_array['wr_spice_data_file_list'] = data_file_list
    data_array['best_params'] = best_params
    data_array['error_mat_master__mu1_mu2'] = error_mat_master__mu1_mu2
    data_array['error_mat_master__mu3_mu4'] = error_mat_master__mu3_mu4
    print('\n\nsaving session data ...')
    save_session_data(data_array,save_string)
    
    #plot errors
    title_string = '{}; amp_best_mu12 = {:2.2f}, amp_best_mu34 = {:2.2f}, mu1_best = {:1.2f}, mu2_best = {:1.2f}, mu3_best = {:1.2f}, mu4_best = {:1.2f}'.format(master_error_plot_name,amp_best_mu12,amp_best_mu34,mu1_best,mu2_best,mu3_best,mu4_best)    
    for aa in range(len(amp_vec)):
        save_str_1 = '{}__master_error__mu1_mu2__amp_{:2.2f}'.format(master_error_plot_name,amp_vec[aa])
        save_str_2 = '{}__master_error__mu3_mu4__amp_{:2.2f}'.format(master_error_plot_name,amp_vec[aa])
        plot_error_mat(error_mat_master__mu1_mu2[aa,:,:],mu1_vec,mu2_vec,'mu1','mu2','amp = {}'.format(amp_vec[aa]),title_string,save_str_1)
        plot_error_mat(error_mat_master__mu3_mu4[aa,:,:],mu3_vec,mu4_vec,'mu3','mu4','amp = {}'.format(amp_vec[aa]),title_string,save_str_2)
        
    return best_params, error_mat_master__mu1_mu2, error_mat_master__mu3_mu4