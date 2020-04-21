#%%
import numpy as np
from matplotlib import pyplot as plt

from soen_sim import input_signal, synapse, dendrite, neuron
from _functions import save_session_data, read_wr_data, chi_squared_error, dendritic_drive__piecewise_linear, dendritic_drive__square_pulse_train, dendritic_drive__exp_pls_train__LR
from _plotting import plot_wr_comparison, plot_error_mat, plot_wr_comparison__drive_and_response

#%%
def dendrite_model__parameter_sweep(data_file_list,L_di_vec,tau_di_vec,dt_vec,tf_vec,drive_info,mu1_vec,mu2_vec,master_error_plot_name):
    
    master_error_plot_name = 'mstr_err__'+master_error_plot_name
    
    best_params = dict()
    best_params['mu1'] = []
    best_params['mu2'] = []
    
    data_array = dict()
    data_array['mu1_vec'] = mu1_vec
    data_array['mu2_vec'] = mu2_vec
    
    num_sims = len(data_file_list)
    num_mu1 = len(mu1_vec)
    num_mu2 = len(mu2_vec)
        
    print('\n\nrunning dendrite_model__parameter_sweep\n\nnum_files = {:d}\nnum_mu1 = {:d}\nnum_mu2 = {:d}'.format(num_sims,num_mu1,num_mu2))
    
    error_mat_master__mu1_mu2 = np.zeros([num_mu1,num_mu2])
    
    if drive_info['drive_type'] == 'piecewise_linear':
        pwl_drive = drive_info['pwl_drive']
        directory_string = 'constant_drive'
        wr_drive_string = '@I0[c]'
        wr_target_string = 'L9#branch'
    if drive_info['drive_type'] == 'linear_ramp':
        pwl_drive = drive_info['pwl_drive']
        directory_string = 'linear_ramp'
        wr_drive_string = '@I0[c]'
        wr_target_string = 'L9#branch'
    if drive_info['drive_type'] == 'sq_pls_trn':
        sq_pls_trn_params = drive_info['sq_pls_trn_params']
        directory_string = 'square_pulse_sequence'
        wr_drive_string = '@I0[c]'
        wr_target_string = 'L9#branch'
    if drive_info['drive_type'] == 'exp_pls_trn':
        exp_pls_trn_params = drive_info['exp_pls_trn_params']
        directory_string = 'exponential_pulse_sequence'
        wr_drive_string = 'L5#branch'
        wr_target_string = 'L10#branch'
    
    for ii in range(num_sims):
        print('\ndata_file {} of {}\n'.format(ii+1,num_sims))
        plt.close('all')
        
        dt = dt_vec[ii]
        tf = tf_vec[ii]
        
        # WR data
        print('reading wr data ...\n')
        file_name = data_file_list[ii]+'.dat'
        data_dict = read_wr_data('wrspice_data/'+directory_string+'/'+file_name)
        target_data = np.vstack((data_dict['time'],data_dict[wr_target_string]))
        
        #----------------------
        # compare drive signals
        #----------------------
        target_data__drive = np.vstack((data_dict['time'],data_dict[wr_drive_string]))
        
        # initialize input signal
        if drive_info['drive_type'] == 'piecewise_linear' or drive_info['drive_type'] == 'linear_ramp':
            input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)
            dendritic_drive = dendritic_drive__piecewise_linear(input_1.time_vec,pwl_drive)
        if drive_info['drive_type'] == 'sq_pls_trn':
            input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), square_pulse_train = sq_pls_trn_params)
            dendritic_drive = dendritic_drive__square_pulse_train(input_1.time_vec,sq_pls_trn_params)
        if drive_info['drive_type'] == 'exp_pls_trn':
            input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                    time_vec = np.arange(0,tf+dt,dt), exponential_pulse_train = exp_pls_trn_params)
            dendritic_drive = dendritic_drive__exp_pls_train__LR(input_1.time_vec,exp_pls_trn_params)
        
        actual_data__drive = np.vstack((input_1.time_vec[:],dendritic_drive[:,0])) 
        print('comparing drive signals ...\n')
        error__drive = chi_squared_error(target_data__drive,actual_data__drive)
             
        #------------------------
        # find best amp, mu1, mu2
        #------------------------
        
        error_mat_1 = np.zeros([num_mu1,num_mu2])        
        print('seeking mu1, mu2 ...')
        for aa in range(num_mu1):
            for bb in range(num_mu2):
                     
                print('aa = {} of {}, bb = {} of {}'.format(aa+1,num_mu1,bb+1,num_mu2))
                                    
                # create sim_params dictionary
                sim_params = dict()
                sim_params['mu1'] = mu1_vec[aa]
                sim_params['mu2'] = mu2_vec[bb]
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
                                                                                        
                dendrite_1.run_sim()
                                
                actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))    
                error_mat_1[aa,bb] = chi_squared_error(target_data,actual_data)
        
        error_mat_master__mu1_mu2 += error_mat_1
        ind_best = np.where(error_mat_1 == np.amin(error_mat_1))#error_mat.argmin()
        mu1_best = mu1_vec[ind_best[0]][0]
        mu2_best = mu2_vec[ind_best[1]][0]
        print('mu1_best = {}'.format(mu1_best))
        print('mu2_best = {}\n\n'.format(mu2_best))
        best_params['mu1'].append(mu1_best)
        best_params['mu2'].append(mu2_best)
        data_array['error_mat__amp_mu1_mu2'] = error_mat_1
        
        #plot errors
        title_string = '{}\nmu1_best = {:1.2f}, mu2_best = {:1.2f}'.format(data_file_list[ii],mu1_best,mu2_best)
        save_str = '{}__error__mu1_mu2'.format(data_file_list[ii])    
        # plot_error_mat(error_mat_1[:,:],mu1_vec,mu2_vec,'mu1','mu2',title_string,save_str)
            
        # #repeat best one and plot   
        # if drive_info['drive_type'] == 'piecewise_linear' or drive_info['drive_type'] == 'linear_ramp':
        #     input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
        #                             time_vec = np.arange(0,tf+dt,dt), piecewise_linear = pwl_drive)
        #     dendritic_drive = dendritic_drive__piecewise_linear(input_1.time_vec,pwl_drive)
        # if drive_info['drive_type'] == 'sq_pls_trn':
        #     input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
        #                             time_vec = np.arange(0,tf+dt,dt), square_pulse_train = sq_pls_trn_params)
        #     dendritic_drive = dendritic_drive__square_pulse_train(input_1.time_vec,sq_pls_trn_params)
        # if drive_info['drive_type'] == 'exp_pls_trn':
        #     input_1 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
        #                             time_vec = np.arange(0,tf+dt,dt), exponential_pulse_train = exp_pls_trn_params)
        #     dendritic_drive = dendritic_drive__exp_pls_train__LR(input_1.time_vec,exp_pls_trn_params) 
                    
        # sim_params = dict()
        # sim_params['mu1'] = mu1_best
        # sim_params['mu2'] = mu2_best
        # sim_params['dt'] = dt
        # sim_params['tf'] = tf
        
        # dendrite_1 = dendrite('dendrite_under_test', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
        #                     input_synaptic_connections = [], input_synaptic_inductances = [[]], 
        #                     input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
        #                     input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
        #                     thresholding_junction_critical_current = 40e-6, bias_currents = [72e-6,29e-6,35e-6],
        #                     integration_loop_self_inductance = L_di_vec[ii], integration_loop_output_inductance = 0e-12,
        #                     integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di_vec[ii],
        #                     integration_loop_saturation_current = 11.75e-6, dendrite_model_params = sim_params)
                                                                    
        # dendrite_1.run_sim()
        # # plot_dendritic_integration_loop_current(dendrite_1)
        # actual_data = np.vstack((input_1.time_vec[:],dendrite_1.I_di[:,0]))
        main_title = '{}\nmu1_best_{:1.4f}_mu2_best_{:1.4f}'.format(data_file_list[ii],mu1_best,mu2_best)
        error__signal =  np.amin(error_mat_1)
        plot_wr_comparison__drive_and_response(main_title,target_data__drive,actual_data__drive,target_data,actual_data,data_file_list[ii],error__drive,error__signal)
    
    # # save data
    # save_string = 'wr_fits__finding_mu1_mu2'
    # data_array['wr_spice_data_file_list'] = data_file_list
    # data_array['best_params'] = best_params
    # data_array['error_mat_master__mu1_mu2'] = error_mat_master__mu1_mu2
    # print('\n\nsaving session data ...')
    # save_session_data(data_array,save_string)
    
    # #plot errors
    # title_string = '{}; mu1_best = {:1.2f}, mu2_best = {:1.2f}'.format(master_error_plot_name,mu1_best,mu2_best)    
    # save_str_1 = '{}__master_error__mu1_mu2'.format(master_error_plot_name)
    # plot_error_mat(error_mat_master__mu1_mu2[:,:],mu1_vec,mu2_vec,'mu1','mu2',title_string,save_str_1)
        
    return best_params, error_mat_master__mu1_mu2