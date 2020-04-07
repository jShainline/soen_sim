#%%
import numpy as np
from matplotlib import pyplot as plt

from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_dendritic_integration_loop_current, plot_wr_data
from _functions import dendrite_current_splitting, Ljj, amp_fitter, mu_fitter_3_4, read_wr_data

plt.close('all')

#%% no leak, vary L_di


#%% temporal
tf = 100e-9
dt = 0.1e-9

I_b1 = 72e-6
tau_di = 1000000e-9


#%% get WRSpice data
plot_save_string = 'dend__cnst_drv__Idrv20uA_Ldi7.75nH_taudi7.75ms_tsim200ns' 
file_name = plot_save_string+'.dat'
data_dict = read_wr_data('wrspice_data/constant_drive/'+file_name)
data_to_plot = ['L9#branch']
plot_wr_data(data_dict,data_to_plot,plot_save_string)

# #%% run it
# error_mat = np.zeros([len(mu3_vec),len(mu4_vec)])
# for ii in range(len(mu3_vec)):
#     for jj in range(len(mu4_vec)):
#         print('ii = {:d} of {:d}, jj = {:d} of {:d}'.format(ii+1,len(mu3_vec),jj+1,len(mu4_vec)))
    
#         input_2 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
#                                time_vec = np.arange(0,tf+dt,dt), piecewise_linear = [[0e-9,0e-6],[2e-9,0e-6],[50e-9,30e-6]])  
                    
#         # initialize dendrite
#         dendrite_1 = dendrite('intermediate_dendrite', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
#                               input_synaptic_connections = [], input_synaptic_inductances = [[]], 
#                               input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
#                               input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
#                               thresholding_junction_critical_current = 40e-6, bias_currents = [I_b1,29e-6,35e-6],
#                               integration_loop_self_inductance = 50e-9, integration_loop_output_inductance = 0e-12,
#                               integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
#                               integration_loop_saturation_current = 17.65e-6)
                       
#         # propagate in time                         
#         sim_params = dict()
#         sim_params['dt'] = dt
#         sim_params['tf'] = tf
#         dendrite_1.sim_params = sim_params
#         dendrite_1.run_sim()
#         error_mat[ii,jj] = mu_fitter_3_4(data_dict,input_2.time_vec,dendrite_1.I_di,mu3_vec[ii],mu4_vec[jj])
#         # plot_dendritic_integration_loop_current(dendrite_1)

# #%%
# ind_best = np.where(error_mat == np.amin(error_mat))
# mu3_best = mu3_vec[ind_best[0]]
# mu4_best = mu4_vec[ind_best[1]]
# print('mu3_best = {}'.format(mu3_best))
# print('mu4_best = {}'.format(mu4_best))

# fig, ax = plt.subplots(1,1)
# error = ax.imshow(error_mat[:,:], cmap = plt.cm.viridis, interpolation='none', extent=[mu3_vec[0],mu3_vec[-1],mu4_vec[0],mu4_vec[-1]], aspect = 'auto')
# cbar = fig.colorbar(error, extend='both')
# cbar.minorticks_on()
 
# fig.suptitle('Error versus mu3 and mu4, mu1 = {}, mu2 = {}, amp = {}'.format(mu1,mu2,amp))
# plt.title('mu3_best = {}, mu4_best = {}'.format(mu3_best,mu4_best))
# save_str = 'error__mu3_mu4'.format()
# fig.savefig('figures/'+save_str+'.png')  

# ax.set_xlabel(r'mu3')
# ax.set_ylabel(r'mu4') 

# fig, ax = plt.subplots(1,1)
# error = ax.imshow(np.log10(error_mat[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[mu3_vec[0],mu3_vec[-1],mu4_vec[0],mu4_vec[-1]], aspect = 'auto')
# cbar = fig.colorbar(error, extend='both')
# cbar.minorticks_on()
 
# fig.suptitle('log10(Error) versus mu3 and mu4, mu1 = {}, mu2 = {}, amp = {}'.format(mu1,mu2,amp))
# plt.title('mu3_best = {}, mu4_best = {}'.format(mu3_best,mu4_best))
# save_str = 'error__mu3_mu4'.format()
# fig.savefig('figures/'+save_str+'.png')  

# ax.set_xlabel(r'mu3')
# ax.set_ylabel(r'mu4') 

# #%% run it again for best mu parameters

# input_2 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
#                                time_vec = np.arange(0,tf+dt,dt), piecewise_linear = [[0e-9,0e-6],[2e-9,0e-6],[50e-9,30e-6]])  
                    
# # initialize dendrite
# dendrite_1 = dendrite('intermediate_dendrite', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
#                       input_synaptic_connections = [], input_synaptic_inductances = [[]], 
#                       input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
#                       input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
#                       thresholding_junction_critical_current = 40e-6, bias_currents = [I_b1,29e-6,35e-6],
#                       integration_loop_self_inductance = 50e-9, integration_loop_output_inductance = 0e-12,
#                       integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
#                       integration_loop_saturation_current = 17.65e-6)
               
# # propagate in time                         
# sim_params = dict()
# sim_params['dt'] = dt
# sim_params['tf'] = tf
# sim_params['mu1'] = mu1
# sim_params['mu2'] = mu2
# sim_params['mu3'] = mu3_best
# sim_params['mu4'] = mu4_best
# sim_params['A'] = amp
# dendrite_1.sim_params = sim_params
# dendrite_1.run_sim()

# #%% plot the best fit        
# time_vec_spice = data_dict['time']
# target_vec = data_dict['L9#branch']

# fig, ax = plt.subplots(nrows = 1, ncols = 1)   
# fig.suptitle('WR vs soen_sim')
# plt.title('amp = {}; mu1 = {}; mu2 = {}; mu3 = {}; mu4 = {}'.format(amp,mu1,mu2,mu3_best,mu4_best))

# ax.plot(time_vec_spice*1e9,target_vec*1e9,'-', label = 'WR')        
# ax.plot(dendrite_1.time_vec*1e9,dendrite_1.I_di*1e9,'o-', markersize = 2, label = 'soen_sim')    
# ax.legend()
# ax.set_xlabel(r'Time [ns]')
# ax.set_ylabel(r'$I_{di} [nA]$') 

