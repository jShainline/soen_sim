#%%
import numpy as np
from matplotlib import pyplot as plt

from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_dendritic_integration_loop_current
from _functions import dendrite_current_splitting, Ljj, amp_fitter, mu_fitter, read_wr_data

plt.close('all')

#%% temporal
tf = 50e-9
dt = 0.1e-9

I_b1 = 72e-6
tau_di = 1000e-9

# A_prefactor = 101.919192
amp_vec = np.linspace(7,13,30)
mu1_vec = np.linspace(1.9,2.3,30)
mu2_vec = np.linspace(0.35,0.55,30)

#%% get WRSpice data
file_name = '_dat__sweep_Iflux__01'
data_dict = read_wr_data('dendrite_testing/'+file_name)

#%% run it
error_mat = np.zeros([len(amp_vec),len(mu1_vec),len(mu2_vec)])
for kk in range(len(amp_vec)):
    for ii in range(len(mu1_vec)):
        for jj in range(len(mu2_vec)):
            print('kk = {:d} of {:d}, ii = {:d} of {:d}, jj = {:d} of {:d}'.format(kk+1,len(amp_vec),ii+1,len(mu1_vec),jj+1,len(mu2_vec)))
        
            input_2 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                                   time_vec = np.arange(0,tf+dt,dt), slope = 30e-6/48e-9, time_on = 2e-9)  
                        
            # initialize dendrite
            dendrite_1 = dendrite('intermediate_dendrite', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                                  input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                                  input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                                  input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                                  thresholding_junction_critical_current = 40e-6, bias_currents = [I_b1,29e-6,35e-6],
                                  integration_loop_self_inductance = 10e-6, integration_loop_output_inductance = 0e-12,
                                  integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
                                  integration_loop_saturation_current = 13e-6)
                           
            # propagate in time                         
            sim_params = dict()
            sim_params['dt'] = dt
            sim_params['tf'] = tf
            sim_params['mu1'] = mu1_vec[ii]
            sim_params['mu2'] = mu2_vec[jj]
            sim_params['A'] = amp_vec[kk]
            dendrite_1.sim_params = sim_params
            dendrite_1.run_sim()
            error_mat[kk,ii,jj] = mu_fitter(data_dict,input_2.time_vec,dendrite_1.I_di,mu1_vec[ii],mu2_vec[jj],amp_vec[kk])
            # plot_dendritic_integration_loop_current(dendrite_1)
            # dendrite_1.I_di[-1]

#%%
ind_best = np.where(error_mat == np.amin(error_mat))#error_mat.argmin()
amp_best = amp_vec[ind_best[0]]
mu1_best = mu1_vec[ind_best[1]]
mu2_best = mu2_vec[ind_best[2]]
print('amp_best = {}'.format(amp_best))
print('mu1_best = {}'.format(mu1_best))
print('mu2_best = {}'.format(mu2_best))

for ii in range(len(amp_vec)):
    fig, ax = plt.subplots(1,1)
    error = ax.imshow(error_mat[ii,:,:], cmap = plt.cm.viridis, interpolation='none', extent=[mu1_vec[0],mu1_vec[-1],mu2_vec[0],mu2_vec[-1]], aspect = 'auto')
    cbar = fig.colorbar(error, extend='both')
    cbar.minorticks_on()
     
    fig.suptitle('Error versus mu1 and mu2, amp = {}'.format(amp_vec[ii]))
    plt.title('amp_best = {}, mu1_best = {}, mu2_best = {}'.format(amp_best,mu1_best,mu2_best))
    save_str = 'error_amp={:2.4f}'.format(amp_vec[ii])
    fig.savefig('figures/'+save_str+'.png')  
    
    ax.set_xlabel(r'mu1')
    ax.set_ylabel(r'mu2') 

#%% run it again for best mu parameters

input_2 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                       time_vec = np.arange(0,tf+dt,dt), slope = 30e-6/48e-9, time_on = 2e-9)  

dendrite_1 = dendrite('intermediate_dendrite', inhibitory_or_excitatory = 'excitatory', circuit_inductances = [10e-12,26e-12,200e-12,77.5e-12], 
                      input_synaptic_connections = [], input_synaptic_inductances = [[]], 
                      input_dendritic_connections = [], input_dendritic_inductances = [[]],                      
                      input_direct_connections = ['input_dendritic_drive'], input_direct_inductances = [[10e-12,1]],
                      thresholding_junction_critical_current = 40e-6, bias_currents = [I_b1,29e-6,35e-6],
                      integration_loop_self_inductance = 10e-6, integration_loop_output_inductance = 0e-12,
                      integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_di,
                      integration_loop_saturation_current = 13e-6)
               
# propagate in time                         
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf
sim_params['mu1'] = mu1_best
sim_params['mu2'] = mu2_best
sim_params['A'] = amp_best
dendrite_1.sim_params = sim_params
dendrite_1.run_sim()

#%% plot the best fit        
time_vec_spice = data_dict['time']
target_vec = data_dict['L9#branch']

fig, ax = plt.subplots(nrows = 1, ncols = 1)   
fig.suptitle('WR vs soen_sim')
plt.title('amp = {}; mu1 = {}; mu2 = {}'.format(amp_best,mu1_best,mu2_best))

ax.plot(time_vec_spice*1e9,target_vec*1e9,'-', label = 'WR')        
ax.plot(dendrite_1.time_vec*1e9,dendrite_1.I_di*1e9,'o-', markersize = 2, label = 'soen_sim')    
ax.legend()
ax.set_xlabel(r'Time [ns]')
ax.set_ylabel(r'$I_{di} [nA]$') 
