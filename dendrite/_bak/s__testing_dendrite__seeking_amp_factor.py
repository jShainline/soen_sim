#%%
import numpy as np
from matplotlib import pyplot as plt

from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_dendritic_integration_loop_current
from _functions import dendrite_current_splitting, Ljj, amp_fitter

plt.close('all')

#%% temporal
tf = 50e-9
dt = 0.1e-9

# I_sy = 37e-6
I_b1 = 72e-6
# I_th_n = 35e-6
# tau_si = 1000e-9
tau_di = 1000e-9
# tau_ref = 25e-9
# jitter_params = [0,25e-9]#[gaussian center, gaussian deviation]

# num_spikes_out_mat = np.zeros([len(I_sy_vec),num_synapses_tot])

#%% run it
# amp_vec = np.logspace(0,3,100)
amp_vec = np.linspace(90,110,100)
error_vec = np.zeros([len(amp_vec),1]) 
for ii in range(len(amp_vec)):     
    
    print('ii = {:d} of {:d}'.format(ii,len(amp_vec)))
    
    input_2 = input_signal('input_dendritic_drive', input_temporal_form = 'analog_dendritic_drive', output_inductance = 200e-12, 
                           time_vec = np.arange(0,tf+dt,dt), amplitude = 20e-6, time_on = 2e-9)  
                
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
    sim_params['A'] = amp_vec[ii]
    dendrite_1.sim_params = sim_params
    dendrite_1.run_sim()
    error_vec[ii] = amp_fitter(input_2.time_vec,dendrite_1.I_di)
    # plot_dendritic_integration_loop_current(dendrite_1)
    dendrite_1.I_di[-1]

#%%
ind_best = error_vec.argmin()
A_best = amp_vec[ind_best]
print('A_best = {}'.format(A_best))

fig, ax = plt.subplots(nrows = 1, ncols = 1)   
fig.suptitle('Error versus Amplitude')
# plt.title(plot_save_string)

ax.plot(amp_vec,error_vec,'o-')        
ax.plot(A_best,error_vec[ind_best],'ro',label = 'best; amp = {}'.format(A_best))
ax.legend()
ax.set_xlabel(r'Amplitude Factor')
ax.set_ylabel(r'Error') 

#%%
# Ljj(40e-6,0e-6)

#     dendrite_current_splitting(Ic,   Iflux,Ib1,  Ib2,  Ib3,  M,                      Lm2,   Ldr1,  Ldr2,  L1,     L2,      L3)
# Idr = dendrite_current_splitting(40e-6,20e-6,72e-6,29e-6,35e-6,np.sqrt(200e-12*10e-12),10e-12,10e-12,26e-12,200e-12,77.5e-12,1e-6)
# print('Idr = {}uA'.format(Idr*1e6))