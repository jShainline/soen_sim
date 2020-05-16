#%%
import numpy as np
from matplotlib import pyplot as plt

from _plotting import plot_synaptic_integration_loop_current, plot_synaptic_integration_loop_current__multiple_synapses, plot_Isi_vs_Isy
from soen_sim import input_signal, synapse

plt.close('all')

#%%
t0 = 5e-9
tf = 1e-6 # 100e-9
dt = 0.1e-9
num_jjs = 1

dt_isi = 50e-9
spike_times = np.arange(t0,tf+dt_isi,dt_isi) # [t0]


#%%  calculate I_si vs I_sy

dI_sy = 1e-6
I_sy_vec = np.arange(21e-6,39e-6+dI_sy,dI_sy)

synapse_list = []            
# create sim_params dictionary
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf


L_si = 77.5e-9
tau_si = 1e6 # 250e-9

num_files = len(I_sy_vec)
for ii in range(num_files): # range(1): # 
    
    print('\nvary Isy, ii = {} of {}\n'.format(ii+1,num_files))

    # initialize input signal
    input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)
        
    # initialize synapse    
    synapse_1 = synapse('sy', num_jjs = num_jjs, integration_loop_temporal_form = 'exponential', 
                        integration_loop_time_constant = tau_si, integration_loop_self_inductance = L_si, synaptic_bias_currents = [I_sy_vec[ii]], 
                        integration_loop_output_inductance = 0e-12, integration_loop_bias_current = 35e-6, input_signal_name = 'in', 
                        synapse_model_params = sim_params)
    
    synapse_1.run_sim() 
    synapse_list.append(synapse_1)

    plot_synaptic_integration_loop_current(synapse_1)    

#%%

plot_synaptic_integration_loop_current__multiple_synapses(synapse_list)
plot_Isi_vs_Isy(synapse_list)
