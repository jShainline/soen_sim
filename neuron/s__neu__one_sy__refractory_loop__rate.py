import numpy as np
from matplotlib import pyplot as plt

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_neuronal_response, plot_num_in_burst, plot_phase_portrait
from soen_sim import input_signal, synapse, neuron

plt.close('all')

#%% sim params
dt = 100e-12


# synapse
I_spd = 20e-6
input_rate = 10e6
time_first_spike = 5e-9
time_last_spike = 1700e-9
tf = 2e-6

num_L_si = 1
num_I_sy = 1
num_tau_si = 1
L_si_vec = np.linspace(400e-9,1e-6,num_L_si) # [7.75e-9,77.5e-9,775e-9,7.75e-6]
I_sy_vec = np.linspace(33e-6,38e-6,num_I_sy) # [28e-6,30e-6,32e-6,34e-6,36e-6]
tau_si_vec = np.linspace(250e-9,500e-9,num_tau_si)

# num_L_si = 1
# num_I_sy = 1
# num_tau_si = 1
# L_si_vec = np.array([77.5e-9])
# I_sy_vec = np.array([33e-6])
# tau_si_vec = np.array([250e-9])

sim_params = dict()
sim_params['dt'] = 100e-12
sim_params['tf'] = tf
sim_params['synapse_model'] = 'lookup_table'

#%% run loops
# num_in_burst__array = np.zeros([num_L_si,num_I_sy,num_tau_si])
for ii in range(num_L_si):
    for jj in range(num_I_sy):
        for kk in range(num_tau_si):
            
            print('ii = {:d} of {:d} (L_si); jj = {:d} of {:d} (I_sy); kk = {:d} of {:d} (tau_si)'.format(ii+1,num_L_si,jj+1,num_I_sy,kk+1,num_tau_si))
            
            # initialize input signal
            input_1 = input_signal('in', input_temporal_form = 'constant_rate', spike_times = [input_rate,time_first_spike,time_last_spike])
                
            # initialize synapse
            synapse_1 = synapse('sy', num_jjs = 3, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si_vec[kk], 
                                integration_loop_self_inductance = L_si_vec[ii], integration_loop_output_inductance = 400e-12, 
                                synaptic_bias_currents = [I_spd,I_sy_vec[jj],36e-6,35e-6],
                                input_signal_name = 'in', synapse_model_params = sim_params)
                    
            # neuron
            time_params = dict()
            time_params['dt'] = dt
            time_params['tf'] = tf
            neuron_1 = neuron('ne', num_jjs = 4,
                              circuit_inductances = [0e-12,0e-12,200e-12,77.5e-12],
                              input_synaptic_connections = ['sy'], 
                              input_synaptic_inductances = [[20e-12,1]],                     
                              thresholding_junction_critical_current = 40e-6,
                              bias_currents = [74e-6,36e-6,35e-6],
                              integration_loop_self_inductance = 0e-12, 
                              integration_loop_output_inductances = [[200e-12,1],[200e-12,1]], # first is to drive latching JJ, second is to drive refractory dendrite; both are of the form [L_self,k]
                              integration_loop_temporal_form = 'exponential',
                              integration_loop_time_constant = 5e-9,
                              refractory_temporal_form = 'exponential',
                              refractory_loop_circuit_inductances = [0e-12,20e-12,200e-12,77.5e-12],
                              refractory_time_constant = 50e-9,
                              refractory_thresholding_junction_critical_current = 40e-6,
                              refractory_loop_self_inductance = 500e-12,
                              refractory_loop_output_inductance = 100e-12,
                              refractory_bias_currents = [74e-6,36e-6,35e-6],
                              refractory_receiving_input_inductance = [20e-12,1],                       
                              time_params = time_params)
                          
            neuron_1.run_sim()
            plot_neuronal_response(neuron_1)    
            plot_phase_portrait(neuron_1)
            # num_in_burst__array[ii,jj,kk] = len(neuron_1.spike_times)

#%%
# plot_num_in_burst(I_sy_vec,L_si_vec,tau_si_vec,num_in_burst__array)

