#%%
import numpy as np
from matplotlib import pyplot as plt
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3
from util import physical_constants
p = physical_constants()

plt.close('all')

#%% load wr data
I_sy_vec = np.arange(22,40,1)
directory = 'wrspice_data/fitting_data'
# file_name = 'syn__single_spd_pulse__no_jj_biases.dat'

time_pulse = 5e-9
time_sim = 300e-9
I_spd_array = []
for ii in range(len(I_sy_vec)):
    
    print('ii = {} of {}'.format(ii+1,len(I_sy_vec)))
    
    file_name = 'syn__single_spd_pulse__Isy{:05.2f}uA.dat'.format(I_sy_vec[ii])
    data_dict = read_wr_data(directory+'/'+file_name)

    # construct spd response data structure
    
    time_vec = data_dict['time']    
    initial_ind = (np.abs(time_vec-time_pulse)).argmin()
    final_ind = (np.abs(time_vec-time_sim)).argmin()
    time_vec = time_vec[initial_ind:final_ind]
    time_vec = [(x-time_vec[0])*1e6 for x in time_vec]
    
    I_spd = data_dict['L0#branch']
    I_spd = I_spd[initial_ind:final_ind]
    I_spd = [x*1e6 for x in I_spd]
    I_spd_array.append(I_spd)
    
#%% 
fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
fig.suptitle('spd response') 
for ii in range(len(I_sy_vec)):  
    ax.plot(time_vec,I_spd_array[ii], label = 'I_sy = {}uA'.format(I_sy_vec[ii]))    
ax.set_xlabel(r'Time [$\mu$s]')
ax.set_ylabel(r'$I_{spd}$ [$\mu$A]')
ax.legend()
plt.show()

#%% save data
I_spd_mat = np.zeros([len(I_sy_vec),len(time_vec)])
for ii in range(len(I_sy_vec)):
    I_spd_mat[ii,:] = I_spd_array[ii][:]
    
save_string = 'master__syn__spd_response'
data_array = dict()
data_array['master_spd_response_matrix'] = I_spd_mat
data_array['I_sy_vec'] = I_sy_vec
data_array['time_vec'] = time_vec
print('\n\nsaving session data ...')
save_session_data(data_array,save_string)

#%% load test
with open('../master__syn__spd_response.soen', 'rb') as data_file:         
    data_array__imported = pickle.load(data_file)
spd_response_matrix__imported = data_array__imported['master_spd_response_matrix']
I_sy_vec__imported = data_array__imported['I_sy_vec']
time_vec__imported = data_array__imported['time_vec']

dt = 1e-3 # units of us
tf = time_vec__imported[-1]
nt_spd = np.floor(tf/dt).astype(int)
spd_t = np.zeros([nt_spd])
spd_i = np.zeros([nt_spd])

I_sy = 29.2
I_sy_ind = (np.abs(I_sy_vec__imported[:] - I_sy)).argmin()
for ii in range(nt_spd):
    ti = (np.abs(np.asarray(time_vec__imported[:]) - ii*dt)).argmin()
    spd_t[ii] = time_vec__imported[ti] # spd time vector
    spd_i[ii] = spd_response_matrix__imported[I_sy_ind,ti] # spd current vector
            
# fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
# fig.suptitle('master spd response')   
# ax.plot(spd_t,spd_i)      
# ax.set_xlabel(r'Time [$\mu$s]')
# ax.set_ylabel(r'$I_{spd}$ [$\mu$A]')
