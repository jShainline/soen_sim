#%%
import numpy as np
from matplotlib import pyplot as plt
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _functions import save_session_data, read_wr_data
from _plotting import plot_spd_response, plot_spd_response__waterfall

plt.close('all')

#%%
num_jjs = 3

#%% load wr data
if num_jjs == 1:
    dI = 0.25
    I_sy_list = np.concatenate([np.array([20.01,20.1]),np.arange(20.25,40+dI,dI)])
elif num_jjs == 2:
    dI = 0.25
    I_sy_list = np.concatenate([np.array([22.2]),np.arange(22.25,40+dI,dI)])
    I_sc = 35
elif num_jjs == 3:    
    dI = 0.25
    I_sy_list = np.arange(22,40+dI,dI)
    
directory = 'wrspice_data/fitting_data/{:1d}jj'.format(num_jjs)

time_pulse = 5e-9
time_sim = 300e-9
I_spd_array = []
for ii in range(len(I_sy_list)):
    
    print('ii = {} of {}'.format(ii+1,len(I_sy_list)))
    
    if num_jjs == 1:
        file_name = 'syn__1jj__single_spd_pulse__Isy{:05.2f}uA.dat'.format(I_sy_list[ii])
    elif num_jjs == 2:
        file_name = 'syn__2jj__single_spd_pulse__Isy{:05.2f}uA_Isc{:05.2f}uA_dt10.0ps.dat'.format(I_sy_list[ii],I_sc)
    elif num_jjs == 3:
        file_name = 'syn__3jj__single_spd_pulse__Isy{:05.2f}uA.dat'.format(I_sy_list[ii])
            
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


#%% reduce temporal resolution

print('\nreducing temporal resolution ...')
dt_vec = 1e-9*np.concatenate((np.linspace(0.01,0.1,10),np.linspace(0.2,1,9),np.linspace(2,10,9))) # np.linspace(0.01,0.1,10) # np.concatenate((np.linspace(0.01,0.1,10),np.linspace(0.2,1,9)))
for ii in range(len(dt_vec)):
    
    dt = dt_vec[ii] # desired temporal resolution
    
    time_vec_reduced = np.arange(time_vec[0],time_vec[-1]+dt*1e6,dt*1e6)
    nt_spd = len(time_vec_reduced)
    I_spd_array_reduced = np.zeros([len(I_sy_list),nt_spd])
    
    # I_spd_array_reduced = []
    for jj in range(nt_spd):
    
        print('jj = {} of {}'.format(jj+1,nt_spd))
        
        ti = (np.abs(np.asarray(time_vec[:]) - time_vec_reduced[jj])).argmin()
            
        for ii in range(len(I_sy_list)):     
            
            I_spd_array_reduced[ii][jj] = I_spd_array[ii][ti] # spd current vector
                
    save_string = 'master_syn_spd_response_{:d}jj_dt{:04.0f}ps'.format(num_jjs,dt*1e12)
    data_array = dict()
    data_array['spd_response_array'] = I_spd_array_reduced
    data_array['I_sy_list'] = I_sy_list
    data_array['time_vec'] = time_vec_reduced
    # save_session_data(data_array,save_string,True)
    save_session_data(data_array,save_string+'.soen',False)

print('done reducing temporal resolution.')

#%% plot 

num_jjs = 3
dt = 100e-12
file_name = 'master_syn_spd_response_{:d}jj_dt{:04.0f}ps.soen'.format(num_jjs,dt*1e12)
# plot_spd_response__waterfall(file_name = file_name, I_drive_reduction_factor = 1)
plot_spd_response(file_name = file_name, I_sy_reduction_factor = 3)
    
#%% load test
if 1 == 2:
    with open('../_circuit_data/master__syn__spd_response_dt0100ns.soen', 'rb') as data_file:         
        data_array__imported = pickle.load(data_file)
    spd_response_array__imprt = data_array__imported['spd_response_array']
    I_sy_list__imprt = data_array__imported['I_sy_list']
    time_vec__imprt = data_array__imported['time_vec']

