#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import pickle
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import numpy.matlib

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_syn_rate_array
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3
from util import physical_constants
p = physical_constants()

plt.close('all')

#%% load wr data, find peaks, find rates

I_sy = 40#uA

dI = 0.25
I_drive_list = np.concatenate([np.array([0.01,0.1]),np.arange(0.25,20.0+dI,dI)])

#%%
    
j_sf_ifi_array = [] # array of inter-fluxon intervals at the synaptic firing junction
j_sf_rate_array = [] # array of fluxon production rate at the synaptic firing junction
j_jtl_ifi_array = [] # array of inter-fluxon intervals at the jtl junction
j_jtl_rate_array = [] # array of fluxon production rate at the jtl junction
j_si_ifi_array = [] # array of inter-fluxon intervals at the synaptic integration loop junction
j_si_rate_array = [] # array of fluxon production rate at the synaptic integration loop junction

I_si_array = [] # array of values of I_si

num_drives = len(I_drive_list) 
directory = 'wrspice_data/fitting_data/1jj'
for ii in range(num_drives):
    
    print('ii = {:d} of {:d}'.format(ii+1,num_drives))        
    
    file_name = 'syn_1jj_cnst_drv_Isy{:5.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0077.5ms_dt00.01ps.dat'.format(I_sy,I_drive_list[ii])
    data_dict = read_wr_data(directory+'/'+file_name)
    
    # find peaks for each jj
    time_vec = data_dict['time']
    j_si = data_dict['v(3)']
    
    # initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
    # final_ind = (np.abs(time_vec-t_sim_list[ii]*1e-9)).argmin()

    # time_vec = time_vec[initial_ind:final_ind]
    # j_sf = j_sf[initial_ind:final_ind]
    # j_jtl = j_jtl[initial_ind:final_ind]
    # j_si = j_si[initial_ind:final_ind]
  
    j_si_peaks, _ = find_peaks(j_si, height = 200e-6)
   
    # find inter-fluxon intervals and fluxon generation rates for each JJ
    I_si = data_dict['L1#branch']
    # I_si = I_si[initial_ind:final_ind]
    I_drive = data_dict['L0#branch']
    # I_drive = I_drive[initial_ind:final_ind]            
    
    j_si_ifi = np.diff(time_vec[j_si_peaks])
    
    j_si_rate = 1/j_si_ifi

    j_si_ifi_array.append(j_si_ifi)
    j_si_rate_array.append(j_si_rate)
    
    I_si_array.append(I_si[j_si_peaks])
    
    
#%% assemble data and change units
I_si_pad = 100e-9 # amount above the observed max of Isi that the simulation will allow before giving a zero rate

master_rate_array = []
I_si_array__scaled = []
    
for ii in range(num_drives):
                
    temp_rate_vec = np.zeros([len(j_si_rate_array[ii])])
    temp_I_si_vec = np.zeros([len(j_si_rate_array[ii])])
    for jj in range(len(j_si_rate_array[ii])):
        temp_rate_vec[jj] = 1e-6*j_si_rate_array[ii][jj] # master_rate_array has units of fluxons per microsecond
        temp_I_si_vec[jj] = 1e6*( I_si_array[ii][jj]+I_si_array[ii][jj+1] )/2 # this makes I_si_array__scaled have the same dimensions as j_si_rate_array and units of uA

    master_rate_array.append([])
    master_rate_array[ii] = np.append(temp_rate_vec,0) # this zero is added so that a current that rounds to I_si + I_si_pad will give zero rate
    I_si_array__scaled.append([])
    I_si_array__scaled[ii] = np.append(temp_I_si_vec,np.max(temp_I_si_vec)+I_si_pad) # this additional I_si + I_si_pad is included so that a current that rounds to I_si + I_si_pad will give zero rate


#%% plot the rate vectors 

# plot_syn_rate_array(I_si_array = I_si_array__scaled, master_rate_array = master_rate_array, I_drive_list = I_drive_list)

file_name = 'master__syn__rate_array__{:1d}jj__Isipad{:04.0f}nA.soen'.format(1,I_si_pad*1e9)
plot_syn_rate_array(file_name = file_name, I_drive_reduction_factor = 2)
    
#%% save the data    
 
save_string = 'master__syn__1jj__rate_array__Isipad{:04.0f}nA'.format(I_si_pad*1e9)
data_array = dict()
data_array['rate_array'] = master_rate_array
data_array['I_drive_list'] = I_drive_list#[x*1e-6 for x in I_drive_vec]
data_array['I_si_array'] = I_si_array__scaled
print('\n\nsaving session data ...')
save_session_data(data_array,save_string)
save_session_data(data_array,save_string+'.soen',False)

#%% load test

if 1 == 2:
    
    with open('../_circuit_data/master__syn__rate_array__Isipad0100nA.soen', 'rb') as data_file:         
            data_array_imprt = pickle.load(data_file)
    # data_array_imported = load_session_data('session_data__master_rate_matrix__syn__2020-04-24_10-24-23.dat')
    I_si_array__imprt = data_array_imprt['I_si_array']
    I_drive_list__imprt = data_array_imprt['I_drive_list']
    rate_array__imprt = data_array_imprt['rate_array']
    
    I_drive_sought = 14.45
    I_drive_sought_ind = (np.abs(np.asarray(I_drive_list__imprt)-I_drive_sought)).argmin()
    I_si_sought = 4.552
    I_si_sought_ind = (np.abs(np.asarray(I_si_array__imprt[I_drive_sought_ind])-I_si_sought)).argmin()
    rate_obtained = rate_array__imprt[I_drive_sought_ind][I_si_sought_ind]
    
    print('I_drive_sought = {:7.4f}uA, I_drive_sought_ind = {:d}\nI_si_sought = {:7.4f}uA, I_si_sought_ind = {:d}\nrate_obtained = {:10.4f} fluxons per us'.format(I_drive_sought,I_drive_sought_ind,I_si_sought,I_si_sought_ind,rate_obtained))

