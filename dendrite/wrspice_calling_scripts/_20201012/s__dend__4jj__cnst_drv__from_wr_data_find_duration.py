#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# from soen_sim import input_signal, synapse, dendrite, neuron
from _functions import save_session_data, read_wr_data
from util import physical_constants, color_dictionary

colors = color_dictionary()

p = physical_constants()
Phi0 = p['Phi0__pH_ns']

plt.close('all')

#%% set case

num_jjs = 4 # 2 or 4

#%% inputs

M = np.sqrt(200*20) # pH

if num_jjs == 2:

    # dendritic firing junction bias current
    dI_de = 1
    I_de_0 = 52
    I_de_f = 80

elif num_jjs == 4:
    
    # dendritic firing junction bias current
    dI_de = 1
    I_de_0 = 56
    I_de_f = 85
    
I_de_list = np.arange(I_de_0,I_de_f+dI_de,dI_de) # uA
num_I_de = len(I_de_list)

if num_jjs == 2:        
    min_peak_height = 10e-6 # units of volts for WR
    min_peak_distance = 1 # units of samples
elif num_jjs == 4:        
    min_peak_height = 10e-6 # units of volts for WR
    min_peak_distance = 1 # units of samples

#%%

I_drive_on__vec = [16.1648,15.7114,15.2568,14.8008,14.3432,13.8840,13.4228,12.9596,12.4940,12.0260,11.5552,11.0812,10.6038,10.1226, 9.6372, 9.1470, 8.6510, 8.1490, 7.6398, 7.1220, 6.5942, 6.0544, 5.4998, 4.9266, 4.3284, 3.6958, 3.0102, 2.2290, 1.1768, 0.0000]

num_steps = 200

max_flux = p['Phi0__pH_ns']/2
flux_resolution = max_flux/num_steps
I_drive_off = np.ceil(max_flux/flux_resolution)*flux_resolution/M
dI_drive = flux_resolution/M

#%%
time_off__array = []
I_drive__array = []
for qq in range(num_I_de): # [0]: # [3]: # 
    I_drive_vec = np.arange(I_drive_on__vec[qq],I_drive_off,dI_drive)
    time_off = np.zeros([len(I_drive_vec)])
    for jj in range(len(I_drive_vec)):
    
        print('qq = {:d} of {:d} (I_de = {:5.2f}uA); jj = {:d} of {:d}'.format(qq+1,num_I_de,I_de_list[qq],jj+1,len(I_drive_vec)))             
    
        directory = 'wrspice_data/{:1d}jj'.format(num_jjs)
        file_name = 'dend_{:d}jj_cnst_drv_seek_dur_Ide{:05.2f}uA_Idrive{:08.5f}uA.dat'.format(num_jjs,I_de_list[qq],I_drive_vec[jj])
        if num_jjs == 2:            
            j_di_str = 'v(3)'
            I_a_str = 'i(L0)'
        elif num_jjs == 4: 
            j_di_str = 'v(5)'
            j_di_phase_str = 'v(12)'
            I_a_str = 'L3#branch'
        data_dict = read_wr_data(directory+'/'+file_name)
                    
        # assign data
        time_vec = data_dict['time']
        j_di = data_dict[j_di_str]
                    
        # find peaks    
        j_di_peaks, _ = find_peaks(j_di, height = min_peak_height, distance = min_peak_distance)
                    
        time_off[jj] = time_vec[j_di_peaks[-1]]+5*(time_vec[j_di_peaks[-1]]-time_vec[j_di_peaks[-2]])
    
        # fig, ax = plt.subplots(nrows = 1, ncols = 1)
        # ax.plot(time_vec*1e9,j_di*1e6, color = colors['blue3'])
        # ax.plot(time_vec[j_di_peaks[-1]]*1e9,j_di[j_di_peaks[-1]]*1e6,'x', color = colors['red3'])
        # ax.set_xlim([0,time_off[jj]*1e9])
        # ax.set_xlabel(r'time [ns]')
        # ax.set_ylabel(r'$J_{di}$ [$\mu$V]')
        # plt.show()
        
    time_off__array.append(time_off)
    I_drive__array.append(I_drive_vec)
    
#%% check that none are close to the end of sim time
for ii in range(len(time_off__array)):
    print('I_de = {:5.2f}uA; max(time_off) = {:4.0f}ns'.format(I_de_list[ii],np.max(time_off__array[ii]*1e9)))
        
            
#%% save data
save_string = 'dend_{:1d}jj__I_drive_array'.format(num_jjs)
data_array = dict()
data_array['I_de_list'] = I_de_list # uA
data_array['I_drive__array'] = I_drive__array # uA
save_session_data(data_array,save_string+'.soen',False)

#%% print for grumpy
np.set_printoptions(precision=1)
_ts = 'time_off__array = [['
for ii in range(num_I_de):
    if ii > 0:
        _ts = '{}],['.format(_ts)
    for jj in range(len(time_off__array[ii])):
        _ts = '{}{:6.1f},'.format(_ts,time_off__array[ii][jj]*1e9)
    _ts = '{}'.format(_ts[0:-1])
_ts = '{}]]'.format(_ts)
print(_ts)
# print('{}]'.format(_ts[0:-1]))
# print('Phi_a_on = {}'.format(Phi_a_on))
# print('Phi_a_on/Phi0 = {}'.format(Phi_a_on/Phi0))
