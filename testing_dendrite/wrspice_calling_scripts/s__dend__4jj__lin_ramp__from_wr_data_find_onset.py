#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# from soen_sim import input_signal, synapse, dendrite, neuron
from _functions import save_session_data, read_wr_data
from util import physical_constants

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
    min_peak_height = 100e-6 # units of volts for WR
    min_peak_distance = 1 # units of samples
elif num_jjs == 4:        
    min_peak_height = 182e-6 # units of volts for WR
    min_peak_distance = 10 # units of samples

Phi_a_on = np.zeros([len(I_de_list)])
I_drive_on = np.zeros([len(I_de_list)])
for qq in range(num_I_de): # [0]: # [3]: # 
                
    print('qq = {:d} of {:d} (I_de = {:5.2f}uA)'.format(qq+1,num_I_de,I_de_list[qq]))                

    directory = 'wrspice_data/{:1d}jj'.format(num_jjs)
    file_name = 'dend_{:d}jj_lin_ramp_I_de{:05.2f}uA.dat'.format(num_jjs,I_de_list[qq])
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
    I_a = 1e6*data_dict[I_a_str]
                
    # find peaks    
    j_di_peaks, _ = find_peaks(j_di, height = min_peak_height, distance = min_peak_distance)
                
    I_drive_on[qq] = I_a[j_di_peaks[0]]
    Phi_a_on[qq] = M*I_a[j_di_peaks[0]]

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(time_vec*1e9,j_di*1e6)
    ax.plot(time_vec[j_di_peaks[0]]*1e9,j_di[j_di_peaks[0]]*1e6,'x')
    ax.set_xlim([0,time_vec[j_di_peaks[1]]*1e9])
    ax.set_xlabel(r'time [ns]')
    ax.set_ylabel(r'$J_{di}$ [$\mu$V]')
    plt.show()
            
#%% save data
save_string = 'dend_{:1d}jj_flux_onset'.format(num_jjs)
data_array = dict()
data_array['I_de_list'] = I_de_list # uA
data_array['Phi_a_on'] = Phi_a_on # pH ns
save_session_data(data_array,save_string+'.soen',False)

np.set_printoptions(precision=9)
_ts = 'I_drive_on__vec = ['
for ii in range(len(I_drive_on)):
    _ts = '{}{:7.4f},'.format(_ts,I_drive_on[ii])
print('{}]'.format(_ts[0:-1]))
print('Phi_a_on = {}'.format(Phi_a_on))
print('Phi_a_on/Phi0 = {}'.format(Phi_a_on/Phi0))
