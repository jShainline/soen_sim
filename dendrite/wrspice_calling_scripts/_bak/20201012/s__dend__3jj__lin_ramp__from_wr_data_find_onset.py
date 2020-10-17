#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# from soen_sim import input_signal, synapse, dendrite, neuron
from _functions import save_session_data, read_wr_data
from _util import physical_constants, color_dictionary

p = physical_constants()
Phi0 = p['Phi0__pH_ns']

colors = color_dictionary()

plt.close('all')

#%% inputs

num_jjs = 3

M = np.sqrt(200*20) # pH

# DI loop saturation capacity current
dI_sc = 5
I_sc_vec = np.arange(20,30+dI_sc,dI_sc) # uA # np.asarray([30]) # 
num_I_sc = len(I_sc_vec)

# dendritic firing junction bias current
dI_de = 1
I_de_0__list = [65,64,60] # uA
I_de_f__list = [93,92,89] # uA

L_di = 77.5 # nH
         
min_peak_height = 100e-6 # units of volts for WR
min_peak_distance = 1 # units of samples

Phi_a_on__array = []
I_drive_on__array = []
I_de_list__array = []

for jj in range(num_I_sc):
    
    I_sc = I_sc_vec[jj]
    
    print('jj = {:d} of {:d} (I_sc = {:5.2f}uA)'.format(jj+1,num_I_sc,I_sc)) 
    
    
    I_de_list = np.arange(I_de_0__list[jj],I_de_f__list[jj]+dI_de,dI_de) # uA
    I_de_list__array.append(I_de_list)
    num_I_de = len(I_de_list)
    
    Phi_a_on = np.zeros([len(I_de_list)])
    I_drive_on = np.zeros([len(I_de_list)])
    
    for qq in range(num_I_de): # [0]: # [3]: # 
                    
        print('qq = {:d} of {:d} (I_de = {:5.2f}uA)'.format(qq+1,num_I_de,I_de_list[qq]))                
    
        directory = 'wrspice_data/{:1d}jj'.format(num_jjs)
        file_name = 'dend_{:d}jj_lin_ramp_Ide{:05.2f}uA_Isc{:05.2f}uA_Ldi{:07.2f}nH.dat'.format(num_jjs,I_de_list[qq],I_sc,L_di)
            
        j_di_str = 'v(4)'
        I_a_str = 'L2#branch'

        data_dict = read_wr_data(directory+'/'+file_name)
                    
        # assign data
        time_vec = data_dict['time']
        j_di = data_dict[j_di_str]
        I_a = 1e6*data_dict[I_a_str]
                    
        # find peaks    
        j_di_peaks, _ = find_peaks(j_di, height = min_peak_height, distance = min_peak_distance)
                    
        I_drive_on[qq] = I_a[j_di_peaks[0]]
        Phi_a_on[qq] = M*I_a[j_di_peaks[0]]
    
        # fig, ax = plt.subplots(nrows = 1, ncols = 1)
        # fig.suptitle('I_sc = {}uA, I_de = {}uA'.format(I_sc,I_de_list[qq]))
        # ax.plot(time_vec*1e9,j_di*1e6, color = colors['blue3'])
        # ax.plot(time_vec[j_di_peaks[0]]*1e9,j_di[j_di_peaks[0]]*1e6,'x', color = colors['red3'])
        # ax.set_xlim([0,time_vec[j_di_peaks[1]]*1e9])
        # ax.set_xlabel(r'time [ns]')
        # ax.set_ylabel(r'$J_{di}$ [$\mu$V]')
        # plt.show()
        
    Phi_a_on__array.append(Phi_a_on)
    I_drive_on__array.append(I_drive_on)
            
#%% save data
save_string = 'dend_{:1d}jj_flux_onset__Ldi{:7.2f}'.format(num_jjs,L_di)
data_array = dict()
data_array['I_drive_on__array'] = I_drive_on__array # uA
data_array['Phi_a_on__array'] = Phi_a_on__array # pH ns
data_array['I_de_list__array'] = I_de_list__array
data_array['I_sc_list'] = I_sc_vec
save_session_data(data_array,save_string+'.soen',False)

np.set_printoptions(precision=9)
_ts = 'I_drive_on__array = [['
for jj in range(len(I_sc_vec)):
    for ii in range(len(I_drive_on__array[jj])):
        _ts = '{}{:7.4f},'.format(_ts,I_drive_on__array[jj][ii])    
    _ts = '{}],['.format(_ts[0:-1])
    
print('{}]'.format(_ts[0:-2]))
# print('Phi_a_on = {}'.format(Phi_a_on))
# print('Phi_a_on/Phi0 = {}'.format(Phi_a_on/Phi0))
