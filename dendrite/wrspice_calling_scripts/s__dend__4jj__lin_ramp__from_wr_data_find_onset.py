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

# plt.close('all')

#%% inputs

Ma = np.sqrt(400*12.5) # pH
Mp = 77.5 # pH

# dendritic series bias current
dIb = 2
Ib_vec = np.arange(130,134+dIb,dIb) # uA

# dendritic plasticity bias current
num_steps = 11 # num steps on either side of zero
max_flux = Phi0/2
flux_resolution = max_flux/num_steps
Mp = 77.5
Ip_max = max_flux/Mp
Ip_vec = np.linspace(0,Ip_max,num_steps)
Ip_vec = np.concatenate((np.flipud(-Ip_vec)[0:-1],Ip_vec))

# activity flux input
Ma = np.sqrt(400*12.5)
Ia_max = max_flux/Ma

# L_di = 77.5 # nH

# find peaks
         
min_peak_height = 20e-6 # units of volts for WR
min_peak_distance = 1 # units of samples (10 ps time step)

Phi_a_on_pos__array = []
Phi_a_on_neg__array = []

Phi_p_neg__array = []
Phi_p_pos__array = []

for ii in range(len(Ib_vec)):
    Ib = Ib_vec[ii]
    Phi_a_on_neg__list = []
    Phi_a_on_pos__list = []
    Phi_p_neg__list = []
    Phi_p_pos__list = []
    for jj in range(len(Ip_vec)):
        Ip = Ip_vec[jj]
        
        for aa in [-1,1]:
    
    
            if aa == -1:
                _tn = 1
            elif aa == 1:
                _tn = 2
            print('ii = {} of {}; jj = {} of {}; aa = {} of {}'.format(ii+1,len(Ib_vec),jj+1,len(Ip_vec),_tn,2))             
            
            directory = 'wrspice_data/4jj'
            file_name = 'dend_4jj_one_bias_plstc_lin_ramp_Ib{:05.2f}uA_Ip{:05.2f}uA_rmp{:1d}.dat'.format(Ib,Ip,aa)
                
            j_sq_str = 'v(2)'
            Ia_str = 'L8#branch'
    
            data_dict = read_wr_data(directory+'/'+file_name)
                        
            # assign data
            time_vec = data_dict['time']
            j_sq = data_dict[j_sq_str]
            Ia = 1e6*data_dict[Ia_str]
                        
            # find peaks    
            j_sq_peaks, _ = find_peaks(j_sq, height = min_peak_height, distance = min_peak_distance)
                                    
            #%% plot
            # fig, ax = plt.subplots(nrows = 1, ncols = 1)
            # fig.suptitle('Ib = {:6.2f}uA, Ip = {:7.4f}uA'.format(Ib,Ip))
            # ax.plot(time_vec*1e9,j_sq*1e6, color = colors['blue3'])
            # ax.plot(time_vec[j_sq_peaks[0]]*1e9,j_sq[j_sq_peaks[0]]*1e6,'x', color = colors['red3'])
            # ax.set_xlim([0,time_vec[j_sq_peaks[1]]*1e9+3])
            # ax.set_xlabel(r'time [ns]')
            # ax.set_ylabel(r'$J_{di}$ [$\mu$V]')
            # plt.show()
            
            if len(j_sq_peaks) > 1:
                
                if aa == -1:
                    Phi_p_neg__list.append(Ip*Mp)
                    Phi_a_on_neg__list.append(Ma*Ia[j_sq_peaks[0]])
                elif aa == 1:
                    Phi_p_pos__list.append(Ip*Mp)
                    Phi_a_on_pos__list.append(Ma*Ia[j_sq_peaks[0]])
                        
    Phi_a_on_neg__array.append(np.asarray(Phi_a_on_neg__list))
    Phi_a_on_pos__array.append(np.asarray(Phi_a_on_pos__list))
    Phi_p_neg__array.append(np.asarray(Phi_p_neg__list))
    Phi_p_pos__array.append(np.asarray(Phi_p_pos__list))
    
#%% plot

fig, ax = plt.subplots(nrows = 1, ncols = 1)
# fig.suptitle('Ib = {:6.2f}uA, Ip = {:7.4f}uA'.format(Ib,Ip))
color_list = ['blue3','green3','yellow3']
for ii in range(len(Ib_vec)):
    ax.plot(Phi_p_pos__array[ii]/Phi0,Phi_a_on_pos__array[ii]/Phi0, color = colors[color_list[ii]], label = 'Ib = {}uA'.format(Ib_vec[ii]))
ax.set_xlim([-1/2,1/2])
ax.set_ylim([0,1/2])
ax.set_xlabel(r'$\Phi_p/\Phi_0$')
ax.set_ylabel(r'$\Phi_a^{on+}/\Phi_0$')
ax.legend()
plt.show()
 
fig, ax = plt.subplots(nrows = 1, ncols = 1)
# fig.suptitle('Ib = {:6.2f}uA, Ip = {:7.4f}uA'.format(Ib,Ip))
color_list = ['blue3','green3','yellow3']
for ii in range(len(Ib_vec)):
    ax.plot(Phi_p_neg__array[ii]/Phi0,-Phi_a_on_neg__array[ii]/Phi0, color = colors[color_list[ii]], label = 'Ib = {}uA'.format(Ib_vec[ii]))
ax.set_xlim([-1/2,1/2])
ax.set_ylim([0,1/2])
ax.set_xlabel(r'$\Phi_p/\Phi_0$')
ax.set_ylabel(r'$\Phi_a^{on-}/\Phi_0$')
ax.legend()
plt.show()
           
#%% save data
save_string = 'dend_4jj_flux_onset'
data_array = dict()
data_array['Ib_vec'] = Ib_vec # uA
data_array['Phi_a_on_neg__array'] = Phi_a_on_neg__array # pH ns
data_array['Phi_a_on_pos__array'] = Phi_a_on_pos__array
data_array['Phi_p_neg__array'] = Phi_p_neg__array
data_array['Phi_p_pos__array'] = Phi_p_pos__array
save_session_data(data_array,save_string+'.soen',False)

#%% print for grumpy

np.set_printoptions(precision=9)
_ts = 'Phi_a_on_neg__array = [['
for jj in range(len(Ib_vec)):
    for ii in range(len(Phi_a_on_neg__array[jj])):
        _ts = '{}{:f},'.format(_ts,Phi_a_on_neg__array[jj][ii])    
    _ts = '{}],['.format(_ts[0:-1])
print('{}]'.format(_ts[0:-2]))

_ts = 'Phi_a_on_pos__array = [['
for jj in range(len(Ib_vec)):
    for ii in range(len(Phi_a_on_pos__array[jj])):
        _ts = '{}{:f},'.format(_ts,Phi_a_on_pos__array[jj][ii])    
    _ts = '{}],['.format(_ts[0:-1])
print('{}]'.format(_ts[0:-2]))

_ts = 'Phi_p_neg__array = [['
for jj in range(len(Ib_vec)):
    for ii in range(len(Phi_p_neg__array[jj])):
        _ts = '{}{:f},'.format(_ts,Phi_p_neg__array[jj][ii])    
    _ts = '{}],['.format(_ts[0:-1])
print('{}]'.format(_ts[0:-2]))

_ts = 'Phi_p_pos__array = [['
for jj in range(len(Ib_vec)):
    for ii in range(len(Phi_p_pos__array[jj])):
        _ts = '{}{:f},'.format(_ts,Phi_p_pos__array[jj][ii])    
    _ts = '{}],['.format(_ts[0:-1])
print('{}]'.format(_ts[0:-2]))
