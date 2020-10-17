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

# dendritic series bias current
# dIb = 10
# Ib_vec = np.arange(85,105+dIb,dIb) # uA
Ib_vec = [85] # uA

# inductor in plasticity loop
# Lp_list = [20,30,40] # [35] # [25,30,35] # data sets available: 12.5,17.5,18.75,20,25,30,35,40 # pH
Lp_list = [35] # [35] # [25,30,35] # data sets available: 12.5,17.5,18.75,20,25,30,35,40 # pH

# dendritic plasticity bias current
num_steps = 21 # num steps on either side of zero # 11 for Lp_list = 12.5,17.5,18.75,20,25,30, Ib_vec = 85:5:105
max_flux = Phi0/2
flux_resolution = max_flux/num_steps

# excitatory flux input
Mex = np.sqrt(400*12.5)
Iex_max = max_flux/Mex

# find peaks
         
min_peak_height = 20e-6 # units of volts for WR
min_peak_distance = 1 # units of samples (10 ps time step)

Phi_ex_on__array = []
Phi_p__array = []

for qq in range(len(Lp_list)):
    
    Lp = Lp_list[qq]
    Mp = np.sqrt(200*Lp)
    Ip_max = max_flux/Mp
    Ip_vec = np.linspace(0,Ip_max,num_steps)
    Ip_vec = np.concatenate((np.flipud(-Ip_vec)[0:-1],Ip_vec)) 
    
    
    Phi_ex_on__sub_array = []
    Phi_p__sub_array = []
    
    for ii in range(len(Ib_vec)):
        Ib = Ib_vec[ii]
        Phi_ex_on__list = []
        Phi_p__list = []
        for jj in range(len(Ip_vec)):
            Ip = Ip_vec[jj]
            
            print('qq = {} of {}; ii = {} of {}; jj = {} of {}'.format(qq+1,len(Lp_list),ii+1,len(Ib_vec),jj+1,len(Ip_vec)))             
            
            directory = 'wrspice_data/3jj'
            file_name = 'dend_3jj_nest_lin_ramp_Ib{:05.2f}uA_Lp{:05.2f}pH_Ip{:09.6f}uA.dat'.format(Ib,Lp,Ip)
                
            j_sq_str = 'v(2)'
            Iex_str = 'i(L4)'
    
            data_dict = read_wr_data(directory+'/'+file_name)
                        
            # assign data
            time_vec = data_dict['time']
            j_sq = data_dict[j_sq_str]
            Iex = 1e6*data_dict[Iex_str]
                        
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
    
                Phi_p__list.append(Ip*Mp)
                Phi_ex_on__list.append(Mex*Iex[j_sq_peaks[0]])
                            
        Phi_ex_on__sub_array.append(np.asarray(Phi_ex_on__list))
        Phi_p__sub_array.append(np.asarray(Phi_p__list))
        
    Phi_ex_on__array.append(Phi_ex_on__sub_array)
    Phi_p__array.append(Phi_p__sub_array)
    
#%% plot

fig, ax = plt.subplots(nrows = 1, ncols = 1)
# fig.suptitle('Lp'.format(Lp))
color_list = [['blue1','green1','yellow1'],['blue3','green3','yellow3'],['blue5','green5','yellow5'],['grey5','grey8','grey10']]
for jj in range(len(Lp_list)):
    for ii in range(len(Ib_vec)):
        ax.plot(Phi_p__array[jj][ii]/Phi0,Phi_ex_on__array[jj][ii]/Phi0, color = colors[color_list[jj][ii]], label = 'Lp = {:05.2f}pH; Ib = {:06.2f}uA'.format(Lp_list[jj],Ib_vec[ii]))
ax.set_xlim([-1/2,1/2])
ax.set_ylim([0,1/2])
ax.set_xlabel(r'$\Phi_p/\Phi_0$')
ax.set_ylabel(r'$\Phi_ex^{on}/\Phi_0$')
ax.legend()
plt.show()
           
#%% save data
save_string = 'dend_3jj_flux_onset'
data_array = dict()
data_array['Lp_list'] = Lp_list # pH
data_array['Ib_vec'] = Ib_vec # uA
data_array['Phi_ex_on__array'] = Phi_ex_on__array # pH ns
data_array['Phi_p__array'] = Phi_p__array
save_session_data(data_array,save_string+'.soen',False)

#%% print for grumpy

Lp_ind = 0
Ib_ind = 0

np.set_printoptions(precision=9)
_temp_array  = Phi_ex_on__array[Lp_ind][Ib_ind]
_ts = 'Phi_ex_on_list = ['
for ii in range(len(_temp_array)):
    _ts = '{}{:f},'.format(_ts,_temp_array[ii])    
print('{}]'.format(_ts[0:-1]))

_temp_array = Phi_p__array[Lp_ind][Ib_ind]
_ts = 'Phi_p_list = ['
for ii in range(len(_temp_array)):
    _ts = '{}{:f},'.format(_ts,_temp_array[ii])
print('{}]'.format(_ts[0:-1]))

