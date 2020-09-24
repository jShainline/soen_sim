import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from _functions import read_wr_data, save_session_data

from util import physical_constants, color_dictionary

p = physical_constants()
colors = color_dictionary()

plt.close('all')

#%%

num_jjs = 4

if num_jjs == 2:       
    dI_de = 1
    I_de_0 = 53
    I_de_f = 80

if num_jjs == 4:
    dI_de = 1
    I_de_0 = 74
    I_de_f = 100
    
I_de_vec = np.arange(I_de_0,I_de_f+dI_de,dI_de)
num_I_de = len(I_de_vec)
 
Phi_th_vec = np.zeros([len(I_de_vec)])
L_in = 200e-12
L_left = 20e-12
M = np.sqrt(L_in*L_left)
directory_name = 'wrspice_data/{:d}jj/direct_drive__lin_ramp__find_flux_threshold'.format(num_jjs)
for ii in range(len(I_de_vec)):
    
    print('ii = {} of {}'.format(ii+1,len(I_de_vec)))
    
    I_de = I_de_vec[ii] 
    
    file_name = 'ne_{:d}jj_direct_drive_lin_ramp_alt_ind_Ldrv200pH_Lnr20pH20pH_Ide{:05.2f}uA_taunf50.00ns_dt01.0ps.dat'.format(num_jjs,I_de)    
    data_dict = read_wr_data('{}/{}'.format(directory_name,file_name))
    if num_jjs == 2:
        I_nf_str = 'L0#branch'
        I_drive_str = 'L1#branch'
    elif num_jjs == 4:
        I_nf_str = 'L2#branch'
        I_drive_str = 'L3#branch'
    
    time_vec = data_dict['time']
    dt = time_vec[1]-time_vec[0]
    I_nf_vec = data_dict[I_nf_str]
    flux_drive_vec = M*data_dict[I_drive_str]

    I_nf_peaks, _ = find_peaks(I_nf_vec, distance = 10e-9/dt, height = 10e-6) #     
    Phi_th_vec[ii] = flux_drive_vec[I_nf_peaks[0]]
    # else:
        
    
    # fig = plt.figure() 
    # ax = fig.gca()
    # ax.plot(time_vec*1e9,I_nf_vec*1e6, '-', color = colors['blue3'], label = 'I_nf')    
    # ax.plot(time_vec[I_nf_peaks]*1e9,I_nf_vec[I_nf_peaks]*1e6, 'x', color = colors['red3'], label = 'I_nf_peaks')    
    # ax.set_xlabel(r'Time [ns]')
    # ax.set_ylabel(r'I_nf [uA]')
    # ax.legend()    
    # plt.show()

I_drive_th_vec = Phi_th_vec/M

print('Phi_th_vec = {}'.format(Phi_th_vec))
print('I_drive_th_vec = {}'.format(I_drive_th_vec))

#%% plot
fig = plt.figure()
# fig.suptitle('Isi vs Isy; tau_si = inf; L_si = {:7.4f} nH'.format(synapse_list[0].integration_loop_total_inductance*1e9)) 
ax = fig.gca()

ax.plot(I_de_vec,Phi_th_vec/p['Phi0'], '-o', color = colors['blue3'])
ax.set_xlabel(r'$I_{de}$ [$\mu$A]')
ax.set_ylabel(r'$\Phi_{th}^{nr}/\Phi_0$')

plt.show()

#%% save data
save_string = 'master_neu_flx_thr_{:1d}jj_Llft20.0_Lrgt20.0_Lnf65.0'.format(num_jjs)
data_array = dict()
data_array['Phi_th_vec'] = Phi_th_vec
data_array['I_de_vec'] = I_de_vec
print('\n\nsaving session data ...\n\n')
# # save_session_data(data_array,save_string)
save_session_data(data_array,save_string+'.soen',False)

#%% calculate I_drive_array for s__neu__2jj__cnst_drv__for_reset.py

I_drive_array = []
dI_drive = 1
I_f = I_drive_th_vec[0]*1e6 # p['Phi0']*1e6/(2*M)
for ii in range(len(I_drive_th_vec)):
    I_0 = I_drive_th_vec[ii]*1e6
    I_drive_vec = np.append(np.arange(I_0,I_f,dI_drive),I_f)
    I_drive_array.append(I_drive_vec)
    
save_string = 'I_drive_array_{:1d}jj_Llft20.0_Lrgt20.0_Lnf65.0'.format(num_jjs)
data_array = dict()
data_array['I_drive_array'] = I_drive_array
data_array['I_de_vec'] = I_de_vec
print('\n\nsaving session data ...\n\n')
# # save_session_data(data_array,save_string)
save_session_data(data_array,save_string+'.soen',False)

#%% because grumpy doesn't run pickle, paste this in s__neu__2jj__cnst_drv__for_reset.py
file_string = 'I_drive_array = ['
for ii in range(len(I_drive_th_vec)):
    I_drive_vec = I_drive_array[ii]
    file_string += '['
    for jj in range(len(I_drive_vec)):
        file_string += '{:4.2f},'.format(np.round(I_drive_vec[jj],6))
    file_string = file_string[0:-1]+'],'
file_string = file_string[0:-1]+']' 
        
print(file_string)

file_string = 'I_de_vec = ['
for ii in range(len(I_de_vec)):
    file_string += '{:4.2f},'.format(np.round(I_de_vec[ii],6))
file_string = file_string[0:-1]+']' 
print(file_string)

        