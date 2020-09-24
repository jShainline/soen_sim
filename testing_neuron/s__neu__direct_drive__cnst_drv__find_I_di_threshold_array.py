import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import pickle

from _functions import read_wr_data, save_session_data

from util import physical_constants, color_dictionary

p = physical_constants()
colors = color_dictionary()

plt.close('all')

#%%

num_jjs = 4
L_in = 200
L_left = 20
M = np.sqrt(L_in*L_left)

Phi_th_array = []

plot_time_traces = False

I_drive_pad = 0.010

#%%

file_name = 'I_drive_array_{:d}jj_Llft20.0_Lrgt20.0_Lnf65.0.soen'.format(num_jjs)
with open('soen_sim_data/'+file_name, 'rb') as data_file:         
    data_array = pickle.load(data_file)
    I_de_vec = data_array['I_de_vec']
    I_drive_array = data_array['I_drive_array']
 
I_de_start = 74 # 57   
I_de_stop = 95 # 90
I_de_start_ind = (np.abs(I_de_vec-I_de_start)).argmin() 
I_de_stop_ind = (np.abs(I_de_vec-I_de_stop)).argmin()
I_de_vec = I_de_vec[I_de_start_ind:I_de_stop_ind+1]
num_I_de = len(I_de_vec)
 
directory_name = 'wrspice_data/{:d}jj/direct_drive__cnst_drv__find_reset'.format(num_jjs)
I_di_th_array = []
Phi_dr_applied_array = []
I_fq1_array = []
I_fq2_array = []
for ii in range(num_I_de):
    I_de = I_de_vec[ii]
    I_drive_vec = I_drive_array[ii+I_de_start_ind]    
    Phi_dr_applied_array.append(np.insert(I_drive_vec,0,I_drive_vec[0]-I_drive_pad)*M) # extra zero for case where flux is below threshold value
    I_di_th_array.append([])
    I_fq1_array.append([])
    I_fq2_array.append([])
    
    I_di_th_vec = np.zeros([len(I_drive_vec)])
    I_fq1_vec = np.zeros([len(I_drive_vec)])
    I_fq2_vec = np.zeros([len(I_drive_vec)])
    for jj in range(len(I_drive_vec)):
        I_drive = I_drive_vec[jj]
    
        print('ii = {} of {} (I_de = {:5.2f}uA); jj = {} of {}'.format(ii+1,len(I_de_vec),I_de_vec[ii],jj+1,len(I_drive_vec)))
        
        file_name = 'ne_{:d}jj_direct_drive_cnst_drv_alt_ind_Lnr20pH20pH_Ide{:5.2f}uA_Ldrv200pH_Idrv{:05.2f}_taunf50.00ns_dt01.0ps.dat'.format(num_jjs,I_de,I_drive)    
        data_dict = read_wr_data('{}/{}'.format(directory_name,file_name))
        if num_jjs == 2:
            I_nf_str = 'L0#branch'
        elif num_jjs == 4:
            I_nf_str = 'L2#branch'
        
        time_vec = data_dict['time']
        dt = time_vec[1]-time_vec[0]
        I_nf_vec = data_dict[I_nf_str]
    
        I_nf_peaks, _ = find_peaks(I_nf_vec,distance = 40e-9/dt) # , height = 1e-6, , 
        I_nf_vec__part = I_nf_vec[I_nf_peaks[0]:]
        I_nf_min = np.min(I_nf_vec__part)
        I_nf_max = I_nf_vec[I_nf_peaks[1]]
        I_di_th_vec[jj] = I_nf_min*1e6
        I_fq1_vec[jj] = 1e6*I_nf_vec[I_nf_peaks[0]]
        I_fq2_vec[jj] = 1e6*(I_nf_max-I_nf_min)
        
        if plot_time_traces == True:
            fig = plt.figure() 
            fig.suptitle(file_name)
            ax = fig.gca()
            ax.plot(time_vec*1e9,I_nf_vec*1e6, '-', color = colors['blue3'], label = 'I_nf')    
            ax.plot(time_vec[I_nf_peaks]*1e9,I_nf_vec[I_nf_peaks]*1e6, 'x', color = colors['green3'], label = 'I_nf_peaks')    
            ax.plot([time_vec[0]*1e9,time_vec[-1]*1e9],[I_nf_min*1e6,I_nf_min*1e6], ':', color = colors['red3'], label = 'I_nf_min') 
            ax.set_xlabel(r'Time [ns]')
            ax.set_ylabel(r'I_nf [uA]')
            ax.legend()    
            plt.show()
        
    I_di_th_array[ii] = np.insert(I_di_th_vec,0,0) # extra zero for case where flux is below threshold value
    I_fq1_array[ii] = np.insert(I_fq1_vec,0,0)
    I_fq2_array[ii] = np.insert(I_fq2_vec,0,0)
        
#%% plot
fig = plt.figure()
# fig.suptitle('Isi vs Isy; tau_si = inf; L_si = {:7.4f} nH'.format(synapse_list[0].integration_loop_total_inductance*1e9)) 
ax = fig.gca()
for ii in range(num_I_de):
    ax.plot(Phi_dr_applied_array[ii]*1e-18/p['Phi0'],I_di_th_array[ii], '-o', label = 'Ide = {:5.2f}'.format(I_de_vec[ii]))
ax.set_xlabel(r'$\Phi_{a}^{dr}/\Phi_0$')
ax.set_ylabel(r'$I^{di}_{th}$ [$\mu$A]')
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.gca()
for ii in range(num_I_de):
    ax.plot(Phi_dr_applied_array[ii]*1e-18/p['Phi0'],I_fq1_array[ii], '-o', label = 'Ifq1; Ide = {:5.2f}'.format(I_de_vec[ii]))
    ax.plot(Phi_dr_applied_array[ii]*1e-18/p['Phi0'],I_fq2_array[ii], '-o', label = 'Ifq2; Ide = {:5.2f}'.format(I_de_vec[ii]))
ax.set_xlabel(r'$\Phi_{a}^{dr}/\Phi_0$')
ax.set_ylabel(r'$I_{fq}$ [$\mu$A]')
ax.legend()
plt.show()

#%% save data
save_string = 'master_neu_Idr_thr_{:1d}jj_Llft20.0_Lrgt20.0_Lnf65.0'.format(num_jjs)
data_array = dict()
data_array['I_di_th_array'] = I_di_th_array
data_array['I_fq1_array'] = I_fq1_array
data_array['I_fq2_array'] = I_fq2_array
data_array['Phi_dr_applied_array'] = Phi_dr_applied_array
data_array['I_de_vec'] = I_de_vec
print('\n\nsaving session data ...\n\n')
# # save_session_data(data_array,save_string)
save_session_data(data_array,save_string+'.soen',False)



        