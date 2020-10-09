#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3
from util import physical_constants
p = physical_constants()

plt.close('all')

#%% load wr data, find peaks, find rates

dI = 1
I_drive_vec = [18.6,19,20,21,22,23,24,25,26,27,28,29,30]#np.arange(19,30+dI,dI)
t_sim_vec = [60,70,50,40,35,32,32,32,32,32,32,32,32]
data_file_list = []
num_files = len(I_drive_vec)
for ii in range(num_files):
    data_file_list.append('dend_cnst_drv_Idrv{:2.2f}uA_Ldi0077.50nH_taudi0775ms_tsim{:04.0f}ns_dt00.1ps.dat'.format(I_drive_vec[ii],t_sim_vec[ii]))

master_data__df1__t_fq = []
master_data__df1__r_fq = []
master_data__df2__t_fq = []
master_data__df2__r_fq = []
master_data__jtl__t_fq = []
master_data__jtl__r_fq = []
master_data__di__t_fq = []
master_data__di__r_fq = []
max_I_di = 0
I_di_list = []
for ii in range(num_files):
    
    print('ii = {:d} of {:d}'.format(ii+1,num_files))
    
    directory = 'wrspice_data/fitting_data'
    file_name = data_file_list[ii]
    data_dict = read_wr_data(directory+'/'+file_name)
    
    # find peaks for each jj
    time_vec = data_dict['time']
    j_df = data_dict['v(3)']
    j_jtl = data_dict['v(4)']
    j_di = data_dict['v(5)']
    
    initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
    final_ind = (np.abs(time_vec-t_sim_vec[ii]*1e-9)).argmin()

    time_vec = time_vec[initial_ind:final_ind]
    j_df = j_df[initial_ind:final_ind]
    j_jtl = j_jtl[initial_ind:final_ind]
    j_di = j_di[initial_ind:final_ind]
  
    j_df1_peaks, _ = find_peaks(j_df, height = [187.8e-6,300e-6])
    j_df2_peaks, _ = find_peaks(j_df, height = [140e-6,187.6e-6])
    j_jtl_peaks, _ = find_peaks(j_jtl, height = 200e-6)
    j_di_peaks, _ = find_peaks(j_di, height = 200e-6)

    fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)
    fig.suptitle(file_name)   
    ax[0].plot(time_vec*1e9,j_df*1e3, '-', label = '$J_{df}$')             
    ax[0].plot(time_vec[j_df1_peaks]*1e9,j_df[j_df1_peaks]*1e3, 'x')
    ax[0].plot(time_vec[j_df2_peaks]*1e9,j_df[j_df2_peaks]*1e3, 'x')
    ax[0].set_xlabel(r'Time [ns]')
    ax[0].set_ylabel(r'Voltage [mV]')
    ax[0].legend()
    ax[1].plot(time_vec*1e9,j_jtl*1e3, '-', label = '$J_{jtl}$')             
    ax[1].plot(time_vec[j_jtl_peaks]*1e9,j_jtl[j_jtl_peaks]*1e3, 'x')
    ax[1].set_xlabel(r'Time [ns]')
    ax[1].set_ylabel(r'Voltage [mV]')
    ax[1].legend()
    ax[2].plot(time_vec*1e9,j_di*1e3, '-', label = '$J_{di}$')             
    ax[2].plot(time_vec[j_di_peaks]*1e9,j_di[j_di_peaks]*1e3, 'x')
    ax[2].set_xlabel(r'Time [ns]')
    ax[2].set_ylabel(r'Voltage [mV]')
    ax[2].legend()
    plt.show()
    
    # find inter-fluxon intervals and fluxon generation rates for each JJ
    I_di = data_dict['L3#branch']
    I_di = I_di[initial_ind:final_ind]
    I_drive = data_dict['L4#branch']
    I_drive = I_drive[initial_ind:final_ind]    
    max_I_di = np.max([np.max(I_di),max_I_di])
    
    # I_dr1 = Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,Iflux_vec[jj],I_b,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
    
    j_df1_ifi = np.diff(time_vec[j_df1_peaks])
    j_df2_ifi = np.diff(time_vec[j_df2_peaks])
    j_jtl_ifi = np.diff(time_vec[j_jtl_peaks])
    j_di_ifi = np.diff(time_vec[j_di_peaks])
    
    j_df1_rate = 1/j_df1_ifi
    j_df2_rate = 1/j_df2_ifi
    j_jtl_rate = 1/j_jtl_ifi
    j_di_rate = 1/j_di_ifi
    
    master_data__df1__t_fq.append(j_df1_ifi)
    master_data__df1__r_fq.append(j_df1_rate)
    master_data__df2__t_fq.append(j_df2_ifi)
    master_data__df2__r_fq.append(j_df2_rate)
    master_data__jtl__t_fq.append(j_jtl_ifi)
    master_data__jtl__r_fq.append(j_jtl_rate)
    master_data__di__t_fq.append(j_di_ifi)
    master_data__di__r_fq.append(j_di_rate)
    
    I_di_list.append(I_di[j_di_peaks])
    
#%% assemble data
longest_rate_vec = 0
fastest_rate = 0
for ii in range(num_files):
    if len(master_data__di__r_fq[ii]) > longest_rate_vec:
        longest_rate_vec = len(master_data__di__r_fq[ii])
    fastest_rate = np.max([np.max(master_data__di__r_fq[ii]),fastest_rate])        

master_rate_matrix = np.zeros([num_files,longest_rate_vec])
for ii in range(num_files):
    tn = len(master_data__di__r_fq[ii])
    for jj in range(tn):
        master_rate_matrix[ii,jj] = master_data__di__r_fq[ii][jj]

fig, ax = plt.subplots(1,1)
rates = ax.imshow(np.transpose(master_rate_matrix[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[I_drive_vec[0],I_drive_vec[-1],0,max_I_di*1e6], aspect = 'auto', origin = 'lower')
cbar = fig.colorbar(rates, extend='both')
cbar.minorticks_on()     
fig.suptitle('$r_{tot}$ versus $I_{drive}$ and $I_{di}$')
# plt.title(title_string)
ax.set_xlabel(r'$I_{drive}$ [$\mu$A]')
ax.set_ylabel(r'$I_{di}$ [$\mu$A]')   
plt.show()      
# fig.savefig('figures/'+save_str+'__log.png')

#%% save data
save_string = 'master_rate_matrix'
data_array = dict()
data_array['master_rate_matrix'] = master_rate_matrix
data_array['I_drive_vec'] = I_drive_vec*1e-6
data_array['I_di_list'] = I_di_list
print('\n\nsaving session data ...')
save_session_data(data_array,save_string)

#%% load test
data_array_imported = load_session_data('session_data__master_rate_matrix__2020-04-17_13-11-03.dat')
I_di_list__imported = data_array_imported['I_di_list']
I_drive_vec__imported = data_array_imported['I_drive_vec']
master_rate_matrix__imported = data_array_imported['master_rate_matrix']

I_drive_sought = 23.45e-6
I_drive_sought_ind = (np.abs(np.asarray(I_drive_vec__imported)-I_drive_sought)).argmin()
I_di_sought = 14.552e-6
I_di_sought_ind = (np.abs(np.asarray(I_di_list__imported[I_drive_sought_ind])-I_di_sought)).argmin()
rate_obtained = master_rate_matrix__imported[I_drive_sought_ind][I_di_sought_ind]

print('I_drive_sought = {:2.2f}uA, I_drive_sought_ind = {:d}\nI_di_sought = {:2.2f}uA, I_di_sought_ind = {:d}\nrate_obtained = {:3.2f}GHz'.format(I_drive_sought*1e6,I_drive_sought_ind,I_di_sought*1e6,I_di_sought_ind,rate_obtained*1e-9))
