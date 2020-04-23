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
I_drive_vec = [8.9,9,10,11,12,13,14,15,16,17,18,19,20]#us
t_sim_vec = [40,60,37,30,41,30,27,25,24,23,22,22,22]#ns
I_sy = 33
data_file_list = []
num_files = len(I_drive_vec)
for ii in range(num_files):
    data_file_list.append('syn_cnst_drv_Isy{:5.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0077.5ms_tsim{:04.0f}ns_dt00.1ps.dat'.format(I_sy,I_drive_vec[ii],t_sim_vec[ii]))

master_data__sf__t_fq = []
master_data__sf__r_fq = []
master_data__jtl__t_fq = []
master_data__jtl__r_fq = []
master_data__si__t_fq = []
master_data__si__r_fq = []
max_I_si = 0
I_si_list = []
for ii in range(num_files):
    
    print('ii = {:d} of {:d}'.format(ii+1,num_files))
    
    directory = 'wrspice_data/fitting_data'
    file_name = data_file_list[ii]
    data_dict = read_wr_data(directory+'/'+file_name)
    
    # find peaks for each jj
    time_vec = data_dict['time']
    j_sf = data_dict['v(3)']
    j_jtl = data_dict['v(4)']
    j_si = data_dict['v(5)']
    
    initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
    final_ind = (np.abs(time_vec-t_sim_vec[ii]*1e-9)).argmin()

    time_vec = time_vec[initial_ind:final_ind]
    j_sf = j_sf[initial_ind:final_ind]
    j_jtl = j_jtl[initial_ind:final_ind]
    j_si = j_si[initial_ind:final_ind]
  
    j_sf_peaks, _ = find_peaks(j_sf, height = 200e-6)
    j_jtl_peaks, _ = find_peaks(j_jtl, height = 200e-6)
    j_si_peaks, _ = find_peaks(j_si, height = 200e-6)

    fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)
    fig.suptitle(file_name)   
    ax[0].plot(time_vec*1e9,j_sf*1e3, '-', label = '$J_{sf}$')             
    ax[0].plot(time_vec[j_sf_peaks]*1e9,j_sf[j_sf_peaks]*1e3, 'x')
    ax[0].set_xlabel(r'Time [ns]')
    ax[0].set_ylabel(r'Voltage [mV]')
    ax[0].legend()
    ax[1].plot(time_vec*1e9,j_jtl*1e3, '-', label = '$J_{jtl}$')             
    ax[1].plot(time_vec[j_jtl_peaks]*1e9,j_jtl[j_jtl_peaks]*1e3, 'x')
    ax[1].set_xlabel(r'Time [ns]')
    ax[1].set_ylabel(r'Voltage [mV]')
    ax[1].legend()
    ax[2].plot(time_vec*1e9,j_si*1e3, '-', label = '$J_{si}$')             
    ax[2].plot(time_vec[j_si_peaks]*1e9,j_si[j_si_peaks]*1e3, 'x')
    ax[2].set_xlabel(r'Time [ns]')
    ax[2].set_ylabel(r'Voltage [mV]')
    ax[2].legend()
    plt.show()
    
    # find inter-fluxon intervals and fluxon generation rates for each JJ
    I_si = data_dict['L3#branch']
    I_si = I_si[initial_ind:final_ind]
    I_drive = data_dict['L0#branch']
    I_drive = I_drive[initial_ind:final_ind]    
    max_I_si = np.max([np.max(I_si),max_I_si])
    
    # I_dr1 = Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,Iflux_vec[jj],I_b,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
    
    j_sf_ifi = np.diff(time_vec[j_sf_peaks])
    j_jtl_ifi = np.diff(time_vec[j_jtl_peaks])
    j_si_ifi = np.diff(time_vec[j_si_peaks])
    
    j_sf_rate = 1/j_sf_ifi
    j_jtl_rate = 1/j_jtl_ifi
    j_si_rate = 1/j_si_ifi
    
    master_data__sf__t_fq.append(j_sf_ifi)
    master_data__sf__r_fq.append(j_sf_rate)
    master_data__jtl__t_fq.append(j_jtl_ifi)
    master_data__jtl__r_fq.append(j_jtl_rate)
    master_data__si__t_fq.append(j_si_ifi)
    master_data__si__r_fq.append(j_si_rate)
    
    I_si_list.append(I_si[j_si_peaks])
    
#%% assemble data
longest_rate_vec = 0
fastest_rate = 0
for ii in range(num_files):
    if len(master_data__si__r_fq[ii]) > longest_rate_vec:
        longest_rate_vec = len(master_data__si__r_fq[ii])
    fastest_rate = np.max([np.max(master_data__si__r_fq[ii]),fastest_rate])        

master_rate_matrix = np.zeros([num_files,longest_rate_vec])
for ii in range(num_files):
    tn = len(master_data__si__r_fq[ii])
    for jj in range(tn):
        master_rate_matrix[ii,jj] = master_data__si__r_fq[ii][jj]

fig, ax = plt.subplots(1,1)
rates = ax.imshow(np.transpose(master_rate_matrix[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[I_drive_vec[0],I_drive_vec[-1],0,max_I_si*1e6], aspect = 'auto', origin = 'lower')
cbar = fig.colorbar(rates, extend='both')
cbar.minorticks_on()     
fig.suptitle('$r_{tot}$ versus $I_{drive}$ and $I_{si}$')
# plt.title(title_string)
ax.set_xlabel(r'$I_{drive}$ [$\mu$A]')
ax.set_ylabel(r'$I_{si}$ [$\mu$A]')   
plt.show()      
# fig.savefig('figures/'+save_str+'__log.png')

#%% save data
save_string = 'master_rate_matrix__syn'
data_array = dict()
data_array['master_rate_matrix'] = master_rate_matrix
data_array['I_drive_vec'] = [x*1e-6 for x in I_drive_vec]
data_array['I_si_list'] = I_si_list
print('\n\nsaving session data ...')
save_session_data(data_array,save_string)

#%% load test
data_array_imported = load_session_data('session_data__master_rate_matrix__syn__2020-04-23_16-50-17.dat')
I_si_list__imported = data_array_imported['I_si_list']
I_drive_vec__imported = data_array_imported['I_drive_vec']
master_rate_matrix__imported = data_array_imported['master_rate_matrix']

I_drive_sought = 13.45e-6
I_drive_sought_ind = (np.abs(np.asarray(I_drive_vec__imported)-I_drive_sought)).argmin()
I_si_sought = 14.552e-6
I_si_sought_ind = (np.abs(np.asarray(I_si_list__imported[I_drive_sought_ind])-I_si_sought)).argmin()
rate_obtained = master_rate_matrix__imported[I_drive_sought_ind][I_si_sought_ind]

print('I_drive_sought = {:2.2f}uA, I_drive_sought_ind = {:d}\nI_si_sought = {:2.2f}uA, I_si_sought_ind = {:d}\nrate_obtained = {:3.2f}GHz'.format(I_drive_sought*1e6,I_drive_sought_ind,I_si_sought*1e6,I_si_sought_ind,rate_obtained*1e-9))
