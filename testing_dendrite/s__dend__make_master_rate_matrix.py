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

#%% inputs

# DR loop inductances (in all cases, left side has additional 10pH through which flux is coupled (M = k*sqrt(L1*L2); in this case k = 1, L1 = 200pH, L2 = 10pH))
dL = 1 # pH
L_left_list = np.arange(7,13+dL,dL) # pH
L_right_list = np.flip(np.arange(17,23+dL,dL)) # pH
num_L = len(L_right_list)

# dendritic firing junction bias current
dI = 2 # uA
I_de_list = np.arange(70,80+dI,dI) # uA
num_I_de = len(I_de_list)

# drive current
dI = 1 # uA
dI_2 = 0.1 # uA
I_drive_array = [[np.concatenate((np.arange(18.6,18.9+dI,dI),np.arange(19,30+dI,dI))),np.arange(18,30+dI,dI),np.arange(16,30+dI,dI),np.arange(15,30+dI,dI),np.arange(14,30+dI,dI),np.arange(12,30+dI,dI)], # L_left = 7 pH
                 [np.arange(17,30+dI,dI),np.arange(16,30+dI,dI),np.arange(15,30+dI,dI),np.arange(13,30+dI,dI),np.arange(12,30+dI,dI),np.arange(10,30+dI,dI)], # L_left = 8 pH
                 [np.arange(16,30+dI,dI),np.arange(14,30+dI,dI),np.arange(13,30+dI,dI),np.arange(11,30+dI,dI),np.arange(10,30+dI,dI),np.arange(8,30+dI,dI)], # L_left = 9 pH
                 [np.arange(14,30+dI,dI),np.arange(13,30+dI,dI),np.arange(11,30+dI,dI),np.arange(10,30+dI,dI),np.arange(8,30+dI,dI),np.arange(7,30+dI,dI)], # L_left = 10 pH
                 [np.arange(12,30+dI,dI),np.arange(11,30+dI,dI),np.arange(10,30+dI,dI),np.arange(8,30+dI,dI),np.arange(6,30+dI,dI),np.arange(5,30+dI,dI)], # L_left = 11 pH
                 [np.arange(11,30+dI,dI),np.arange(9,30+dI,dI),np.arange(8,30+dI,dI),np.arange(6,30+dI,dI),np.arange(5,30+dI,dI),np.arange(3,30+dI,dI)], # L_left = 12 pH 
                 [np.arange(9,30+dI,dI),np.arange(8,30+dI,dI),np.arange(6,30+dI,dI),np.arange(5,30+dI,dI),np.arange(3,30+dI,dI),np.arange(1,30+dI,dI)]] # L_left = 13 pH # units of uA

t_sim = 100 # ns
inductance_conversion = 1e12

#%%

# loop to create rate files for all cases
for pp in [6]: # in range(num_L):
    print('pp = {:d} of {:d}'.format(pp+1,num_L))
    
    for qq in [5]: # in range(num_I_de):
        print('qq = {:d} of {:d}'.format(qq+1,num_I_de))
                
        # load wr data, find peaks, find rates
        I_drive_list = I_drive_array[pp][qq] # [18.6,19,20,21,22,23,24,25,26,27,28,29,30]#np.arange(19,30+dI,dI)
        # t_sim_list = [60,70,50,40,35,32,32,32,32,32,32,32,32]

        # j_df1_ifi_array = []
        # j_df1_rate_array = []
        # j_df2_ifi_array = []
        # j_df2_rate_array = []
        # j_jtl_ifi_array = []
        # j_jtl_rate_array = []
        j_di_ifi_array = []
        j_di_rate_array = []
        
        I_di_array = []
        
        num_drives = len(I_drive_list)
        for ii in range(num_drives):
            
            print('ii = {:d} of {:d}'.format(ii+1,num_drives))
            
            directory = 'wrspice_data/constant_drive/from_saeed'            
            # directory = 'wrspice_data/fitting_data'            
            # dend_cnst_drv_Llft08.00pH_Lrgt22.00pH_Ide78.00uA_Idrv30.00uA_Ldi0077.50nH_taudi0775ms_dt00.1ps.dat
            I_drive = I_drive_list[ii]
            file_name = 'dend_cnst_drv_Llft{:05.2f}pH_Lrgt{:05.2f}pH_Ide{:05.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0775ms_dt00.1ps.dat'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq],I_drive)
            data_dict = read_wr_data(directory+'/'+file_name)
            
            # find peaks for each jj
            time_vec = data_dict['time']
            # j_df = data_dict['v(3)']
            # j_jtl = data_dict['v(4)']
            j_di = data_dict['v(4)'] # j_di = data_dict['v(5)']
            
            initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
            final_ind = (np.abs(time_vec-t_sim*1e-9)).argmin()
        
            time_vec = time_vec[initial_ind:final_ind]
            # j_df = j_df[initial_ind:final_ind]
            # j_jtl = j_jtl[initial_ind:final_ind]
            j_di = j_di[initial_ind:final_ind]
          
            # j_df1_peaks, _ = find_peaks(j_df, height = [187.8e-6,300e-6])
            # j_df2_peaks, _ = find_peaks(j_df, height = [140e-6,187.6e-6])
            # j_jtl_peaks, _ = find_peaks(j_jtl, height = 200e-6)
            j_di_peaks, _ = find_peaks(j_di, height = 200e-6)
        
            fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
            fig.suptitle(file_name)   
            # ax[0].plot(time_vec*1e9,j_df*1e3, '-', label = '$J_{df}$')             
            # ax[0].plot(time_vec[j_df1_peaks]*1e9,j_df[j_df1_peaks]*1e3, 'x')
            # ax[0].plot(time_vec[j_df2_peaks]*1e9,j_df[j_df2_peaks]*1e3, 'x')
            # ax[0].set_xlabel(r'Time [ns]')
            # ax[0].set_ylabel(r'Voltage [mV]')
            # ax[0].legend()
            # ax[1].plot(time_vec*1e9,j_jtl*1e3, '-', label = '$J_{jtl}$')             
            # ax[1].plot(time_vec[j_jtl_peaks]*1e9,j_jtl[j_jtl_peaks]*1e3, 'x')
            # ax[1].set_xlabel(r'Time [ns]')
            # ax[1].set_ylabel(r'Voltage [mV]')
            # ax[1].legend()
            ax.plot(time_vec*1e9,j_di*1e3, '-', label = '$J_{di}$')             
            ax.plot(time_vec[j_di_peaks]*1e9,j_di[j_di_peaks]*1e3, 'x')
            ax.set_xlabel(r'Time [ns]')
            ax.set_ylabel(r'Voltage [mV]')
            ax.legend()
            plt.show()
            
            # find inter-fluxon intervals and fluxon generation rates for each JJ
            I_di = data_dict['i(L4)']
            I_di = I_di[initial_ind:final_ind]
            # I_drive = I_drive_listdata_dict['L4#branch']
            # I_drive = I_drive[initial_ind:final_ind]   
            
            # I_dr1 = Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,Iflux_vec[jj],I_b,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
            
            # j_df1_ifi = np.diff(time_vec[j_df1_peaks])
            # j_df2_ifi = np.diff(time_vec[j_df2_peaks])
            # j_jtl_ifi = np.diff(time_vec[j_jtl_peaks])
            j_di_ifi = np.diff(time_vec[j_di_peaks])
            
            # j_df1_rate = 1/j_df1_ifi
            # j_df2_rate = 1/j_df2_ifi
            # j_jtl_rate = 1/j_jtl_ifi
            j_di_rate = 1/j_di_ifi
            
            # j_df1_ifi_array.append(j_df1_ifi)
            # j_df1_rate_array.append(j_df1_rate)
            # j_df2_ifi_array.append(j_df2_ifi)
            # j_df2_rate_array.append(j_df2_rate)
            # j_jtl_ifi_array.append(j_jtl_ifi)
            # j_jtl_rate_array.append(j_jtl_rate)
            j_di_ifi_array.append(j_di_ifi)
            j_di_rate_array.append(j_di_rate)
            
            I_di_array.append(I_di[j_di_peaks])
 
        # convert current drive to flux
        influx_list = []
        M = inductance_conversion*np.sqrt(200e-12*10e-12)
        for ii in range(len(I_drive_list)):
            influx_list.append(M*I_drive_list[ii])
            
        # assemble data and change units
        I_di_pad = 10e-9 # amount above the observed max of Isi that the simulation will allow before giving a zero rate
        
        master_rate_array = []
        I_di_array__scaled = []
            
        for ii in range(num_drives):
                        
            temp_rate_vec = np.zeros([len(j_di_rate_array[ii])])
            temp_I_di_vec = np.zeros([len(j_di_rate_array[ii])])
            for jj in range(len(j_di_rate_array[ii])):
                temp_rate_vec[jj] = 1e-6*j_di_rate_array[ii][jj] # master_rate_array has units of fluxons per microsecond
                temp_I_di_vec[jj] = 1e6*( I_di_array[ii][jj]+I_di_array[ii][jj+1] )/2 # this makes I_si_array__scaled have the same dimensions as j_si_rate_array and units of uA
        
            master_rate_array.append([])
            master_rate_array[ii] = np.append(temp_rate_vec,0) # this zero is added so that a current that rounds to I_si + I_si_pad will give zero rate
            I_di_array__scaled.append([])
            I_di_array__scaled[ii] = np.append(temp_I_di_vec,np.max(temp_I_di_vec)+I_di_pad) # this additional I_di + I_di_pad is included so that a current that rounds to I_di + I_di_pad will give zero rate

        # plot the rate vectors
        fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
        fig.suptitle('master _ de _ rates') 
        # plt.title('$I_de = $ {} $\mu$A'.format(I_sy))
        for ii in range(num_drives):
            ax.plot(I_di_array__scaled[ii][:],master_rate_array[ii][:]*1e-3, '-', label = 'I_drive = {}'.format(I_drive_list[ii]))    
        ax.set_xlabel(r'$I_{si}$ [$\mu$A]')
        ax.set_ylabel(r'$r_{j_{si}}$ [kilofluxons per $\mu$s]')
        # ax.legend()
        plt.show()
        
        # save data
        save_string = 'master__dnd__rate_array__Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq])
        data_array = dict()
        data_array['rate_array'] = master_rate_array
        data_array['I_drive_list'] = I_drive_list
        data_array['influx_list'] = influx_list
        data_array['I_di_array'] = I_di_array__scaled
        print('\n\nsaving session data ...')
        save_session_data(data_array,save_string)
        save_session_data(data_array,save_string+'.soen',False)


#%% assemble data
# longest_rate_vec = 0
# fastest_rate = 0
# for ii in range(num_drives):
#     if len(master_data__di__r_fq[ii]) > longest_rate_vec:
#         longest_rate_vec = len(master_data__di__r_fq[ii])
#     fastest_rate = np.max([np.max(master_data__di__r_fq[ii]),fastest_rate])        

# master_rate_matrix = np.zeros([num_drives,longest_rate_vec])
# for ii in range(num_drives):
#     tn = len(master_data__di__r_fq[ii])
#     for jj in range(tn):
#         master_rate_matrix[ii,jj] = master_data__di__r_fq[ii][jj]

#%% color plot
# fig, ax = plt.subplots(1,1)
# rates = ax.imshow(np.transpose(master_rate_matrix[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[I_drive_list[0],I_drive_list[-1],0,max_I_di*1e6], aspect = 'auto', origin = 'lower')
# cbar = fig.colorbar(rates, extend='both')
# cbar.minorticks_on()     
# fig.suptitle('$r_{tot}$ versus $I_{drive}$ and $I_{di}$')
# # plt.title(title_string)
# ax.set_xlabel(r'$I_{drive}$ [$\mu$A]')
# ax.set_ylabel(r'$I_{di}$ [$\mu$A]')   
# plt.show()      
# fig.savefig('figures/'+save_str+'__log.png')

#%% flux/current comparison
# I_drive = 21.2
# _ind1 = (np.abs(np.asarray(I_drive_list)-I_drive)).argmin()

# influx = I_drive*inductance_conversion*np.sqrt(200e-12*10e-12)
# ind1 = (np.abs(np.asarray(influx_list)-influx)).argmin()

# print('_ind1 = {}; ind1 = {}'.format(_ind1,ind1))           


#%% load test
# data_array_imported = load_session_data('session_data__master_rate_matrix__2020-04-17_13-11-03.dat')
# I_di_list__imported = data_array_imported['I_di_list']
# I_drive_vec__imported = data_array_imported['I_drive_vec']
# master_rate_matrix__imported = data_array_imported['master_rate_matrix']

# I_drive_sought = 23.45e-6
# I_drive_sought_ind = (np.abs(np.asarray(I_drive_vec__imported)-I_drive_sought)).argmin()
# I_di_sought = 14.552e-6
# I_di_sought_ind = (np.abs(np.asarray(I_di_list__imported[I_drive_sought_ind])-I_di_sought)).argmin()
# rate_obtained = master_rate_matrix__imported[I_drive_sought_ind][I_di_sought_ind]

# print('I_drive_sought = {:2.2f}uA, I_drive_sought_ind = {:d}\nI_di_sought = {:2.2f}uA, I_di_sought_ind = {:d}\nrate_obtained = {:3.2f}GHz'.format(I_drive_sought*1e6,I_drive_sought_ind,I_di_sought*1e6,I_di_sought_ind,rate_obtained*1e-9))
