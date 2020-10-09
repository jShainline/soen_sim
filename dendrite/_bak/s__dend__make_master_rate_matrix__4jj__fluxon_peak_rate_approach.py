#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_dend_rate_array, plot_dend_time_traces
from _functions import cv, save_session_data, read_wr_data
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
d1 = 0.1 # uA
d2 = 1 # uA
# cv(start1,stop1,d1,start2,stop2,d2)

#                 I_de = 70                 I_de = 72                 I_de = 74                 I_de = 76                 I_de = 78                 I_de = 80
I_drive_array = [[cv(18.3,18.9,d1,19,30,d2),cv(17.1,17.9,d1,18,30,d2),cv(15.8,15.9,d1,16,30,d2),cv(14.5,14.9,d1,15,30,d2),cv(13.1,13.9,d1,14,30,d2),cv(11.6,11.9,d1,12,30,d2)], # L_left = 7 pH
                 [cv(16.8,16.9,d1,17,30,d2),cv(15.5,15.9,d1,16,30,d2),cv(14.1,14.9,d1,15,30,d2),cv(12.7,12.9,d1,13,30,d2),cv(11.3,11.9,d1,12,30,d2),cv( 9.8, 9.9,d1,10,30,d2)], # L_left = 8 pH                 
                 [cv(15.2,15.9,d1,16,30,d2),cv(13.8,13.9,d1,14,30,d2),cv(12.5,12.9,d1,13,30,d2),cv(11.0,10.9,d1,11,30,d2),cv( 9.5, 9.9,d1,10,30,d2),cv( 7.9, 7.9,d1, 8,30,d2)], # L_left = 9 pH                 
                 [cv(13.6,13.9,d1,14,30,d2),cv(12.2,12.9,d1,13,30,d2),cv(10.8,10.9,d1,11,30,d2),cv( 9.3, 9.9,d1,10,30,d2),cv( 7.8, 7.9,d1, 8,30,d2),cv( 6.1, 6.9,d1, 7,30,d2)], # L_left = 10 pH                 
                 [cv(12.0,11.9,d1,12,30,d2),cv(10.6,10.9,d1,11,30,d2),cv( 9.1, 9.9,d1,10,30,d2),cv( 7.6, 7.9,d1, 8,30,d2),cv( 6.0, 5.9,d1, 6,30,d2),cv( 4.3, 4.9,d1, 5,30,d2)], # L_left = 11 pH                 
                 [cv(10.4,10.9,d1,11,30,d2),cv( 8.9, 8.9,d1, 9,30,d2),cv( 7.4, 7.9,d1, 8,30,d2),cv( 5.8, 5.9,d1, 6,30,d2),cv( 4.2, 4.9,d1, 5,30,d2),cv( 2.4, 2.9,d1, 3,30,d2)], # L_left = 12 pH                  
                 [cv( 8.8, 8.9,d1, 9,30,d2),cv( 7.3, 7.9,d1, 8,30,d2),cv( 5.7, 5.9,d1, 6,30,d2),cv( 4.1, 4.9,d1, 5,30,d2),cv( 2.4, 2.9,d1, 3,30,d2),cv( 0.5, 0.9,d1, 1,30,d2)]] # L_left = 13 pH # units of uA

t_sim = 100 # ns
inductance_conversion = 1e12

#%%

# loop to create rate files for all cases
plot_time_traces = True
plot_rate_arrays = True
min_peak_height = 182e-6 # units of volts
min_peak_distance = 175 # units of samples

min_ht = 1000
max_ht = -1000
min_dist = 1000
max_dist = -1000

for pp in [4]: # range(num_L): #  
    
    for qq in [4]: # range(num_I_de): # 
                
        # load wr data, find peaks, find rates
        I_drive_list = I_drive_array[pp][qq] # [18.6,19,20,21,22,23,24,25,26,27,28,29,30]#np.arange(19,30+dI,dI)
        # t_sim_list = [60,70,50,40,35,32,32,32,32,32,32,32,32]

        j_di_ifi_array = []
        j_di_rate_array = []
        
        I_di_array = []
        
        num_drives = len(I_drive_list)
        for ii in range(num_drives):
            
            print('pp = {:d} of {:d}; qq = {:d} of {:d}; ii = {:d} of {:d}'.format(pp+1,num_L,qq+1,num_I_de,ii+1,num_drives))
            
            directory = 'wrspice_data/constant_drive/from_saeed/4jj/'            
            # directory = 'wrspice_data/fitting_data'            
            # dend_cnst_drv_Llft08.00pH_Lrgt22.00pH_Ide78.00uA_Idrv30.00uA_Ldi0077.50nH_taudi0775ms_dt00.1ps.dat
            I_drive = I_drive_list[ii]
            file_name = 'dend_cnst_drv_Llft{:05.2f}pH_Lrgt{:05.2f}pH_Ide{:05.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0775ms_dt00.1ps.dat'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq],I_drive)
            data_dict = read_wr_data(directory+'/'+file_name)
            
            # find peaks for each jj
            time_vec = data_dict['time']
            j_di = data_dict['v(4)'] # j_di = data_dict['v(5)']
            
            initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
            final_ind = (np.abs(time_vec-np.max(time_vec))).argmin()
        
            time_vec = time_vec[initial_ind:final_ind]
            j_di = j_di[initial_ind:final_ind]
          
            # j_di_peaks, _ = find_peaks(j_di, height = jph[pp][qq][ii])
            j_di_peaks, _ = find_peaks(j_di, height = min_peak_height, distance = min_peak_distance)
            if len(j_di_peaks) > 1:
                print('min(diff(j_di_peaks)) = {}'.format(np.min(np.diff(j_di_peaks))))
                print('max(diff(j_di_peaks)) = {}'.format(np.max(np.diff(j_di_peaks))))
                if np.min(np.diff(j_di_peaks)) < min_dist:
                    min_dist = np.min(np.diff(j_di_peaks))
                if np.max(np.diff(j_di_peaks)) > max_dist:
                    max_dist = np.max(np.diff(j_di_peaks))
                if np.min(j_di[j_di_peaks]) < min_ht:
                    min_ht = np.min(j_di[j_di_peaks])
                if np.max(j_di[j_di_peaks]) > max_ht:
                    max_ht = np.max(j_di[j_di_peaks])
            else:
                print('no peaks')
            
            # find inter-fluxon intervals and fluxon generation rates for each JJ
            I_di = data_dict['i(L4)']
            I_di = I_di[initial_ind:final_ind]
        
            if plot_time_traces == True:
                plot_dend_time_traces(time_vec,j_di,j_di_peaks,min_peak_height,I_di,file_name)

            j_di_ifi = np.diff(time_vec[j_di_peaks])
            j_di_rate = 1/j_di_ifi
            
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
                        
            tn = np.max([len(j_di_rate_array[ii]),1])
            temp_rate_vec = np.zeros([tn])
            temp_I_di_vec = np.zeros([tn])
            for jj in range(len(j_di_rate_array[ii])):
                temp_rate_vec[jj] = 1e-6*j_di_rate_array[ii][jj] # master_rate_array has units of fluxons per microsecond
                temp_I_di_vec[jj] = 1e6*( I_di_array[ii][jj]+I_di_array[ii][jj+1] )/2 # this makes I_di_array__scaled have the same dimensions as j_di_rate_array and units of uA
        
            master_rate_array.append([])
            master_rate_array[ii] = np.insert(np.append(temp_rate_vec,0),0,0) # these zeros added so that a current that rounds to I_di + I_di_pad will give zero rate
            # master_rate_array[ii].insert(0,0) # this zero is added so that a current that rounds to I_di - I_di_pad will give zero rate
            I_di_array__scaled.append([])
            I_di_array__scaled[ii] = np.insert(np.append(temp_I_di_vec,np.max(temp_I_di_vec)+I_di_pad),0,0) # this additional I_di + I_di_pad is included so that a current that rounds to I_di + I_di_pad will give zero rate
            # I_di_array__scaled[ii].insert(0,np.min(temp_I_di_vec)-I_di_pad) # this additional I_di - I_di_pad is included so that a current that rounds to I_di - I_di_pad will give zero rate

        # plot the rate array  
        if plot_rate_arrays == True:
            plot_dend_rate_array(I_di_array = I_di_array__scaled, I_drive_list = I_drive_list, master_rate_array = master_rate_array)
        
        # fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
        # fig.suptitle('master _ de _ rates') 
        # # plt.title('$I_de = $ {} $\mu$A'.format(I_sy))
        # for ii in range(num_drives):
        #     ax.plot(I_di_array__scaled[ii][:],master_rate_array[ii][:]*1e-3, '-', label = 'I_drive = {}'.format(I_drive_list[ii]))    
        # ax.set_xlabel(r'$I_{di}$ [$\mu$A]')
        # ax.set_ylabel(r'$r_{j_{di}}$ [kilofluxons per $\mu$s]')
        # # ax.legend()
        # plt.show()
        
        # save data
        save_string = 'master__dnd_2jj__rate_array__Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq])
        data_array = dict()
        data_array['rate_array'] = master_rate_array
        data_array['I_drive_list'] = I_drive_list
        data_array['influx_list'] = influx_list
        data_array['I_di_array'] = I_di_array__scaled
        print('\n\nsaving session data ...')
        save_session_data(data_array,save_string)
        save_session_data(data_array,save_string+'.soen',False)
        
print('\n\nmin_ht = {}'.format(min_ht))
print('max_ht = {}'.format(max_ht))
print('min_dist = {}'.format(min_dist))
print('max_dist = {}'.format(max_dist))

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
