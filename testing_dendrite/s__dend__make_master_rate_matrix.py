#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_dend_rate_array, plot_dend_time_traces
from _functions import cv, save_session_data, read_wr_data
from util import physical_constants

import pickle

p = physical_constants()

plt.close('all')

#%% set case

num_jjs = 4 # 2 or 4

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

if num_jjs == 2:

    #                 I_de = 70                  I_de = 72                    I_de = 74                   I_de = 76                   I_de = 78                  I_de = 80
    I_drive_array = [[cv(15.1,15.9,d1,16.0,30,d2),cv(13.8,13.9,d1,14.0,30,d2),cv(12.3,12.9,d1,13.0,30,d2),cv(10.8,10.9,d1,11.0,30,d2),cv(00.0,0.9,d1,10.0,30,d2),cv(0.0,0.9,d1,1.0,30,d2)], # L_left = 7
                     [cv(13.6,13.9,d1,14.0,30,d2),cv(12.2,12.9,d1,13.0,30,d2),cv(10.7,10.9,d1,11.0,30,d2),cv(09.1,09.9,d1,10.0,30,d2),cv(07.2,07.9,d1,8.0,30,d2),cv(0.0,0.9,d1,1.0,30,d2)], # L_left = 8
                     [cv(11.9,11.1,d1,12.0,30,d2),cv(10.6,10.9,d1,11.0,30,d2),cv(09.9,09.0,d1,09.0,30,d2),cv(07.4,07.9,d1,08.0,30,d2),cv(05.5,05.9,d1,6.0,30,d2),cv(0.9,0.0,d1,1.0,30,d2)], # L_left = 9
                     [cv(10.4,10.9,d1,11.0,30,d2),cv(08.9,08.9,d1,09.0,30,d2),cv(07.4,07.9,d1,08.0,30,d2),cv(05.7,05.9,d1,06.0,30,d2),cv(03.7,03.9,d1,4.0,30,d2),cv(0.1,0.9,d1,1.0,30,d2)], # L_left = 10
                     [cv(08.9,08.9,d1,09.0,30,d2),cv(07.3,07.9,d1,08.0,30,d2),cv(05.7,05.9,d1,06.0,30,d2),cv(03.0,03.9,d1,04.0,30,d2),cv(01.0,01.0,d1,2.0,30,d2),cv(0.0,0.9,d1,1.0,30,d2)], # L_left = 11
                     [cv(07.3,07.9,d1,08.0,30,d2),cv(05.7,05.9,d1,06.0,30,d2),cv(04.1,04.9,d1,05.0,30,d2),cv(02.3,02.9,d1,03.0,30,d2),cv(00.2,00.9,d1,1.0,30,d2),cv(0.0,0.9,d1,1.0,30,d2)], # L_left = 12
                     [cv(05.7,05.9,d1,06.0,30,d2),cv(04.1,04.9,d1,05.0,30,d2),cv(02.4,02.9,d1,03.0,30,d2),cv(00.6,00.9,d1,01.0,30,d2),cv(00.0,00.9,d1,1.0,30,d2),cv(0.0,0.9,d1,1.0,30,d2)]] # L_left = 13
    
elif num_jjs == 4:
        
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
DI_loop_inductance = 77.5e-9

plot_time_traces = False
plot_rate_arrays = True
min_peak_height = 100e-6 # 182e-6 # units of volts
min_peak_distance = 10 # 175 # units of samples

downsample_factor = 500 # dt = 0.1ps
window_size = 51 # samples/time steps for savitzky-golay filter (applied after downsample)

#%%

# loop to create rate files for all cases
for pp in range(num_L): # [3]: # 
    
    for qq in range(num_I_de): # [3]: #
                
        # load wr data, find peaks, find rates
        I_drive_list = I_drive_array[pp][qq]
        j_di_rate_array = []
        I_di_array = []
        
        num_drives = len(I_drive_list)
        for ii in range(num_drives):
            
            print('pp = {:d} of {:d}; qq = {:d} of {:d}; ii = {:d} of {:d}'.format(pp+1,num_L,qq+1,num_I_de,ii+1,num_drives))
            
            I_drive = I_drive_list[ii]
            if num_jjs == 2:            
                directory = 'wrspice_data/constant_drive/from_saeed/2jj/'            
                file_name = 'dend_cnst_drv_1jj_Llft{:05.2f}pH_Lrgt{:05.2f}pH_Ide{:05.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0775ms_dt00.1ps.dat'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq],I_drive)
                j_di_str = 'v(2)'
                I_di_str = 'i(L2)'
            elif num_jjs == 4:
                directory = 'wrspice_data/constant_drive/from_saeed/4jj/'
                file_name = 'dend_cnst_drv_Llft{:05.2f}pH_Lrgt{:05.2f}pH_Ide{:05.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0775ms_dt00.1ps.dat'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq],I_drive) 
                j_di_str = 'v(4)'
                I_di_str = 'i(L4)'
            data_dict = read_wr_data(directory+'/'+file_name)
            
            # find peaks for each jj
            time_vec = data_dict['time']
            j_di = data_dict[j_di_str] # j_di = data_dict['v(5)']
                      
            # j_di_peaks, _ = find_peaks(j_di, height = jph[pp][qq][ii])
            j_di_peaks, _ = find_peaks(j_di, height = min_peak_height, distance = min_peak_distance)
            
            initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
            final_ind = j_di_peaks[-1]*2*downsample_factor     
            time_vec = time_vec[initial_ind:final_ind]
            j_di = j_di[initial_ind:final_ind]

            # find fluxon generation rate
            I_di = data_dict[I_di_str]
            I_di = I_di[initial_ind:final_ind]           
            
            # downsample
            index_vec = np.arange(0,len(time_vec),downsample_factor)
            I_di__avg = I_di[index_vec]
            time_vec__avg = time_vec[index_vec]
            
            # apply smoothing filter
            I_di__avg = savgol_filter(I_di__avg, window_size, 3) # polynomial order 3
            time_vec__avg = savgol_filter(time_vec__avg, np.min([window_size,len(time_vec__avg)]), 3) # polynomial order 3

            if plot_time_traces == True:
                plot_dend_time_traces(time_vec,j_di,j_di_peaks,min_peak_height,I_di,file_name)

            j_di_rate = (np.diff(I_di__avg)/np.diff(time_vec__avg))/(p['Phi0']/DI_loop_inductance)
            for nn in range(len(j_di_rate)):
                if j_di_rate[nn] < 0:
                    j_di_rate[nn] = 0
            j_di_rate_array.append(j_di_rate)
            I_di_array.append(I_di__avg)
 
        # convert current drive to flux
        influx_list = []
        M = inductance_conversion*np.sqrt(200e-12*10e-12)
        I_drive__pad = 10e-9
        I_drive_list__pad = np.insert(I_drive_list,0,I_drive_list[0]-I_drive__pad) # adding extra zero so flux that rounds below minimum value gives zero rate
        for ii in range(len(I_drive_list__pad)):
            influx_list.append(M*I_drive_list__pad[ii])
            
        # assemble data and change units
        I_di_pad = 10e-9 # amount above the observed max of Isi that the simulation will allow before giving a zero rate
        
        master_rate_array = [np.array([0,0])] # initializing with these zeros so influx that rounds below minimum value gives zero rate
        I_di_array__scaled = [np.array([0,I_di_pad])] # initializing with these zeros so influx that rounds below minimum value gives zero rate     
            
        for ii in range(num_drives):
                        
            tn = np.max([len(j_di_rate_array[ii]),1])
            temp_rate_vec = np.zeros([tn])
            temp_I_di_vec = np.zeros([tn])
            for jj in range(len(j_di_rate_array[ii])):
                temp_rate_vec[jj] = 1e-6*j_di_rate_array[ii][jj] # master_rate_array has units of fluxons per microsecond
                temp_I_di_vec[jj] = 1e6*( I_di_array[ii][jj]+I_di_array[ii][jj+1] )/2 # this makes I_di_array__scaled have the same dimensions as j_di_rate_array and units of uA
        
            master_rate_array.append([])
            # master_rate_array[ii] = np.insert(np.append(temp_rate_vec,0),0,0) # this zero added so that a current that rounds to I_di + I_di_pad will give zero rate
            master_rate_array[ii+1] = np.append(temp_rate_vec,0) # this zero added so that a current that rounds to I_di + I_di_pad will give zero rate
            I_di_array__scaled.append([])
            # I_di_array__scaled[ii] = np.insert(np.append(temp_I_di_vec,np.max(temp_I_di_vec)+I_di_pad),0,0) # this additional I_di + I_di_pad is included so that a current that rounds to I_di + I_di_pad will give zero rate
            I_di_array__scaled[ii+1] = np.append(temp_I_di_vec,np.max(temp_I_di_vec)+I_di_pad) # this additional I_di + I_di_pad is included so that a current that rounds to I_di + I_di_pad will give zero rate
            
        # plot the rate array  
        if plot_rate_arrays == True:
            plot_dend_rate_array(I_di_array = I_di_array__scaled, I_drive_list = I_drive_list, influx_list = influx_list, master_rate_array = master_rate_array)    
        
        # save data
        save_string = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}'.format(num_jjs,L_left_list[pp]+10,L_right_list[pp],I_de_list[qq])
        data_array = dict()
        data_array['rate_array'] = master_rate_array
        data_array['I_drive_list'] = I_drive_list
        data_array['influx_list'] = influx_list
        data_array['I_di_array'] = I_di_array__scaled
        print('\n\nsaving session data ...\n\n')
        # save_session_data(data_array,save_string)
        save_session_data(data_array,save_string+'.soen',False)


#%% just plot
if 1 == 2:
    
    L_left = 10
    L_right = 20
    I_de = 72
    
    file_name = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}.soen'.format(num_jjs,L_left+10,L_right,I_de)
    plot_dend_rate_array(file_name = file_name)
            
    # for pp in range(num_L):
    #     for qq in range(num_I_de):
            
    #         file_name = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}.soen'.format(numjj,L_left_list[pp]+10,L_right_list[pp],I_de_list[qq])
    #         plot_dend_rate_array(file_name = file_name)

        
#%% debugging
if 1 == 2:

    L_left = 10
    L_right = 20
    I_de = 72
    file_name = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}.soen'.format(num_jjs,L_left+10,L_right,I_de)
    with open('../_circuit_data/'+file_name, 'rb') as data_file:         
        data_array = pickle.load(data_file)
        # data_array = load_session_data(kwargs['file_name'])
        rate_array = data_array['rate_array']
        influx_list = data_array['influx_list']
        I_di_array = data_array['I_di_array']
            
    I_di = 0
    ind1 = (np.abs(np.asarray(influx_list)-1000)).argmin()
    ind2 = (np.abs(I_di_array[ind1]-I_di)).argmin()        
    rate = rate_array[ind1][ind2]                  
    print('ind1 = {}; ind2 = {}; rate = {}'.format(ind1,ind2,rate))