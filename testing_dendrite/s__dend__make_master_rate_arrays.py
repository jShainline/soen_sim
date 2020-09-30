#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_dend_time_traces, plot_dend_rate_array__norm_to_phi0
from _functions import save_session_data, read_wr_data
from util import physical_constants

p = physical_constants()

plt.close('all')

#%% set case

num_jjs = 4 # 2 or 4


#%% options

plot_time_traces = False
plot_rate_arrays = True

method = 'j_di_phase' # 'j_di_rate' # 'I_di' # 'j_di_phase' # 'j_di_phase__no_downsample'


#%% inputs

if num_jjs == 2:

    # dendritic firing junction bias current
    dI_de = 1
    I_de_0 = 52
    I_de_f = 80

elif num_jjs == 4:
    
    # dendritic firing junction bias current
    dI_de = 1
    I_de_0 = 56
    I_de_f = 90
    
I_de_list = np.arange(I_de_0,I_de_f+dI_de,dI_de)
num_I_de = len(I_de_list)

# current for flux bias
Phi_vec = np.linspace(0,p['Phi0__pH_ns']/2,50) # units of uA pH
M = np.sqrt(200*20)
I_drive_vec = Phi_vec/M # units of uA
resolution = 10e-3 # units of uA
I_drive_vec_round = np.round((Phi_vec/M)/resolution)*resolution
I_drive_vector = np.round(I_drive_vec_round,3) # units of uA

# establish flux drive array
SpikeStart = np.loadtxt('wrspice_data/{:d}jj/SpikeStart__{:d}jj.dat'.format(num_jjs,num_jjs))
# print(SpikeStart)

I_drive_array=[]
for iIde in np.arange(len(I_de_list)):
    temp_I_drive=[]
    I_drive=np.round(SpikeStart[iIde],3)
    temp_I_drive.append(I_drive)
    
    index=(np.abs(I_drive_vector-I_drive)).argmin()

    if I_drive>I_drive_vector[index]:
        index=index+1
        
    while (I_drive+0.1)< I_drive_vector[index]:
        I_drive=I_drive+0.1
        temp_I_drive.append(round(I_drive,3))
    while I_drive< I_drive_vector[-1]:
        I_drive=I_drive_vector[index]
        temp_I_drive.append(round(I_drive,3))
        index=index+1

    I_drive_array.append(temp_I_drive)
# print(I_drive_array)
I_drive_array = [I_drive_array]
    
# inductances
L_left_list = [20] # np.arange(17,23+dL,dL) # pH
L_right_list = [20] # np.flip(np.arange(17,23+dL,dL)) # pH
num_L = len(L_right_list)

DI_loop_inductance = 775e3

#%%

if method == 'j_di_rate':
    if num_jjs == 2:        
        min_peak_height = 100e-6 # units of volts for WR
        min_peak_distance = 1 # units of samples
        min_ht = 1000
        max_ht = -1000
        min_dist = 1000
        max_dist = -1000
    elif num_jjs == 4:        
        min_peak_height = 182e-6 # units of volts for WR
        min_peak_distance = 10 # units of samples
        min_ht = 1000
        max_ht = -1000
        min_dist = 1000
        max_dist = -1000
    
if method == 'j_di_phase' or method == 'I_di':        
    min_peak_height = 100e-6 # 182e-6 # units of volts for WR
    min_peak_distance = 10 # 175 # units of samples
    if num_jjs == 2:
        downsample_factor = 2000 # 300 # dt = 1ps
        window_size = 11 # samples/time steps for savitzky-golay filter (applied after downsample)
    elif num_jjs == 4:
        downsample_factor = 2000 # dt = 1ps
        window_size = 11 # samples/time steps for savitzky-golay filter (applied after downsample)
    plot_phase_vec_during_processing = False
       
#%%

run_all = True
# loop to create rate files for all cases
if run_all == True:
    for pp in range(num_L): # [0]: # 
        
        for qq in range(num_I_de): # [0]: # [3]: # 
                    
            # load wr data, find peaks, find rates
            I_drive_list = I_drive_array[pp][qq]
            j_di_rate_array = []
            j_di_ifi_array = []            
            I_di_array = []
            
            # I_drive_list = [10]
            num_drives = len(I_drive_list)
            for ii in range(num_drives): # [0]: # 
                
                print('pp = {:d} of {:d} (L_left = {} pH); qq = {:d} of {:d} (I_de = {} uA); ii = {:d} of {:d} (I_drive = {} uA)'.format(pp+1,num_L,L_left_list[pp],qq+1,num_I_de,I_de_list[qq],ii+1,num_drives,I_drive_list[ii]))
                
                I_drive = I_drive_list[ii]
                if num_jjs == 2:            
                    directory = 'wrspice_data/2jj'            
                    file_name = 'dend_cnst_drv_2jj_Llft{:05.2f}pH_Lrgt{:05.2f}pH_Ide{:05.2f}uA_Idrv{:05.2f}uA_Ldi0775.0nH_taudi0775ms_dt01.0ps.dat'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq],I_drive)
                    j_di_str = 'v(3)'
                    j_di_phase_str = 'v(8)'
                    I_di_str = 'i(L0)'
                elif num_jjs == 4:
                    directory = 'wrspice_data/4jj'
                    file_name = 'dend_cnst_drv_4jj_Llft{:05.2f}pH_Lrgt{:05.2f}pH_Ide{:05.2f}uA_Idrv{:05.2f}uA_Ldi0775.0nH_taudi0775ms_dt01.0ps.dat'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq],I_drive) 
                    j_di_str = 'v(5)'
                    j_di_phase_str = 'v(12)'
                    I_di_str = 'i(L2)'
                data_dict = read_wr_data(directory+'/'+file_name)
                
                # assign data
                time_vec = data_dict['time']
                j_di = data_dict[j_di_str] # j_di = data_dict['v(5)']
                j_di_phase = data_dict[j_di_phase_str]
                j_di_phase = j_di_phase-j_di_phase[0]
                I_di = data_dict[I_di_str]
                
                if method == 'j_di_phase':
                    if plot_phase_vec_during_processing == True:
                        fig, ax = plt.subplots(nrows = 1, ncols = 1)
                        fig.suptitle('as loaded')
                        ax.plot(time_vec*1e9,j_di_phase)
                        ax.set_xlabel(r'time [ns]')
                        ax.set_ylabel(r'$J_{di}$ phase [rad.]')
                        plt.show()
                
                # find peaks for each jj          
                j_di_peaks, _ = find_peaks(j_di, height = min_peak_height, distance = min_peak_distance)
                
                if method == 'j_di_rate':
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

                if plot_time_traces == True and method != 'j_di_phase__no_downsample':
                    plot_dend_time_traces(time_vec,j_di,j_di_peaks,min_peak_height,I_di,file_name)
                                                            
                # find fluxon generation rate   

                if method == 'j_di_rate':
                    if num_jjs == 2:
                        j_di_peaks = j_di_peaks[1::2]                        
                    j_di_ifi = np.diff(time_vec[j_di_peaks])
                    j_di_rate = 1/j_di_ifi                        
                    j_di_ifi_array.append(j_di_ifi)
                    j_di_rate_array.append(j_di_rate)                        
                    I_di_array.append(I_di[j_di_peaks])                          
                
                if method == 'j_di_phase' or method == 'I_di':
                    
                    # downsample
                    initial_ind = (np.abs(time_vec-1.0e-9)).argmin()
                    # if method == 'j_di_rate':
                    #     final_ind = (np.abs(time_vec-np.max(time_vec))).argmin()
                    # elif method == 'j_di_phase' or method == 'I_di':
                    #     final_ind = len(time_vec)-1 # np.min([j_di_peaks[-1]+window_size*downsample_factor,len(time_vec)-1]) # j_di_peaks[-1]*2*downsample_factor
                    final_ind = len(time_vec)-1
                    time_vec = time_vec[initial_ind:final_ind]
                    j_di = j_di[initial_ind:final_ind]
                    j_di_phase = j_di_phase[initial_ind:final_ind]-j_di_phase[initial_ind]
                    I_di = I_di[initial_ind:final_ind]
                    
                    if plot_phase_vec_during_processing == True:
                        fig, ax = plt.subplots(nrows = 1, ncols = 1)
                        fig.suptitle('cropped')
                        ax.plot(time_vec*1e9,j_di_phase)
                        ax.set_xlabel(r'time [ns]')
                        ax.set_ylabel(r'$J_{di}$ phase [rad.]')
                        plt.show()
                    
                    index_vec = np.arange(0,len(time_vec),downsample_factor)
                    I_di__avg = I_di[index_vec]
                    j_di_phase__avg = j_di_phase[index_vec]
                    time_vec__avg = time_vec[index_vec]
                    
                    if plot_phase_vec_during_processing == True:
                        fig, ax = plt.subplots(nrows = 1, ncols = 1)
                        fig.suptitle('downsampled')
                        ax.plot(time_vec__avg*1e9,j_di_phase__avg)
                        ax.set_xlabel(r'time [ns]')
                        ax.set_ylabel(r'$J_{di}$ phase [rad.]')
                        plt.show()                
                    
                    # apply smoothing filter
                    I_di__avg = savgol_filter(I_di__avg, window_size, 3) # polynomial order 3
                    j_di_phase__avg = savgol_filter(j_di_phase__avg, window_size, 3) # polynomial order 3
                    time_vec__avg = savgol_filter(time_vec__avg, np.min([window_size,len(time_vec__avg)]), 3) # polynomial order 3
                    
                    if plot_phase_vec_during_processing == True:
                        fig, ax = plt.subplots(nrows = 1, ncols = 1)
                        fig.suptitle('smoothed')
                        ax.plot(time_vec__avg*1e9,j_di_phase__avg)
                        ax.set_xlabel(r'time [ns]')
                        ax.set_ylabel(r'$J_{di}$ phase [rad.]')
                        plt.show()                
                    
                    # calculation based on j_di_phase
                    if method == 'j_di_phase':
                        j_di_rate = (np.diff(j_di_phase__avg)/(2*np.pi))/np.diff(time_vec__avg)
                    # calculation based on I_di
                    if method == 'I_di':
                        j_di_rate = (np.diff(I_di__avg)/np.diff(time_vec__avg))/(p['Phi0']/DI_loop_inductance)
                        
                    for nn in range(len(j_di_rate)):
                        if j_di_rate[nn] < 0:
                            j_di_rate[nn] = 0
                    j_di_rate_array.append(j_di_rate)
                    I_di_array.append(I_di__avg)
     
            # convert current drive to flux
            influx_list = []
            receiver_inductance = L_left_list[pp]
            M = np.sqrt(200*receiver_inductance)
            I_drive__pad = 10e-3 # amount above the observed max of Idrive that the simulation will allow before giving a zero rate, units of uA
            I_drive_list__pad = np.insert(I_drive_list,0,I_drive_list[0]-I_drive__pad) # adding extra zero so flux that rounds below minimum value gives zero rate
            for ii in range(len(I_drive_list__pad)):
                influx_list.append(M*I_drive_list__pad[ii])
                
            # assemble data and change units
            I_di_pad = 10e-3 # amount above the observed max of Idi that the simulation will allow before giving a zero rate, units of uA
            
            master_rate_array = [np.array([0,0])] # initializing with these zeros so influx that rounds below minimum value gives zero rate
            I_di_array__scaled = [np.array([0,I_di_pad])] # initializing with these zeros so influx that rounds below minimum value gives zero rate     
                
            for ii in range(num_drives):
                            
                tn = np.max([len(j_di_rate_array[ii]),1])
                temp_rate_vec = np.zeros([tn])
                temp_I_di_vec = np.zeros([tn])
                for jj in range(len(j_di_rate_array[ii])):
                    temp_rate_vec[jj] = 1e-9*j_di_rate_array[ii][jj] # master_rate_array has units of fluxons per nanoosecond
                    temp_I_di_vec[jj] = 1e6*( I_di_array[ii][jj]+I_di_array[ii][jj+1] )/2 # this makes I_di_array__scaled have the same dimensions as j_di_rate_array and units of uA
            
                master_rate_array.append([])
                master_rate_array[ii+1] = np.append(temp_rate_vec,0) # this zero added so that a current that rounds to I_di + I_di_pad will give zero rate
                I_di_array__scaled.append([])
                I_di_array__scaled[ii+1] = np.append(temp_I_di_vec,np.max(temp_I_di_vec)+I_di_pad) # this additional I_di + I_di_pad is included so that a current that rounds to I_di + I_di_pad will give zero rate
                
            # plot the rate array  
            if plot_rate_arrays == True:
                plot_dend_rate_array__norm_to_phi0(I_di_array = I_di_array__scaled, I_drive_list = I_drive_list, influx_list = influx_list, master_rate_array = master_rate_array, L_left = L_left_list[pp], I_de = I_de_list[qq])
                # plot_dend_rate_array(I_di_array = I_di_array__scaled, I_drive_list = I_drive_list, influx_list = influx_list, master_rate_array = master_rate_array, L_left = L_left_list[pp], I_de = I_de_list[qq])    
            
            # save data
            save_string = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}'.format(num_jjs,L_left_list[pp],L_right_list[pp],I_de_list[qq])
            data_array = dict()
            data_array['rate_array'] = master_rate_array
            data_array['I_drive_list'] = I_drive_list
            data_array['influx_list'] = influx_list
            data_array['I_di_array'] = I_di_array__scaled
            print('\n\nsaving session data ...\n\n')
            # save_session_data(data_array,save_string)
            save_session_data(data_array,save_string+'.soen',False)
