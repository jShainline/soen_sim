#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import pickle

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_dend_time_traces, plot_dend_rate_array__norm_to_phi0, plot_dend_rate_array__norm_to_phi0__two_panel_pos_neg
from _functions import save_session_data, read_wr_data
from _util import physical_constants

p = physical_constants()
Phi0 = p['Phi0__pH_ns']

plt.close('all')

#%% options

plot_time_traces = False
plot_rate_arrays = True

method = 'j_di_phase' # 'j_di_rate' # 'I_di' # 'j_di_phase' # 'j_di_phase__no_downsample'


#%% inputs

_temp_str_1 = 'wrspice_calling_scripts/soen_sim_data/'
_temp_str_2 = 'dend_4jj__flux_drive_arrays'

with open('{}{}.soen'.format(_temp_str_1,_temp_str_2), 'rb') as data_file:         
    data_array_imported = pickle.load(data_file)

Ib_vec = data_array_imported['Ib_vec']
num_Ib = len(Ib_vec)
Phi_p_list = data_array_imported['Phi_p_list']
num_Phi_p = len(Phi_p_list)
Phi_a_array = data_array_imported['Phi_a__array']

# parameters from WR sims
Ldi = 77.5 # nH # 775e3 # was 775e3 for Saeed's data set; may still need to reduce downsample by an order of magnitude
dt = 1 # ps

Ma = np.sqrt(400*12.5) # pH
Mp = 77.5 # pH

Phi_a__num_steps = 20
max_flux = p['Phi0__pH_ns']/2
_fr = max_flux/Phi_a__num_steps # flux_resolution
_fr_0 = _fr/100 # flux resolution of turn on

#%%

if method == 'j_di_rate':   
    min_peak_height = 182e-6 # units of volts for WR
    min_peak_distance = 10 # units of samples
    min_ht = 1000
    max_ht = -1000
    min_dist = 1000
    max_dist = -1000
    
if method == 'j_di_phase' or method == 'I_di':        
    min_peak_height = 50e-6 # 182e-6 # units of volts for WR
    min_peak_distance = 10 # 175 # units of samples
    downsample_factor = 400 # dt = 1ps
    window_size = 11 # samples/time steps for savitzky-golay filter (applied after downsample)
    plot_phase_vec_during_processing = False
       
#%%

run_all = True
# loop to create rate files for all cases
if run_all == True:
    for pp in range(num_Ib): # [0]: # 
        Ib = Ib_vec[pp]
        for qq in range(num_Phi_p): # [0]: # [3]: #        
            Phi_p = Phi_p_list[qq]
            Ip = Phi_p/Mp
            # print(Ip)
            
            Phi_a_vec = Phi_a_array[pp][qq]
            j_di_rate_array = []
            j_di_ifi_array = []            
            I_di_array = []
            
            num_drives = len(Phi_a_vec)
            
            # load wr data, find peaks, find rates
            for ii in range(num_drives): # [0]: #
                Phi_a = Phi_a_vec[ii] 
                # print(Phi_a)
                Ia = Phi_a/Ma
                # print(Ia)
                
                print('pp = {:d} of {:d} (Ib = {:06.2f}uA); qq = {:d} of {:d} (Phi_p/Phi_0 = {:07.5f}); ii = {:d} of {:d} (Phi_a/Phi0 = {:07.5f})'.format(pp+1,num_Ib,Ib,qq+1,num_Phi_p,Phi_p/Phi0,ii+1,num_drives,Phi_a/Phi0))
                
                directory = 'wrspice_data/4jj'          
                file_name = 'dend_4jj_one_bias_plstc_cnst_drv_seek_rt_arry_Ib{:06.2f}uA_Ip{:09.6f}uA_Ia{:09.6f}.dat'.format(Ib,Ip,Ia)
                j_di_str = 'v(4)'
                j_di_phase_str = 'v(15)'
                I_di_str = 'L7#branch'
                data_dict = read_wr_data(directory+'/'+file_name)
                
                # assign data
                time_vec = 1e9*data_dict['time']
                j_di = data_dict[j_di_str] # j_di = data_dict['v(5)']
                j_di_phase = data_dict[j_di_phase_str]
                j_di_phase = j_di_phase-j_di_phase[0]
                I_di = 1e6*data_dict[I_di_str]
                
                if method == 'j_di_phase':
                    if plot_phase_vec_during_processing == True:
                        fig, ax = plt.subplots(nrows = 1, ncols = 1)
                        fig.suptitle('as loaded')
                        ax.plot(time_vec,j_di_phase)
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
                    j_di_ifi = np.diff(time_vec[j_di_peaks])
                    j_di_rate = 1/j_di_ifi                        
                    j_di_ifi_array.append(j_di_ifi)
                    j_di_rate_array.append(j_di_rate)                        
                    I_di_array.append(I_di[j_di_peaks])                          
                
                if method == 'j_di_phase' or method == 'I_di':
                    
                    # downsample
                    initial_ind = (np.abs(time_vec-1.0)).argmin()
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
                        ax.plot(time_vec,j_di_phase)
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
                        ax.plot(time_vec__avg,j_di_phase__avg)
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
                        ax.plot(time_vec__avg,j_di_phase__avg)
                        ax.set_xlabel(r'time [ns]')
                        ax.set_ylabel(r'$J_{di}$ phase [rad.]')
                        plt.show()                
                    
                    # calculation based on j_di_phase
                    if method == 'j_di_phase':
                        j_di_rate = (np.diff(j_di_phase__avg)/(2*np.pi))/np.diff(time_vec__avg)
                    # calculation based on I_di
                    if method == 'I_di':
                        j_di_rate = (np.diff(I_di__avg)/np.diff(time_vec__avg))/(p['Phi0']/(Ldi*1e-9))
                        
                    for nn in range(len(j_di_rate)):
                        if j_di_rate[nn] < 0:
                            j_di_rate[nn] = 0
                    j_di_rate_array.append(j_di_rate)
                    I_di_array.append(I_di__avg)
     
            # convert current drive to flux
            influx_list = Phi_a_vec # []
            # M = Ma # np.sqrt(200*receiver_inductance)
            # I_drive__pad = 10e-3 # amount above the observed max of Idrive that the simulation will allow before giving a zero rate, units of uA
            # I_drive_list__pad = np.insert(I_drive_list,0,I_drive_list[0]-I_drive__pad) # adding extra zero so flux that rounds below minimum value gives zero rate
            # for ii in range(len(I_drive_list__pad)):
                # influx_list.append(M*I_drive_list__pad[ii])
                
            # assemble data and change units
            I_di_pad = 10e-3 # amount above the observed max of Idi that the simulation will allow before giving a zero rate, units of uA
            
            master_rate_array = [np.array([0,0])] # initializing with these zeros so influx that rounds below minimum value gives zero rate
            I_di_array__scaled = [np.array([0,I_di_pad])] # initializing with these zeros so influx that rounds below minimum value gives zero rate     
                
            for ii in range(num_drives):
                            
                tn = np.max([len(j_di_rate_array[ii]),1])
                temp_rate_vec = np.zeros([tn])
                temp_I_di_vec = np.zeros([tn])
                for jj in range(len(j_di_rate_array[ii])):
                    temp_rate_vec[jj] = j_di_rate_array[ii][jj] # master_rate_array has units of fluxons per nanoosecond
                    temp_I_di_vec[jj] = ( I_di_array[ii][jj]+I_di_array[ii][jj+1] )/2 # this makes I_di_array__scaled have the same dimensions as j_di_rate_array and units of uA
            
                master_rate_array.append([])
                # print(temp_rate_vec)
                master_rate_array[ii+1] = np.append(temp_rate_vec,0) # this zero added so that a current that rounds to I_di + I_di_pad will give zero rate
                I_di_array__scaled.append([])
                I_di_array__scaled[ii+1] = np.append(temp_I_di_vec,np.max(temp_I_di_vec)+I_di_pad) # this additional I_di + I_di_pad is included so that a current that rounds to I_di + I_di_pad will give zero rate
                
            # plot the rate array  
            if plot_rate_arrays == True:
                # plot_dend_rate_array__norm_to_phi0(I_di_array = I_di_array__scaled, influx_list = Phi_a_vec, master_rate_array = master_rate_array, Ib = Ib, Phi_p = Phi_p)
                plot_dend_rate_array__norm_to_phi0__two_panel_pos_neg(I_di_array = I_di_array__scaled, influx_list = Phi_a_vec, master_rate_array = master_rate_array, Ib = Ib, Phi_p = Phi_p)
                # plot_dend_rate_array(I_di_array = I_di_array__scaled, I_drive_list = I_drive_list, influx_list = influx_list, master_rate_array = master_rate_array, L_left = L_left_list[pp], I_de = I_de_list[qq])    
            
            # save data
            save_string = 'master_dnd_rate_array_4jj_Ib{:06.2f}uA_Ip{:09.6f}uA_Ldi{:07.2f}nH_dt{:04.1f}ps_dsf{:d}'.format(Ib,Ip,Ldi,dt,downsample_factor)
            data_array = dict()
            data_array['rate_array'] = master_rate_array
            data_array['influx_list'] = influx_list
            data_array['I_di_array'] = I_di_array__scaled
            print('\n\nsaving session data ...\n\n')
            # save_session_data(data_array,save_string)
            save_session_data(data_array,save_string+'.soen',False)
