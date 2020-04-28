#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import pickle
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3
from util import physical_constants
p = physical_constants()

plt.close('all')

#%% load wr data, find peaks, find rates

I_sy_list = [23,28,33,38]#uA

I_drive_array = ( [ [18.8,18.9,19,19.13,19.25,19.5,20],#Isy=23uA
                    [13.8,13.9,14,14.13,14.25,14.5,15,16,17,18,19,20],#Isy=28uA
                    [8.8,8.9,9,9.13,9.25,9.5,10,11,12,13,14,15,16,17,18,19,20],#Isy=33uA
                    [3.8,3.9,4,4.13,4.25,4.5,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]#Isy=38uA
                  ])#units of uA

t_sim_array = ( [ [20,40,59,90,63,47,37],#Isy=23uA
                  [20,40,60,90,63,47,37,30,41,30,27,25],#Isy=28uA
                  [20,40,60,90,63,47,37,30,41,30,27,25,24,23,22,22,22],#Isy=33uA
                  [20,40,59,90,63,47,38,37,41,30,27,25,24,23,22,22,21,21,21,21,20,20],#Isy=38uA
                ])#units of ns

#%%
    
j_sf_ifi_array = [] # array of inter-fluxon intervals at the synaptic firing junction
j_sf_rate_array = [] # array of fluxon production rate at the synaptic firing junction
j_jtl_ifi_array = [] # array of inter-fluxon intervals at the jtl junction
j_jtl_rate_array = [] # array of fluxon production rate at the jtl junction
j_si_ifi_array = [] # array of inter-fluxon intervals at the synaptic integration loop junction
j_si_rate_array = [] # array of fluxon production rate at the synaptic integration loop junction

I_si_array = [] # array of values of I_si

for ii in range(len(I_sy_list)):
       

    I_si_array.append([])
    # I_si_array[ii] = []
    
    j_sf_ifi_array.append([])
    j_sf_rate_array.append([])
    j_jtl_ifi_array.append([])
    j_jtl_rate_array.append([])
    j_si_ifi_array.append([])
    j_si_rate_array.append([])
        
    # j_sf_ifi_array[ii] = []
    # j_sf_rate_array[ii] = []
    # j_jtl_ifi_array[ii] = []
    # j_jtl_rate_array[ii] = []
    # j_si_ifi_array[ii] = []
    # j_si_rate_array[ii] = []
    
    num_drives = len(I_drive_array[ii]) 
    for jj in range(num_drives):
        
        print('ii = {:d} of {:d}, jj = {:d} of {:d}'.format(ii+1,len(I_sy_list),jj+1,num_drives))        
        
        directory = 'wrspice_data/fitting_data'
        file_name = 'syn_cnst_drv_Isy{:5.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0077.5ms_tsim{:04.0f}ns_dt00.1ps.dat'.format(I_sy_list[ii],I_drive_array[ii][jj],t_sim_array[ii][jj])
        data_dict = read_wr_data(directory+'/'+file_name)
        
        # find peaks for each jj
        time_vec = data_dict['time']
        j_sf = data_dict['v(3)']
        j_jtl = data_dict['v(4)']
        j_si = data_dict['v(5)']
        
        initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
        final_ind = (np.abs(time_vec-t_sim_array[ii][jj]*1e-9)).argmin()
    
        time_vec = time_vec[initial_ind:final_ind]
        j_sf = j_sf[initial_ind:final_ind]
        j_jtl = j_jtl[initial_ind:final_ind]
        j_si = j_si[initial_ind:final_ind]
      
        j_sf_peaks, _ = find_peaks(j_sf, height = 200e-6)
        j_jtl_peaks, _ = find_peaks(j_jtl, height = 200e-6)
        j_si_peaks, _ = find_peaks(j_si, height = 200e-6)
    
        # fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)
        # fig.suptitle(file_name)   
        # ax[0].plot(time_vec*1e9,j_sf*1e3, '-', label = '$J_{sf}$')             
        # ax[0].plot(time_vec[j_sf_peaks]*1e9,j_sf[j_sf_peaks]*1e3, 'x')
        # ax[0].set_xlabel(r'Time [ns]')
        # ax[0].set_ylabel(r'Voltage [mV]')
        # ax[0].legend()
        # ax[1].plot(time_vec*1e9,j_jtl*1e3, '-', label = '$J_{jtl}$')             
        # ax[1].plot(time_vec[j_jtl_peaks]*1e9,j_jtl[j_jtl_peaks]*1e3, 'x')
        # ax[1].set_xlabel(r'Time [ns]')
        # ax[1].set_ylabel(r'Voltage [mV]')
        # ax[1].legend()
        # ax[2].plot(time_vec*1e9,j_si*1e3, '-', label = '$J_{si}$')             
        # ax[2].plot(time_vec[j_si_peaks]*1e9,j_si[j_si_peaks]*1e3, 'x')
        # ax[2].set_xlabel(r'Time [ns]')
        # ax[2].set_ylabel(r'Voltage [mV]')
        # ax[2].legend()
        # plt.show()
        
        # find inter-fluxon intervals and fluxon generation rates for each JJ
        I_si = data_dict['L3#branch']
        I_si = I_si[initial_ind:final_ind]
        I_drive = data_dict['L0#branch']
        I_drive = I_drive[initial_ind:final_ind]            
        
        j_sf_ifi = np.diff(time_vec[j_sf_peaks])
        j_jtl_ifi = np.diff(time_vec[j_jtl_peaks])
        j_si_ifi = np.diff(time_vec[j_si_peaks])
        
        j_sf_rate = 1/j_sf_ifi
        j_jtl_rate = 1/j_jtl_ifi
        j_si_rate = 1/j_si_ifi
        
        j_sf_ifi_array[ii].append(j_sf_ifi)
        j_sf_rate_array[ii].append(j_sf_rate)
        j_jtl_ifi_array[ii].append(j_jtl_ifi)
        j_jtl_rate_array[ii].append(j_jtl_rate)
        j_si_ifi_array[ii].append(j_si_ifi)
        j_si_rate_array[ii].append(j_si_rate)
        
        I_si_array[ii].append(I_si[j_si_peaks])
    
    
#%% assemble data and change units
I_si_pad = 100e-9 # amount above the observed max of Isi that the simulation will allow before giving a zero rate

master_rate_array = []
I_si_array__scaled = []
for ii in range(len(I_sy_list)):    
    
    master_rate_array.append([])
    I_si_array__scaled.append([])
    
    num_drives = len(I_drive_array[ii])
    for jj in range(num_drives):
                
        temp_rate_vec = np.zeros([len(j_si_rate_array[ii][jj])])
        temp_I_si_vec = np.zeros([len(j_si_rate_array[ii][jj])])
        for kk in range(len(j_si_rate_array[ii][jj])):
            temp_rate_vec[kk] = 1e-6*j_si_rate_array[ii][jj][kk] # master_rate_array has units of fluxons per microsecond
            temp_I_si_vec[kk] = 1e6*( I_si_array[ii][jj][kk]+I_si_array[ii][jj][kk+1] )/2 # this makes I_si_array__scaled have the same dimensions as j_si_rate_array and units of uA

        master_rate_array[ii].append([])
        master_rate_array[ii][jj] = np.append(temp_rate_vec,0) # this zero is added so that a current that rounds to I_si + I_si_pad will give zero rate
        I_si_array__scaled[ii].append([])
        I_si_array__scaled[ii][jj] = np.append(temp_I_si_vec,np.max(temp_I_si_vec)+I_si_pad) # this additional I_si + I_si_pad is included so that a current that rounds to I_si + I_si_pad will give zero rate


#%% plot the rate vectors 

for ii in range(len(I_sy_list)):
    num_files = len(I_drive_array[ii])
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
    fig.suptitle('master _ sy _ rates') 
    plt.title('$I_sy = $ {} $\mu$A'.format(I_sy_list[ii]))
    for jj in range(num_files):
        ax.plot(I_si_array__scaled[ii][jj][:],master_rate_array[ii][jj][:]*1e-3, '-', label = 'I_drive = {}'.format(I_drive_array[ii][jj]))    
    ax.set_xlabel(r'$I_{si}$ [$\mu$A]')
    ax.set_ylabel(r'$r_{j_{si}}$ [kilofluxons per $\mu$s]')
    ax.legend()
    plt.show()

fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
fig.suptitle('master _ sy _ rates') 
plt.title('$I_sy = $ {} $\mu$A'.format(I_sy_list[ii]))

for ii in range(len(I_sy_list)):
    num_files = len(I_drive_array[ii])
    
    for jj in range(num_files):
        ax.plot(I_si_array__scaled[ii][jj][:],master_rate_array[ii][jj][:]*1e-3, '-', label = 'I_drive = {}'.format(I_drive_array[ii][jj]))  
        
ax.set_xlabel(r'$I_{si}$ [$\mu$A]')
ax.set_ylabel(r'$r_{j_{si}}$ [kilofluxons per $\mu$s]')
ax.legend()
plt.show()

#%% save the data    
 
save_string = 'master__syn__rate_array__I_si_pad{:04.0f}nA'.format(I_si_pad*1e9)
data_array = dict()
data_array['rate_array'] = master_rate_array
data_array['I_drive_array'] = I_drive_array#[x*1e-6 for x in I_drive_vec]
data_array['I_si_array'] = I_si_array__scaled
data_array['I_sy_list'] = I_sy_list
print('\n\nsaving session data ...')
save_session_data(data_array,save_string)


#%% load test
with open('../_circuit_data/master__syn__rate_array__Isipad0100nA.soen', 'rb') as data_file:         
        data_array_imprt = pickle.load(data_file)
# data_array_imported = load_session_data('session_data__master_rate_matrix__syn__2020-04-24_10-24-23.dat')
I_si_array__imprt = data_array_imprt['I_si_array']
I_drive_array__imprt = data_array_imprt['I_drive_array']
I_sy_list__imprt = data_array_imprt['I_sy_list']
rate_array__imprt = data_array_imprt['rate_array']

I_sy_sought = 29
I_sy_sought_ind = (np.abs(np.asarray(I_sy_list__imprt)-I_sy_sought)).argmin()
I_drive_sought = 14.45
I_drive_sought_ind = (np.abs(np.asarray(I_drive_array__imprt[I_sy_sought_ind])-I_drive_sought)).argmin()
I_si_sought = 4.552
I_si_sought_ind = (np.abs(np.asarray(I_si_array__imprt[I_sy_sought_ind][I_drive_sought_ind])-I_si_sought)).argmin()
rate_obtained = rate_array__imprt[I_sy_sought_ind][I_drive_sought_ind][I_si_sought_ind]

print('I_sy_sought = {:7.4f}uA, I_sy_sought_ind = {:d}\nI_drive_sought = {:7.4f}uA, I_drive_sought_ind = {:d}\nI_si_sought = {:7.4f}uA, I_si_sought_ind = {:d}\nrate_obtained = {:10.4f} fluxons per us'.format(I_sy_sought,I_sy_sought_ind,I_drive_sought,I_drive_sought_ind,I_si_sought,I_si_sought_ind,rate_obtained))

    
#%%

# #%% plot the matrices a few different ways

# for kk in range(len(I_sy_list)):
    
#     fig, ax = plt.subplots(1,1)
#     rates = ax.imshow(np.transpose(master_rate_matrices[kk][:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[I_drive_mat[kk][0],I_drive_mat[kk][-1],0,max_I_si_list[kk]], aspect = 'auto', origin = 'lower')
#     cbar = fig.colorbar(rates, extend='both')
#     cbar.minorticks_on()     
#     fig.suptitle('$Rate [kilofluxons per $\mu$s]')
#     # plt.title(title_string)
#     ax.set_xlabel(r'$I_{drive}$ [$\mu$A]')
#     ax.set_ylabel(r'$I_{si}$ [$\mu$A]')  
#     ax.set_ylim() 
#     plt.show()      
#     # fig.savefig('figures/'+save_str+'__log.png')
    
#     # fig = plt.figure()
#     # ax = fig.gca(projection='3d')
#     # Y = I_drive_mat[kk]
#     # X = I_si_mat__scaled[kk]
#     # XXa, YYb = np.meshgrid(X, Y)
#     # surf = ax.plot_surface(XXa, YYb, master_rate_matrices[kk][:,:]*1e-3, cmap=cm.viridis,linewidth=0, antialiased=False)
#     # ax.set_xlabel('$I_{si}$ [$\mu$A]')
#     # ax.set_ylabel('$I_{spd}$ [$\mu$A]')
#     # ax.set_zlabel('Rate of fluxon generation [kilofluxons per $\mu$s]')
    
#     # num_files = len(I_drive_mat[kk])
#     # fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
#     # fig.suptitle('master _ sy _ rate_matrix') 
#     # for ii in range(num_files):
#     #     ax.plot(I_si_mat__scaled[kk][ii][:],master_rate_matrices[kk][ii,:]*1e-3, '-', label = 'I_drive = {}'.format(I_drive_mat[kk][ii]))    
#     # ax.set_xlabel(r'$I_{si}$ [$\mu$A]')
#     # ax.set_ylabel(r'Rate [kilofluxons per $\mu$s]')
#     # ax.legend()
#     # plt.show()

