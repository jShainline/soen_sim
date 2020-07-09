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

I_sy_vec = [23,28,33,38]#uA
I_drive_mat = ( [ [18.8,18.9,19,20],#Isy=23uA
                  [13.8,13.9,14,15,16,17,18,19,20],#Isy=28uA
                  [8.8,8.9,9,10,11,12,13,14,15,16,17,18,19,20],#Isy=33uA
                  [3.8,3.9,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]#Isy=38uA
                  ])#units of uA
t_sim_mat = ( [ [20,40,59,37],#Isy=23uA
                [20,40,60,37,30,41,30,27,25],#Isy=28uA
                [20,40,60,37,30,41,30,27,25,24,23,22,22,22],#Isy=33uA
                [20,40,59,38,37,41,30,27,25,24,23,22,22,21,21,21,21,20,20],#Isy=38uA
                ])#units of ns

#%%
    
master_data__sf__t_fq = []
master_data__sf__r_fq = []
master_data__jtl__t_fq = []
master_data__jtl__r_fq = []
master_data__si__t_fq = []
master_data__si__r_fq = []
I_si_mat = []
max_I_si_list = []

for jj in range(len(I_sy_vec)):
    
    num_files = len(I_drive_mat[jj])    

    max_I_si_list.append(0)
    I_si_mat.append([])
    I_si_mat[jj] = []
    
    master_data__sf__t_fq.append([])
    master_data__sf__r_fq.append([])
    master_data__jtl__t_fq.append([])
    master_data__jtl__r_fq.append([])
    master_data__si__t_fq.append([])
    master_data__si__r_fq.append([])
        
    master_data__sf__t_fq[jj] = []
    master_data__sf__r_fq[jj] = []
    master_data__jtl__t_fq[jj] = []
    master_data__jtl__r_fq[jj] = []
    master_data__si__t_fq[jj] = []
    master_data__si__r_fq[jj] = []
    
    for ii in range(num_files):
        
        print('jj = {:d} of {:d}, ii = {:d} of {:d}'.format(jj+1,len(I_sy_vec),ii+1,num_files))        
        
        directory = 'wrspice_data/fitting_data'
        file_name = 'syn_cnst_drv_Isy{:5.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0077.5ms_tsim{:04.0f}ns_dt00.1ps.dat'.format(I_sy_vec[jj],I_drive_mat[jj][ii],t_sim_mat[jj][ii])
        data_dict = read_wr_data(directory+'/'+file_name)
        
        # find peaks for each jj
        time_vec = data_dict['time']
        j_sf = data_dict['v(3)']
        j_jtl = data_dict['v(4)']
        j_si = data_dict['v(5)']
        
        initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
        final_ind = (np.abs(time_vec-t_sim_mat[jj][ii]*1e-9)).argmin()
    
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
        max_I_si_list[jj] = np.max([np.max(I_si),max_I_si_list[jj]])
        
        # I_dr1 = Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,Iflux_vec[jj],I_b,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
        
        j_sf_ifi = np.diff(time_vec[j_sf_peaks])
        j_jtl_ifi = np.diff(time_vec[j_jtl_peaks])
        j_si_ifi = np.diff(time_vec[j_si_peaks])
        
        j_sf_rate = 1/j_sf_ifi
        j_jtl_rate = 1/j_jtl_ifi
        j_si_rate = 1/j_si_ifi
        
        master_data__sf__t_fq[jj].append(j_sf_ifi)
        master_data__sf__r_fq[jj].append(j_sf_rate)
        master_data__jtl__t_fq[jj].append(j_jtl_ifi)
        master_data__jtl__r_fq[jj].append(j_jtl_rate)
        master_data__si__t_fq[jj].append(j_si_ifi)
        master_data__si__r_fq[jj].append(j_si_rate)
        
        I_si_mat[jj].append(I_si[j_si_peaks])
    
#%% assemble data

master_rate_matrices = []
I_si_vecs = []
for kk in range(len(I_sy_vec)):
    
    len_of_longest_rate_vec = 0
    fastest_rate = 0
    num_files = len(I_drive_mat[kk])
    for ii in range(num_files):
        if len(master_data__si__r_fq[kk][ii]) > len_of_longest_rate_vec:
            len_of_longest_rate_vec = len(master_data__si__r_fq[kk][ii])
        fastest_rate = np.max([np.max(master_data__si__r_fq[kk][ii]),fastest_rate])        
    
    #construct master I_si_vec
    I_si_resolution = 10e-9
    I_si_vecs.append([])
    I_si_vecs[kk] = np.arange(0,max_I_si_list[kk]+I_si_resolution,I_si_resolution)
    num_I_si = len(I_si_vecs[kk])
    master_rate_matrices.append([])
    master_rate_matrices[kk] = np.zeros([num_files,num_I_si])
    for ii in range(num_files):
        I_si_max__this_file = np.max(I_si_mat[kk][ii][:])
        for jj in range(num_I_si):
            
            if I_si_vecs[kk][jj] > I_si_max__this_file:
                master_rate_matrices[kk][ii,jj] = 0
            else:            
                ind = (np.abs(I_si_mat[kk][ii][:] - I_si_vecs[kk][jj])).argmin()
                if ind == len(master_data__si__r_fq[kk][ii][:]):
                    ind = ind-1
                master_rate_matrices[kk][ii,jj] = 1e-6*master_data__si__r_fq[kk][ii][ind] # units of fluxons per microsecond

#%% change units
I_si_mat__scaled = []
for ii in range(len(I_sy_vec)):
    num_files = len(I_drive_mat[ii])
    I_si_mat__scaled.append([])
    for jj in range(num_files):
        I_si_mat__scaled[ii].append([])
        num_I_si = len(I_si_mat[ii][jj])
        for kk in range(num_I_si):
            I_si_mat__scaled[ii][jj].append([])
            I_si_mat__scaled[ii][jj][kk] = 1e6*I_si_mat[ii][jj][kk]

#%% plot the matrices a few different ways

for kk in range(len(I_sy_vec)):
    
    fig, ax = plt.subplots(1,1)
    rates = ax.imshow(np.transpose(master_rate_matrices[kk][:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[I_drive_mat[kk][0],I_drive_mat[kk][-1],0,max_I_si_list[kk]], aspect = 'auto', origin = 'lower')
    cbar = fig.colorbar(rates, extend='both')
    cbar.minorticks_on()     
    fig.suptitle('$Rate [kilofluxons per $\mu$s]')
    # plt.title(title_string)
    ax.set_xlabel(r'$I_{drive}$ [$\mu$A]')
    ax.set_ylabel(r'$I_{si}$ [$\mu$A]')  
    ax.set_ylim() 
    plt.show()      
    # fig.savefig('figures/'+save_str+'__log.png')
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # Y = I_drive_mat[kk]
    # X = I_si_mat__scaled[kk]
    # XXa, YYb = np.meshgrid(X, Y)
    # surf = ax.plot_surface(XXa, YYb, master_rate_matrices[kk][:,:]*1e-3, cmap=cm.viridis,linewidth=0, antialiased=False)
    # ax.set_xlabel('$I_{si}$ [$\mu$A]')
    # ax.set_ylabel('$I_{spd}$ [$\mu$A]')
    # ax.set_zlabel('Rate of fluxon generation [kilofluxons per $\mu$s]')
    
    # num_files = len(I_drive_mat[kk])
    # fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
    # fig.suptitle('master _ sy _ rate_matrix') 
    # for ii in range(num_files):
    #     ax.plot(I_si_mat__scaled[kk][ii][:],master_rate_matrices[kk][ii,:]*1e-3, '-', label = 'I_drive = {}'.format(I_drive_mat[kk][ii]))    
    # ax.set_xlabel(r'$I_{si}$ [$\mu$A]')
    # ax.set_ylabel(r'Rate [kilofluxons per $\mu$s]')
    # ax.legend()
    # plt.show()

#%% save data
save_string = 'master_rate_matrices__syn'
data_array = dict()
data_array['master_rate_matrices'] = master_rate_matrices
data_array['I_drive_vecs'] = I_drive_mat#[x*1e-6 for x in I_drive_vec]
data_array['I_si_vecs'] = I_si_mat__scaled
data_array['I_sy_vec'] = I_sy_vec
print('\n\nsaving session data ...')
save_session_data(data_array,save_string)

#%% load test
with open('../master__syn__rate_matrices.soen', 'rb') as data_file:         
        data_array_imprt = pickle.load(data_file)
# data_array_imported = load_session_data('session_data__master_rate_matrix__syn__2020-04-24_10-24-23.dat')
I_si_vecs__imprt = data_array_imprt['I_si_vecs']
I_drive_vecs__imprt = data_array_imprt['I_drive_vecs']
I_sy_vec__imprt = data_array_imprt['I_sy_vec']
master_rate_matrices__imprt = data_array_imprt['master_rate_matrices']

I_sy_sought = 29
I_sy_sought_ind = (np.abs(np.asarray(I_sy_vec__imprt)-I_sy_sought)).argmin()
I_drive_sought = 14.45
I_drive_sought_ind = (np.abs(np.asarray(I_drive_vecs__imprt[I_sy_sought_ind])-I_drive_sought)).argmin()
I_si_sought = 4.552
I_si_sought_ind = (np.abs(np.asarray(I_si_vecs__imprt[I_sy_sought_ind][I_drive_sought_ind])-I_si_sought)).argmin()
rate_obtained = master_rate_matrices__imprt[I_sy_sought_ind][I_drive_sought_ind,I_si_sought_ind]

print('I_drive_sought = {:2.2f}uA, I_drive_sought_ind = {:d}\nI_si_sought = {:2.2f}uA, I_si_sought_ind = {:d}\nrate_obtained = {:3.2f} fluxons per us'.format(I_drive_sought,I_drive_sought_ind,I_si_sought,I_si_sought_ind,rate_obtained))

#%%
agf = I_si_mat__scaled[0][1]
