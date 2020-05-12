#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import pickle
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import numpy.matlib

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_syn_rate_array, plot_syn_rate_array__waterfall
from _functions import save_session_data, load_session_data, read_wr_data, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3
from util import physical_constants
p = physical_constants()

plt.close('all')

#%% load wr data, find peaks, find rates

I_sy = 40#uA

I_drive_list = ( [1.76,1.77,1.78,1.79,1.8,#1
                  1.9,2,2.1,2.2,2.3,#2
                  2.4,2.5,2.75,3,3.5,#3
                  4,4.25,4.5,4.75,4.8,#4
                  4.85,4.9,4.95,5,5.25,#5
                  5.5,5.75,6,6.25,6.5,#6
                  6.75,7,7.25,7.5,7.75,8,8.25,8.5,8.75,#7
                  9,9.25,9.5,9.75,10,10.25,10.5,10.75,11,#8
                  11.5,12,12.5,13,13.5,#9
                  14,14.5,15,15.5,16,#10
                  16.5,17,17.5,18,18.5,#11
                  19,19.25,19.5,19.6,19.7,#12
                  19.8,19.9,20] )#13; units of uA

t_sim_list = ( [10,10,13,16,20,#1
                40,60,82,72,58,#2
                51,47,41,37,33,#3
                30,30,29,27,30,#4
                38,49,45,41,35,#5
                32,32,30,29,29,#6
                27,27,27,26,25,25,25,24,24,#7
                24,23,23,23,23,22,23,22,22,#8
                22,22,22,21,21,#9
                21,21,21,21,21,#10
                21,20,20,20,20,#11
                20,20,20,20,20,#12
                20,20,20] )#13; units of ns

#%%
    
j_sf_ifi_array = [] # array of inter-fluxon intervals at the synaptic firing junction
j_sf_rate_array = [] # array of fluxon production rate at the synaptic firing junction
j_jtl_ifi_array = [] # array of inter-fluxon intervals at the jtl junction
j_jtl_rate_array = [] # array of fluxon production rate at the jtl junction
j_si_ifi_array = [] # array of inter-fluxon intervals at the synaptic integration loop junction
j_si_rate_array = [] # array of fluxon production rate at the synaptic integration loop junction

I_si_array = [] # array of values of I_si

num_drives = len(I_drive_list) 
directory = 'wrspice_data/fitting_data/3jj'
for ii in range(num_drives):
    
    print('ii = {:d} of {:d}'.format(ii+1,num_drives))        
    
    file_name = 'syn_cnst_drv_Isy{:5.2f}uA_Idrv{:05.2f}uA_Ldi0077.50nH_taudi0077.5ms_tsim{:04.0f}ns_dt00.1ps.dat'.format(I_sy,I_drive_list[ii],t_sim_list[ii])
    data_dict = read_wr_data(directory+'/'+file_name)
    
    # find peaks for each jj
    time_vec = data_dict['time']
    j_sf = data_dict['v(3)']
    j_jtl = data_dict['v(4)']
    j_si = data_dict['v(5)']
    
    # initial_ind = (np.abs(time_vec-2.0e-9)).argmin()
    # final_ind = (np.abs(time_vec-t_sim_list[ii]*1e-9)).argmin()

    # time_vec = time_vec[initial_ind:final_ind]
    # j_sf = j_sf[initial_ind:final_ind]
    # j_jtl = j_jtl[initial_ind:final_ind]
    # j_si = j_si[initial_ind:final_ind]
  
    j_sf_peaks, _ = find_peaks(j_sf, height = 200e-6)
    j_jtl_peaks, _ = find_peaks(j_jtl, height = 200e-6)
    j_si_peaks, _ = find_peaks(j_si, height = 200e-6)
   
    # find inter-fluxon intervals and fluxon generation rates for each JJ
    I_si = data_dict['L3#branch']
    # I_si = I_si[initial_ind:final_ind]
    I_drive = data_dict['L0#branch']
    # I_drive = I_drive[initial_ind:final_ind]            
    
    j_sf_ifi = np.diff(time_vec[j_sf_peaks])
    j_jtl_ifi = np.diff(time_vec[j_jtl_peaks])
    j_si_ifi = np.diff(time_vec[j_si_peaks])
    
    j_sf_rate = 1/j_sf_ifi
    j_jtl_rate = 1/j_jtl_ifi
    j_si_rate = 1/j_si_ifi
    
    j_sf_ifi_array.append(j_sf_ifi)
    j_sf_rate_array.append(j_sf_rate)
    j_jtl_ifi_array.append(j_jtl_ifi)
    j_jtl_rate_array.append(j_jtl_rate)
    j_si_ifi_array.append(j_si_ifi)
    j_si_rate_array.append(j_si_rate)
    
    I_si_array.append(I_si[j_si_peaks])
    
    
#%% assemble data and change units
I_si_pad = 100e-9 # amount above the observed max of Isi that the simulation will allow before giving a zero rate

master_rate_array = []
I_si_array__scaled = []
    
for ii in range(num_drives):
                
    temp_rate_vec = np.zeros([len(j_si_rate_array[ii])])
    temp_I_si_vec = np.zeros([len(j_si_rate_array[ii])])
    for jj in range(len(j_si_rate_array[ii])):
        temp_rate_vec[jj] = 1e-6*j_si_rate_array[ii][jj] # master_rate_array has units of fluxons per microsecond
        temp_I_si_vec[jj] = 1e6*( I_si_array[ii][jj]+I_si_array[ii][jj+1] )/2 # this makes I_si_array__scaled have the same dimensions as j_si_rate_array and units of uA

    master_rate_array.append([])
    master_rate_array[ii] = np.append(temp_rate_vec,0) # this zero is added so that a current that rounds to I_si + I_si_pad will give zero rate
    I_si_array__scaled.append([])
    I_si_array__scaled[ii] = np.append(temp_I_si_vec,np.max(temp_I_si_vec)+I_si_pad) # this additional I_si + I_si_pad is included so that a current that rounds to I_si + I_si_pad will give zero rate


#%% plot the rate vectors 

plot_syn_rate_array(I_si_array__scaled,master_rate_array,I_drive_list)
# plot_syn_rate_array__waterfall(I_si_array__scaled,master_rate_array,I_drive_list)


#%% surface plot
dI_si = 0.1 # units of uA

I_si_max = 0
for ii in range(num_drives):
    I_si_max = np.max([I_si_max,np.max(I_si_array__scaled[ii][:])])

I_si_plotting_vec = np.arange(0,I_si_max+dI_si,dI_si)
num_I_si = len(I_si_plotting_vec)
I_si_plotting_mat = np.zeros([num_drives,num_I_si])
rate_plotting_mat = np.zeros([num_drives,num_I_si])
for ii in range(num_drives):
    num_I_si__this = len(I_si_array[ii])
    for jj in range(num_I_si__this):
        ind = (np.abs(I_si_plotting_vec[:]-I_si_array__scaled[ii][jj])).argmin()
        I_si_plotting_mat[ii,ind] = I_si_plotting_vec[ind]        
        rate_plotting_mat[ii,ind] = 1e3*master_rate_array[ii][jj]

I_drive_plotting_mat = np.matlib.repmat(np.asarray(I_drive_list),num_I_si,1).transpose()


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(I_si_plotting_mat, I_drive_plotting_mat, rate_plotting_mat, cmap=cm.viridis,linewidth=0, antialiased=False)
ax.set_xlabel('$I_{si}$ [$\mu$A]')
ax.set_ylabel('$I_{drive}$ [$\mu$A]')
ax.set_zlabel('Rate of fluxon generation [kilofluxons per $\mu$s]')

plt.show()

#%% color plot
I_si_max = 0
for ii in range(num_drives):
    I_si_max = np.max([I_si_max,np.max(I_si_array__scaled[ii][:])])
    
fig, ax = plt.subplots(1,1)
rates = ax.imshow(rate_plotting_mat.transpose(), cmap = plt.cm.viridis, interpolation='none', extent=[I_drive_list[0],I_drive_list[-1],0,I_si_max], aspect = 'auto', origin = 'lower')
cbar = fig.colorbar(rates, extend='both')
cbar.minorticks_on()     
fig.suptitle('Rate [kilofluxons per $\mu$s]')
# plt.title(title_string)
ax.set_xlabel(r'$I_{drive}$ [$\mu$A]')
ax.set_ylabel(r'$I_{si}$ [$\mu$A]')  
ax.set_ylim() 
plt.show()      
# fig.savefig('figures/'+save_str+'__log.png')


#%% save the data    
 
save_string = 'master__syn__rate_array__3jj__Isipad{:04.0f}nA'.format(I_si_pad*1e9)
data_array = dict()
data_array['rate_array'] = master_rate_array
data_array['I_drive_list'] = I_drive_list#[x*1e-6 for x in I_drive_vec]
data_array['I_si_array'] = I_si_array__scaled
print('\n\nsaving session data ...')
save_session_data(data_array,save_string)


#%% load test

if 1 == 1:
    
    with open('../_circuit_data/master__syn__rate_array__3jj__Isipad0010nA.soen', 'rb') as data_file:         
            data_array_imprt = pickle.load(data_file)
    # data_array_imported = load_session_data('session_data__master_rate_matrix__syn__2020-04-24_10-24-23.dat')
    I_si_array__imprt = data_array_imprt['I_si_array']
    I_drive_list__imprt = data_array_imprt['I_drive_list']
    rate_array__imprt = data_array_imprt['rate_array']
    
    I_drive_sought = 14.45
    I_drive_sought_ind = (np.abs(np.asarray(I_drive_list__imprt)-I_drive_sought)).argmin()
    I_si_sought = 4.552
    I_si_sought_ind = (np.abs(np.asarray(I_si_array__imprt[I_drive_sought_ind])-I_si_sought)).argmin()
    rate_obtained = rate_array__imprt[I_drive_sought_ind][I_si_sought_ind]
    
    print('I_drive_sought = {:7.4f}uA, I_drive_sought_ind = {:d}\nI_si_sought = {:7.4f}uA, I_si_sought_ind = {:d}\nrate_obtained = {:10.4f} fluxons per us'.format(I_drive_sought,I_drive_sought_ind,I_si_sought,I_si_sought_ind,rate_obtained))

