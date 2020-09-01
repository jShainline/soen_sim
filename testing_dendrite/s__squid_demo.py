import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

from _functions import read_wr_data
from util import physical_constants, color_dictionary
p = physical_constants()
colors = color_dictionary()


#%%

I_de_vec = [45,50,55,60,65,70,75,80,85,90]
M = np.sqrt(200e-12*12.5e-12)

#%%

time_vec__array = []
time_vec_avg__array = []
V_fq__array = []
V_fq_peaks_1__array = []
V_fq_peaks_2__array = []
V_fq_avg__array = []
I_drive__array = []
I_drive_avg__array = []

directory_name = 'wrspice_data/'
for jj in range(len(I_de_vec)):
    
    print('jj = {} of {}'.format(jj+1,len(I_de_vec)))
    file_name = 'squid_demo__Ide{:5.2f}uA'.format(I_de_vec[jj])
    data_dict = read_wr_data('{}{}.dat'.format(directory_name,file_name))
    V_fq_str = 'v(3)'
    I_drive_str = 'L1#branch'
    time_vec = data_dict['time']
    ind0 = (np.abs(time_vec-1e-9)).argmin()
    time_vec = time_vec[ind0:]
    dt = time_vec[2]-time_vec[1]
    
    V_fq = data_dict[V_fq_str]
    V_fq = V_fq[ind0:]
    I_drive = data_dict[I_drive_str]
    I_drive = I_drive[ind0:]
        
    V_fq_peaks, _ = find_peaks(V_fq, height = 100e-6) # , prominence = 5e-6 #  #  # , distance = np.round(10e-12/dt)
    V_fq_peaks_1 = V_fq_peaks[0::2]
    V_fq_peaks_2 = V_fq_peaks[1::2]

    time_vec_avg = np.zeros([len(V_fq_peaks_1)-1])
    V_fq_avg = np.zeros([len(V_fq_peaks_1)-1])
    I_drive_avg = np.zeros([len(V_fq_peaks_1)-1])
    for ii in range(len(V_fq_peaks_1)-1):
        time_vec_avg[ii] = time_vec[V_fq_peaks_1[ii]]+(time_vec[V_fq_peaks_1[ii+1]]-time_vec[V_fq_peaks_1[ii]])/2
        V_fq_avg[ii] = np.sum(V_fq[V_fq_peaks_1[ii]:V_fq_peaks_1[ii+1]])/(V_fq_peaks_1[ii+1]-V_fq_peaks_1[ii])
        I_drive_avg[ii] = I_drive[V_fq_peaks_1[ii]]+(I_drive[V_fq_peaks_1[ii+1]]-I_drive[V_fq_peaks_1[ii]])/2
    
    window_size = 5
    polynomial_order = 3
    V_fq_avg = savgol_filter(V_fq_avg, window_size, polynomial_order) # polynomial order 3
    
    time_vec_avg = np.insert(time_vec_avg,0,time_vec[V_fq_peaks_1[0]])
    I_drive_avg = np.insert(I_drive_avg,0,I_drive[V_fq_peaks_1[0]])
    V_fq_avg = np.insert(V_fq_avg,0,0)
    
    
    time_vec_avg = np.insert(time_vec_avg,0,time_vec[0])
    I_drive_avg = np.insert(I_drive_avg,0,I_drive[0])
    V_fq_avg = np.insert(V_fq_avg,0,0)
    
    time_vec_avg = np.append(time_vec_avg,time_vec[V_fq_peaks_1[-1]])
    I_drive_avg = np.append(I_drive_avg,I_drive[V_fq_peaks_1[-1]])
    V_fq_avg = np.append(V_fq_avg,0)
    
    time_vec_avg = np.append(time_vec_avg,time_vec[-1])
    I_drive_avg = np.append(I_drive_avg,I_drive[-1])
    V_fq_avg = np.append(V_fq_avg,0)    
    
    time_vec__array.append(time_vec)
    time_vec_avg__array.append(time_vec_avg)
    V_fq__array.append(V_fq)
    V_fq_peaks_1__array.append(V_fq_peaks_1)
    V_fq_peaks_2__array.append(V_fq_peaks_2)
    V_fq_avg__array.append(V_fq_avg)
    I_drive__array.append(I_drive)
    I_drive_avg__array.append(I_drive_avg)
    
    
#%%    
ind1 = 6
fig, axs = plt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)   

axs.plot(np.asarray(time_vec__array[ind1])*1e9,np.asarray(V_fq__array[ind1])*1e6, '-', color = colors['blue3'], label = '$V_{sq}$ [$\mu$V]')
axs.plot(np.asarray(time_vec__array[ind1][V_fq_peaks_1__array[ind1]])*1e9,np.asarray(V_fq__array[ind1][V_fq_peaks_1__array[ind1]])*1e6, 'x', color = colors['red3'], label = '$V_{jj1}$ [$\mu$V]')
axs.plot(np.asarray(time_vec__array[ind1][V_fq_peaks_2__array[ind1]])*1e9,np.asarray(V_fq__array[ind1][V_fq_peaks_2__array[ind1]])*1e6, 'x', color = colors['yellow3'], label = '$V_{jj2}$ [$\mu$V]')
axs.plot(time_vec_avg__array[ind1]*1e9,V_fq_avg__array[ind1]*1e6, '-', color = colors['blue5'], label = '$V_{sq_avg}$ [$\mu$V]')
axs.legend()

axs.set_xlabel(r'Time [ps]')
axs.set_ylabel(r'$V_{fq}$ [$\mu$V]')

ax2 = axs.twinx()
ax2.plot(np.asarray(time_vec__array[ind1])*1e9,np.asarray(I_drive__array[ind1])*M/p['Phi0'], '-', color = colors['green3'])
ax2.set_ylabel('$\Phi_a/\Phi_0$', color = colors['green3'])

plt.show()

#%%    
ind1 = 6
fig, axs = plt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)   

t1 = 60001
axs.plot(np.asarray(time_vec__array[ind1])*1e12-t1,np.asarray(V_fq__array[ind1])*1e6, '-', color = colors['blue3'], label = '$V_{sq}$')
axs.plot(np.asarray(time_vec__array[ind1][V_fq_peaks_1__array[ind1]])*1e12-t1,np.asarray(V_fq__array[ind1][V_fq_peaks_1__array[ind1]])*1e6, 'x', color = colors['red3']) # , label = '$V_{jj1}$ [$\mu$V]'
axs.plot(np.asarray(time_vec__array[ind1][V_fq_peaks_2__array[ind1]])*1e12-t1,np.asarray(V_fq__array[ind1][V_fq_peaks_2__array[ind1]])*1e6, 'x', color = colors['yellow3']) #, label = '$V_{jj2}$ [$\mu$V]'
axs.plot(time_vec_avg__array[ind1]*1e12-t1,V_fq_avg__array[ind1]*1e6, '-', color = colors['green3'], label = '$V_{sq_avg}$')
axs.legend()

axs.set_xlabel(r'Time [ps]')
axs.set_ylabel(r'$V_{fq}$ [$\mu$V]')

axs.set_xlim([0,100])
axs.set_ylim([20,165])
        
plt.show()

#%%
fig, axs = plt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)  

color_list = ['green1','green2','green3','green4','green5','blue5','blue4','blue3','blue2','blue1']
for ii in range(len(I_de_vec)):
    axs.plot(M*I_drive_avg__array[ii]/p['Phi0'],V_fq_avg__array[ii]*1e6, '-', color = colors[color_list[ii]], label = 'I_de = {:5.2f}uA'.format(I_de_vec[ii]))
axs.set_xlabel(r'$\Phi_a/\Phi_0$')
axs.set_ylabel(r'$V_{fq}$ [$\mu$V]', color = colors['green3'])

color_list = ['red1','red2','red3','red4','red5','yellow5','yellow4','yellow3','yellow2','yellow1']    
ax2 = axs.twinx()
for ii in range(len(I_de_vec)):
    ax2.plot(M*I_drive_avg__array[ii]/p['Phi0'],1e-9*V_fq_avg__array[ii]/p['Phi0'], '-', color = colors[color_list[ii]], label = 'I_de = {:5.2f}uA'.format(I_de_vec[ii]))
ax2.set_ylabel('Rate [fluxons per ns]', color = colors['red3'])  


plt.show()

#%%
# plot_dend_rate_array__norm_to_phi0(I_di_array = I_di_array__scaled, I_drive_list = I_drive_list, influx_list = influx_list, master_rate_array = master_rate_array, L_left = L_left_list[pp], I_de = I_de_list[qq])

