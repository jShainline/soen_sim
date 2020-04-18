#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_error_mat, plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages
from _functions import Ljj, dendrite_current_splitting, read_wr_data
from _functions__more import dendrite_model__parameter_sweep
from util import physical_constants
p = physical_constants()

plt.close('all')

#%% compare initial currents in dendrite circuit

I_drive = 0

Ic = 40e-6
I_b = [71.5e-6,36e-6,35e-6]
Lm2 = 10e-12
M_direct = np.sqrt(200e-12*Lm2)
Ldr1 = 10e-12
Ldr2 = 26e-12
L1 = 200e-12
L2 = 77.5e-12
L3 = 7.75e-9
Idr1_prev = 31e-6
Idr2_prev = 31e-6
Ij2_prev = 34e-6
Ij3_prev = 36e-6

load_wr = True

for ii in range(5):
    Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,I_drive,I_b,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
    Idr1_prev = Idr1_next
    Idr2_prev = Idr2_next
    Ij2_prev = Ij2_next
    Ij3_prev = Ij3_next
 
Idr1_soen = Idr1_next
Idr2_soen = Idr2_next
Ij2_soen = Ij2_next
Ij3_soen = Ij3_next

if load_wr == True:
    directory = 'wrspice_data'
    file_name = 'dendrite_init_currents.dat'
    # data_to_plot = ['L3#branch','L4#branch','L8#branch','v(3)','v(4)','v(5)']#'L0#branch','L1#branch','L2#branch',
    # plot_save_string = False
    
    data_dict = read_wr_data(directory+'/'+file_name)
    data_dict['file_name'] = file_name
    time_vec = data_dict['time']
# plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)

initial_ind = (np.abs(time_vec-0.5e-9)).argmin()

Idr1_wr = data_dict['L0#branch']
Idr2_wr = data_dict['L6#branch']
Ij2_wr = data_dict['L7#branch']
Ij3_wr = data_dict['L8#branch']
I1_wr = data_dict['L1#branch']
I2_wr = data_dict['L2#branch']
I3_wr = data_dict['L3#branch']
    
print('\n\nIdr1_soen = {:2.4f}uA, Idr1_wr = {:2.4f}uA; Idr1_soen-Idr1_wr = {:3.0f}nA'.format(Idr1_next*1e6,Idr1_wr[initial_ind]*1e6,(Idr1_next-Idr1_wr[initial_ind])*1e9))
print('Idr2_soen = {:2.4f}uA, Idr2_wr = {:2.4f}uA; Idr2_soen-Idr2_wr = {:3.0f}nA'.format(Idr2_next*1e6,Idr2_wr[initial_ind]*1e6,(Idr2_next-Idr2_wr[initial_ind])*1e9))
print('Ij2_soen = {:2.4f}uA, Ij2_wr = {:2.4f}uA; Ij2_soen-Ij2_wr = {:3.0f}nA'.format(Ij2_next*1e6,Ij2_wr[initial_ind]*1e6,(Ij2_next-Ij2_wr[initial_ind])*1e9))
print('Ij3_soen = {:2.4f}uA, Ij3_wr = {:2.4f}uA; Ij3_soen-Ij3_wr = {:3.0f}nA'.format(Ij3_next*1e6,Ij3_wr[initial_ind]*1e6,(Ij3_next-Ij3_wr[initial_ind])*1e9))
print('I1_soen = {:2.4f}uA, I1_wr = {:2.4f}uA; I1_soen-I1_wr = {:3.0f}nA'.format(I1*1e6,I1_wr[initial_ind]*1e6,(I1-I1_wr[initial_ind])*1e9))
print('I2_soen = {:2.4f}uA, I2_wr = {:2.4f}uA; I2_soen-I2_wr = {:3.0f}nA'.format(I2*1e6,I2_wr[initial_ind]*1e6,(I2-I2_wr[initial_ind])*1e9))
print('I3_soen = {:2.4f}uA, I3_wr = {:2.4f}uA; I3_soen-I3_wr = {:3.0f}nA'.format(I3*1e6,I3_wr[initial_ind]*1e6,(I3-I3_wr[initial_ind])*1e9))

#%% calculate Idr2 vs Iflux

Iflux_vec = np.linspace(18e-6,30e-6,1000)
Idr2_vec = np.zeros([len(Iflux_vec)])

for jj in range(len(Idr2_vec)):
    for ii in range(20):
        Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,Iflux_vec[jj],I_b,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
        Idr1_prev = Idr1_next
        Idr2_prev = Idr2_next
        Ij2_prev = Ij2_next
        Ij3_prev = Ij3_next
        if jj == 20:
            print('ii = {:d}, Idr2 = {:2.4f}uA'.format(ii,Idr2_next*1e6))
    Idr2_vec[jj] = Idr2_next
 
Idr2_fit = np.polyfit(Iflux_vec,Idr2_vec,1)
Iflux_vec_dense = np.linspace(0,Iflux_vec[-1],1000)
Idr2_vec_dense = np.polyval(Idr2_fit,Iflux_vec_dense)
threshold_ind = (np.abs(Idr2_vec_dense-Ic)).argmin()
Iflux_threshold = Iflux_vec_dense[threshold_ind]   
 
fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
ax.plot(Iflux_vec*1e6,Idr2_vec*1e6, '-', label = 'data') 
ax.plot(Iflux_vec_dense*1e6,Idr2_vec_dense*1e6, '-', label = 'fit: Idr2 = {:1.5f}*Iflux+{:1.5f}x1e-6'.format(Idr2_fit[0],Idr2_fit[1]*1e6)) 
ax.plot([Iflux_threshold*1e6,Iflux_threshold*1e6], [0,Ic*1e6], '-.', label = 'Iflux at threshold = {}uA'.format(Iflux_threshold*1e6))      
ax.set_xlabel(r'$I_{flux}$ [$\mu$A]')
ax.set_ylabel(r'$I_{dr2}$ [$\mu$A]')
ax.legend()
plt.show()


#%% compare currents associated with fluxons

data_to_plot = ['L1#branch','L2#branch','L3#branch','v(4)','v(5)','v(6)']
plot_save_string = False
# plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)

first_pulse_ind = (np.abs(time_vec-5e-9)).argmin()
second_pulse_ind = (np.abs(time_vec-7e-9)).argmin()
between_pulses_ind = (np.abs(time_vec-6e-9)).argmin()

I_j_df = data_dict['L1#branch']
I_j_2 = data_dict['L2#branch']
I_j_3 = data_dict['L3#branch']
peaks_I_j_df, _ = find_peaks(I_j_df, height = 10e-6, distance = 20)
peaks_I_j_2, _ = find_peaks(I_j_2, height = 10e-6, distance = 20)
peaks_I_j_3, _ = find_peaks(I_j_3, height = 100e-9, distance = 20)
# plot_fq_peaks(time_vec,I_j_df,peaks_I_j_df)

I_j_df__init = I_j_df[initial_ind]
I_j_df__peaks = I_j_df[peaks_I_j_df]
I_j_df__fluxon_wr = I_j_df__peaks[0]-I_j_df__init

I_j_2__init = I_j_2[initial_ind]
I_j_2__peaks = I_j_2[peaks_I_j_2]
I_j_2__fluxon_wr = I_j_2__peaks[0]-I_j_2__init

I_j_3__init = I_j_3[initial_ind]
I_j_3__peaks = I_j_3[peaks_I_j_3]
I_j_3__fluxon_wr = I_j_3__peaks[0]-I_j_3__init

Ljdr1 = Ljj(Ic,Ic)
Ljdr2 = Ljj(Ic,Ic)
Lj2 = Ljj(Ic,Ic)
Lj3 = Ljj(Ic,Ic)
        
L_ppp = Lj3*L3/(Lj3+L3)
L_pp = L2+L_ppp
L_p = Lj2*L_pp/(Lj2+L_pp)

L_qqq = Ldr1+Lm2+Ljdr1
L_qq = Ldr2+Ljdr2
L_q = L_qq*L_qqq/(L_qq+L_qqq)
        
Phi0 = p['Phi0']
# I_j_df_fluxon_soen = Phi0/(L1+Ldr2+L_p)
# I_j_df_fluxon_soen = Phi0/(L1+L_p+L_q)
I_j_df_fluxon_soen = Phi0/(L1+Ldr2+Ljdr2+Lj2)
I_j_2_fluxon_soen = Phi0/(Lj2+L_pp)
I_j_3_fluxon_soen = Phi0/(L3+Lj3)
# print('Phi0/I_j_df_fluxon_wr = {:3.4f}pH'.format(1e12*Phi0/I_j_df__fluxon_wr))
# print('L1+Ljdr2+L_p = {}pH'.format(1e12*(L1+Ljdr2+L_p)))
print('I_j_df__fluxon_wr = {:2.4f}uA, I_j_df_fluxon_soen = {:2.4f}uA; soen - wr = {:2.4f}uA ({:2.4f}%)'.format(I_j_df__fluxon_wr*1e6,I_j_df_fluxon_soen*1e6,(I_j_df_fluxon_soen-I_j_df__fluxon_wr)*1e6,100*(I_j_df_fluxon_soen-I_j_df__fluxon_wr)/I_j_df__fluxon_wr))
print('I_j_2__fluxon_wr = {:2.4f}uA, I_j_2_fluxon_soen = {:2.4f}uA; soen - wr = {:2.4f}uA ({:2.4f}%)'.format(I_j_2__fluxon_wr*1e6,I_j_2_fluxon_soen*1e6,(I_j_2_fluxon_soen-I_j_2__fluxon_wr)*1e6,100*(I_j_2_fluxon_soen-I_j_2__fluxon_wr)/I_j_2__fluxon_wr))
print('I_j_3__fluxon_wr = {:2.4f}uA, I_j_3_fluxon_soen = {:2.4f}uA; soen - wr = {:2.4f}uA ({:2.4f}%)'.format(I_j_3__fluxon_wr*1e6,I_j_3_fluxon_soen*1e6,(I_j_3_fluxon_soen-I_j_3__fluxon_wr)*1e6,100*(I_j_3_fluxon_soen-I_j_3__fluxon_wr)/I_j_3__fluxon_wr))

#%% compare currents added to three loops
# I_1 = data_dict['L1#branch']
# I_2 = data_dict['L2#branch']
# I_3 = data_dict['L3#branch']

# delta_I_loop1_wr = I1_wr[between_pulses_ind] - I1_wr[initial_ind]
# delta_I_loop2_wr = I2_wr[between_pulses_ind] - I2_wr[initial_ind]
# delta_I_loop3_wr = I3_wr[between_pulses_ind] - I3_wr[initial_ind]

# delta_I_fluxon_loop3_wr = 


