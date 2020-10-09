import numpy as np
from matplotlib import pyplot as plt

from _functions import dendritic_drive__piecewise_linear, Vj, read_wr_data, Ljj
from util import color_dictionary, physical_constants
colors = color_dictionary()
p = physical_constants()

# plt.close('all')

#%%
#circuit on pg 24 of green lab notebook started 20200514

#%% 
# time_conversion = 1 # 1e6 # work in us
# inductance_conversion = 1 # 1e12 # work in pH
# current_conversion = 1 # 1e6 # work in uA
# voltage_conversion = inductance_conversion*current_conversion/time_conversion

#%% time

t0 = 0
tf = 100 # ns
dt = 100e-3 # ns

time_vec = np.arange(t0,tf+dt,dt)

#%% drive
I_bias = 35 # uA
L_drive = 200 # pH
L1 = 20 # pH
L2 = 20 # pH
r = 1 # pH ns

# pwl = [[0,0],[1e-9,0],[101e-9,20e-6],[102e-9,0]]
pwl = [[0,0],[1,0],[2,20],[7,20],[8,0]]
I_drive = dendritic_drive__piecewise_linear(time_vec,pwl)
M = -np.sqrt(L_drive*L1)
Lt = L1+L2

#%% jj
Ic = 40 # uA
rj = 6.25e3 # mOhm

Phi0 = 1e18*p['Phi0'] # uA pH

#%% step through time
I1 = np.zeros([len(time_vec)])
I1[0] = I_bias
I2 = np.zeros([len(time_vec)])

state = 'subthreshold'
for ii in range(len(time_vec)-1):
    
    # if ii < 100:
    # print('ii = {:d} of {:d}; time_vec[ii] = {:7.2f}ns; I1[ii] = {:5.2f}uA; I2[ii] = {:5.2f}uA; Ic = {:5.2f}uA'.format(ii+1,len(time_vec),time_vec[ii]*1e9,I1[ii]*1e6,I2[ii]*1e6,Ic*1e6))
    # if Vj(I_bias-I2[ii],Ic,r) > 0:
    #     print('I2[ii] = {:5.2f}uA; Ic = {:5.2f}uA; Vj = {}uV'.format(I2[ii]*1e6,Ic*1e6,Vj(I_bias-I2[ii],Ic,r)*1e6))
        
    # I2[ii+1] = (1-r*dt/Lt)*I2[ii] - ( M/Lt )*( I_drive[ii+1] - I_drive[ii] ) + ( dt/Lt )*Vj(I1[ii],Ic,rj)
    Ltt = Lt+Ljj(Ic,I1[ii])
    # if ii < 20:
    #     print('Ltt = {}pH'.format(Ltt*1e12))
    if state == 'subthreshold':
        V_j = 0
    elif state == 'spiking':
        # print('spiking')
        V_j = Phi0
        state = 'subthreshold'
    I2[ii+1] = (1-r*dt/Ltt)*I2[ii] + ( M/Ltt )*( I_drive[ii+1] - I_drive[ii] ) + ( 1/Ltt )*V_j
    I1[ii+1] = I_bias-I2[ii+1]
    if I1[ii+1] >= Ic:
        state = 'spiking'
    
#%% import WR

directory_name = 'wrspice_data/'
# file_name = 'ne__readout_test.dat'
file_name = 'ne__readout_test__sq_pls_01.dat'
data_dict = read_wr_data('{}{}'.format(directory_name,file_name))  
time_vec_wr = data_dict['time']
I1_wr = data_dict['L2#branch']
I2_wr = data_dict['L0#branch']
I_drive_wr = data_dict['L1#branch']

#%% plot
fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)   
# fig.suptitle('Synapse saturation vs Isy; tau_si = inf; L_si = {:7.4f} nH'.format(synapse_list[0].integration_loop_total_inductance*1e9)) 

axs[0].plot(time_vec_wr*1e9,I_drive_wr*1e6, '-', color = colors['red3'], label = 'wr _ drive')
axs[0].plot(time_vec,I_drive, '-.', color = colors['blue3'], label = 'model _ drive')
axs[0].set_ylabel(r'$I_{drive}$ [$\mu$A]')
axs[0].legend()

axs[1].plot(time_vec_wr*1e9,I1_wr*1e6, '-', color = colors['red3'], label = 'wr _ I1')
axs[1].plot(time_vec,I1, '-.', color = colors['blue3'], label = 'model _ I1')
axs[1].set_ylabel(r'$I_{1}$ [$\mu$A]')
axs[1].legend()

axs[2].plot(time_vec_wr*1e9,I2_wr*1e6, '-', color = colors['red3'], label = 'wr _ I2')
axs[2].plot(time_vec,I2, '-.', color = colors['blue3'], label = 'model _ I2')
axs[2].set_xlabel(r'Time [ns]')
axs[2].set_ylabel(r'$I_{2}$ [$\mu$A]')
axs[2].legend()
axs[2].set_xlim([t0,tf])

plt.show()