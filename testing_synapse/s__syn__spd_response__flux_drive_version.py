#%%
import numpy as np
from matplotlib import pyplot as plt

from _functions import read_wr_data, chi_squared_error

from util import color_dictionary, physical_constants
colors = color_dictionary()
p = physical_constants()

plt.close('all')

#%%
dt = 100e-12
tf = 300e-9
t0 = 200e-12
time_vec = np.arange(0,tf+dt,dt)

L_spd1 = 100e-9
L_spd2 = 100e-9
L_1 = 6e-9
L_tot = L_spd1+L_spd2+L_1

r_spd1_0 = 5e3
r_spd2 = L_tot/50e-9

I_spd = 3e-6

tau_plus = L_tot/(r_spd1_0+r_spd2)
tau_minus = L_tot/r_spd2

I_0 = I_spd*(r_spd1_0/(r_spd1_0+r_spd2))*(1-np.exp(-t0/tau_plus))

#%%
I_spd2 = np.zeros([len(time_vec)])
for ii in range(len(time_vec)):
    _pt = time_vec[ii]
    if _pt <= t0:
        I_spd2[ii] = I_spd*(r_spd1_0/(r_spd1_0+r_spd2))*(1-np.exp(-_pt/tau_plus))
    elif _pt > t0:
        I_spd2[ii] = I_0*np.exp(-(_pt-t0)/tau_minus)

#%% load wr
directory = 'wrspice_data'
file_name = 'syn_0jj.dat'
data_dict = read_wr_data(directory+'/'+file_name)

time_vec_wr = data_dict['time']
initial_ind = (np.abs(time_vec_wr-1.0e-9)).argmin()
time_vec_wr = time_vec_wr[initial_ind:]-time_vec_wr[initial_ind]
I_spd2_wr = data_dict['L0#branch']
I_spd2_wr = I_spd2_wr[initial_ind:]

actual_data = np.vstack((time_vec[:],I_spd2[:]))
target_data = np.vstack((time_vec_wr[:],I_spd2_wr[:]))

error = chi_squared_error(target_data,actual_data)
# error = 1

#%% plot

fig = plt.figure()
plt.title('error = {:6.4e}; r_spd2 = {}'.format(error, r_spd2))    
ax = fig.gca()

color_list = [colors['blue3'],colors['red3'],colors['green3'],colors['yellow3']]
ax.plot(time_vec*1e9,I_spd2*1e6, '-', color = colors['blue3'] , label = 'analytical') #
ax.plot(time_vec_wr*1e9,I_spd2_wr*1e6, '-', color = colors['red3'] , label = 'spice') #

ax.set_xlabel(r'Time [ns]')
ax.set_ylabel(r'$I_{spd2}$ [$\mu$A]')

ax.legend()
# ax.grid(True,which='both')

plt.show()
