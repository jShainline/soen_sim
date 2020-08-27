#%%
import numpy as np
from matplotlib import pyplot as plt

from util import color_dictionary, physical_constants
colors = color_dictionary()
p = physical_constants()

plt.close('all')

#%%

I_spd_vec = np.logspace(-6,-5,100)
L2_vec = [10e-12,20e-12,40e-12,80e-12]
k = 0.5 # mutual inductance coupling factor

L1_mat = np.zeros([len(L2_vec),len(I_spd_vec)])

for ii in range(len(L2_vec)):
    for jj in range(len(I_spd_vec)):
        L1_mat[ii,jj] = (1/L2_vec[ii])*(p['Phi0']/(2*k*I_spd_vec[jj]))**2

#%%

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = 'Verdana'#'Computer Modern Sans Serif'
plt.rcParams['figure.figsize'] = [15,15/1.618]
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['figure.autolayout'] = True

fig = plt.figure()
plt.title('Inductance required to drive a dendrite with flux of $\Phi_0/2$ as a function of SPD bias current')    
ax = fig.gca()

color_list = [colors['blue3'],colors['red3'],colors['green3'],colors['yellow3']]
for ii in range(len(L2_vec)):
    ax.loglog(I_spd_vec*1e6,L1_mat[ii,:]*1e9, '-', color = color_list[ii], label = 'L_2 = {:2.0f} pH'.format(L2_vec[ii]*1e12))

ax.set_xlabel(r'$I_{spd}$ [$\mu$A]')
ax.set_ylabel(r'$L_1$ [nH]')

ax.legend()
ax.grid(True,which='both')

plt.show()