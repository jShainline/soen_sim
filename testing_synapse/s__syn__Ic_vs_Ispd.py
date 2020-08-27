#%%
import numpy as np
from matplotlib import pyplot as plt

from util import color_dictionary, physical_constants
colors = color_dictionary()
p = physical_constants()

plt.close('all')

#%%

beta_L = 2
I_spd_vec = np.logspace(np.log10(1e-6),np.log10(30e-6),100)
L1_vec = [100e-12,200e-12,400e-12,800e-12]
k = 0.5 # mutual inductance coupling factor

Ic_mat = np.zeros([len(L1_vec),len(I_spd_vec)])

for ii in range(len(L1_vec)):
    for jj in range(len(I_spd_vec)):
        Ic_mat[ii,jj] = beta_L*( (k*I_spd_vec[jj])**2 ) * L1_vec[ii]/p['Phi0']

#%%

fig = plt.figure()
plt.title('Ic of SQUID JJs for beta_L = {} and $I_spd = \Phi_0/2M$ (assuming MI coupling factor k = {})'.format(beta_L,k))    
ax = fig.gca()

color_list = [colors['blue3'],colors['red3'],colors['green3'],colors['yellow3']]
for ii in range(len(L1_vec)):
    ax.loglog(I_spd_vec*1e6,Ic_mat[ii,:]*1e6, '-', color = color_list[ii], label = 'L_1 = {:2.0f} pH'.format(L1_vec[ii]*1e12))

ax.set_xlabel(r'$I_{spd}$ [$\mu$A]')
ax.set_ylabel(r'$I_c$ [$\mu$A]')

ax.legend()
ax.grid(True,which='both')

plt.show()