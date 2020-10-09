import numpy as np

#%%
tau_min = 20e-9
tau_max = 30e-9
r_jj = 4.125

#%%
r_spd = r_jj/((tau_max/tau_min)-1)
L = tau_max*r_spd
print('r_spd = {:4.4f}ohm, L = {:4.4f}nH'.format(r_spd,L*1e9))
