#%%
import numpy as np
from _functions import omega_LRC

#%%
omega_r, omega_i = omega_LRC(10e-6,1,1e-9)

print('f = {:2.2f} MHz\nomega_r = {:2.2f} MHz\nomega_i = {:2.2f} MHz\nQ = {:f}'.format(omega_r*1e-6/(2*np.pi),omega_r*1e-6,omega_i*1e-6,omega_r/(2*omega_i)))

