#%%
import numpy as np

from util import physical_constants
p = physical_constants()

#%% set params
beta_c = 2
I_c = 10e-6
C = 5e-15*I_c/1e-6 #capacitace is known to be 5fF per uA Ic

#%% calculate values
R = np.sqrt(p['Phi0']*beta_c/(2*np.pi*I_c*C))
V = I_c*R

#%%
print('R = {:9.4f} ohm, V = {:9.4f} uV'.format(R,V*1e6))