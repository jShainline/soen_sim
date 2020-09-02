import numpy as np
from util import physical_constants

p = physical_constants()

#%%
Phi_vec = np.linspace(0,p['Phi0']/2,50)
M = np.sqrt(200e-12*20e-12)
I_drive_vec = Phi_vec/M
resolution = 10e-9
I_drive_vec_round = np.round((Phi_vec/M)/resolution)*resolution

#%%
print('min(I_drive_vec) = {}uA; max(I_drive_vec = {}uA'.format(np.min(I_drive_vec)*1e6,np.max(I_drive_vec)*1e6))