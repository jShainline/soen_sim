import numpy as np
from matplotlib import pyplot as plt
from pylab import *
from util import physical_constants


#%%
dI = 0.01
I_sy_vec = np.arange(20,40+dI,dI)

d = 2.257804
e = -76.01606
f = 583.005805

#%%
a = 1
c = 1/d
dI_1fq_vec = np.zeros([len(I_sy_vec),2])
for ii in range(len(I_sy_vec)):
    b = -(2*I_sy_vec[ii]+e/d)
    dI_1fq_vec[ii,0] = (-b+np.sqrt(b**2-4*a*c) ) / (2*a)
    dI_1fq_vec[ii,1] = (-b-np.sqrt(b**2-4*a*c) ) / (2*a)

#%%
fig, ax = plt.subplots(1,1)
# ax.plot(I_sy_vec[:],dI_1fq_vec[:,0]*1e3)
ax.plot(I_sy_vec[:],dI_1fq_vec[:,1]*1e3)
ax.set_xlabel(r'$I_{sy}$ [$\mu$A]')
ax.set_ylabel(r'$\Delta Isy_{1fq}$ [nA]')
plt.show()

#%%
p = physical_constants()

L = 100e-12
T = 4.2
I_noise = np.sqrt(2*p['kB']*T/L)
print('I_noise = {}uA'.format(I_noise*1e6))