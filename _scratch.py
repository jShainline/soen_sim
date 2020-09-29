import numpy as np
from util import color_dictionary, physical_constants
colors = color_dictionary()

p = physical_constants()

#%%
C = 1e-12
tau1 = 1e-9
tau2 = 20e-12

L = tau1*tau2/C
R = 2*np.sqrt(L/C)

print('L = {:f}nH\nR = {:f}ohm'.format(L*1e9,R))

#%%
R = 0.1592
tau1 = 20e-12
C = tau1/R
print(C)

#%%
M1 = np.sqrt(20e-12*20e-12)
I1 = 45e-6
I2 = 20e-6
M2 = M1*I1/I2
print(M2)

#%%

M = np.sqrt(200*20)
a = p['Phi0__pH_ns']
N = 500

dI = a/(M*N)
print(dI)
