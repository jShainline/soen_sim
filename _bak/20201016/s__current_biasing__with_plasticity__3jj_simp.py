import numpy as np
from matplotlib import pyplot as plt

from _functions import Ljj_pH as Ljj
from _functions import ind_in_par as ip
from _util import physical_constants, color_dictionary

colors = color_dictionary()
p = physical_constants()

plt.close('all')

#%% 

# units of:
# pH
# uA
# ns
# mohm
# nV

#%% inputs

# rb1 = 11.111111e6 # 7.2e6 # right-most
rb2 = 2.2e6 # 800e3 # middle
# rb3 = 1.9e6 # left-most

L1 = 20 # inductance of each branch of DR squid loop
L2 = 47.5
L3 = 10
L4 = 10
L5 = 200
L6 = 20

Ic = 40 # uA

Vb = 1e9

#%% calculated quantities
M1 = np.sqrt(L3*L4)
M2 = -np.sqrt(L5*L6)

#%% iterate
Ia = 0
I3 = 0

Ib1 = 100 # Vb/rb1
Ib2 = 530
Isy = 0

num_it = 20
Ia_vec = np.zeros([num_it])
I3_vec = np.zeros([num_it])
for ii in range(num_it):
    
    # print('ii = {} of {}'.format(ii+1,num_it))
    
    La = L1+Ljj(Ic,Ia)
    Lb = L4+Ljj(Ic,I3)
    Lc = L2+L6
    Ld = 2*(Lb+Lc) + La
        
    Ia = (Lb/Ld)*Ib1 + (M1/Ld)*Ib2 - (M2/Ld)*Isy
    Ia_vec[ii] = Ia
    
    I3 = Ib1-2*Ia
    I3_vec[ii] = I3
       
    
print('Ib1 = {:9.4f}uA; Ib2 = {:9.4f}uA; Isy = {:9.4f}uA; I3 = {:9.4f}uA; Ia = {:9.4f}uA'.format(Ib1,Ib2,Isy,I3,Ia))    
    
#%% plot
iteration_vec = np.arange(1,num_it+1,1)
fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('Ib1 = {:9.4f}uA; Ib2 = {:9.4f}uA; I3 = {:9.4f}uA; Ia = {:9.4f}uA'.format(Ib1,Ib2,I3,Ia))

axs[0].plot(iteration_vec,I3_vec, '-', color = colors['blue3'], label = 'I6') 
axs[0].set_ylabel(r'I3 [$\mu$A]') 

axs[1].plot(iteration_vec,Ia_vec, '-', color = colors['blue3'], label = 'I6') 
axs[1].set_ylabel(r'Ia [$\mu$A]')

axs[1].set_xlabel(r'Iteration')


# fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)   
# fig.suptitle('I7 = {}uA; I6 = {}uA; I4 = {}uA; Ia = {}uA'.format(I7,I6,I4,Ia))

# axs[0].plot(iteration_vec,np.abs( (I6_vec-I6_vec[-1])/I6_vec[-1] ), '-', color = colors['blue3'], label = 'I6') 
# axs[0].set_ylabel(r'$\Delta I6/I6[-1]$') 

# axs[1].plot(iteration_vec,np.abs( (I4_vec-I4_vec[-1])/I4_vec[-1] ), '-', color = colors['blue3'], label = 'I7') 
# axs[1].set_ylabel(r'$\Delta I4/I4[-1]$')

# axs[2].plot(iteration_vec,np.abs( (Ia_vec-Ia_vec[-1])/Ia_vec[-1] ), '-', color = colors['blue3'], label = 'I6') 
# axs[2].set_ylabel(r'$\Delta Ia [$\mu$A]$')

# axs[2].set_xlabel(r'Iteration')
