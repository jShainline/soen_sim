import numpy as np
from matplotlib import pyplot as plt

from _functions import Ljj_pH as Ljj
from _functions import ind_in_par as ip
from util import physical_constants, color_dictionary

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

Isy1 = 0
Isy2 = 0

Ia_desired = 35 # desired current in each branch of DR squid loop
I6_desired = 35 # desired current in final jj in DI loop
I4_desired = 35 # desired current in jj in JTL

rb1 = 7.2e6 # right-most
rb2 = 800e3 # middle
rb3 = 1.9e6 # left-most

L1 = 20 # inductance of each branch of DR squid loop
L2 = 47.5 # next inductor over
L3 = 47.5 # next one
L4 = 77.5e3 # DI loop
L5 = 10 # driver side of DC MIs
L6 = 10 # receiver side of DC MIs
L7 = 200 # driver side of left plasticity MI
L8 = 20 # receiver side of left plasticity MI
L9 = 200 # driver side of right plasticity MI
L10 = 20 # receiver side of right plasticity MI
L11 = 0 # output of DI loop

Ic = 40 # uA

Vb = 1e9

#%% calculated quantities
M1 = np.sqrt(L5*L6)
M2 = np.sqrt(L7*L8)
M3 = np.sqrt(L9*L10)

Lb = L2+L8
Lc = L3+L10

#%% iterate
Ia = 0
I4 = 0
I6 = 0

Ib1 = Vb/rb1
Ib2 = Vb/rb2
Ib3 = Vb/rb3

I7 = Ib1

num_it = 20
I6_vec = np.zeros([num_it])
I4_vec = np.zeros([num_it])
Ia_vec = np.zeros([num_it])
for ii in range(num_it):
    
    print('ii = {} of {}'.format(ii+1,num_it))
    
    La = (L1+Ljj(Ic,Ia))/2
    Ld = L6+Ljj(Ic,I4)
    Le = L6+Ljj(Ic,I6)
    Lf = Le+Lc+Ld
    
    alpha = 2*(1+Lb/Ld-Ld/Lf)-La/Ld
    
    Ia = ( -M1*(1/Lf+1/Ld)*Ib3 + ( 1-((Lc+Ld)/Lf) )*Ib1 + (M1/Lf)*Ib2 - (M2/Ld)*Isy1 + (M3/Lf)*Isy2 )/alpha
    Ia_vec[ii] = Ia
    
    I6 = Ib1 + (M1/Ld)*Ib3 + (M2/Ld)*Isy1 - ( 2*(1+(Lb/Ld))-La/Ld )*Ia
    I6_vec[ii] = I6
    
    I4 = Ib1-I6-2*Ia
    I4_vec[ii] = I4    
    
print('I7 = {}uA; I6 = {}uA; I4 = {}uA; Ia = {}uA'.format(I7,I6,I4,Ia))    
    
#%% plot
iteration_vec = np.arange(1,num_it+1,1)
fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('I7 = {:9.4f}uA; I6 = {:9.4f}uA; I4 = {:9.4f}uA; Ia = {:9.4f}uA'.format(I7,I6,I4,Ia))

axs[0].plot(iteration_vec,I6_vec, '-', color = colors['blue3'], label = 'I6') 
axs[0].set_ylabel(r'I6 [$\mu$A]') 

axs[1].plot(iteration_vec,I4_vec, '-', color = colors['blue3'], label = 'I7') 
axs[1].set_ylabel(r'I4 [$\mu$A]')

axs[2].plot(iteration_vec,Ia_vec, '-', color = colors['blue3'], label = 'I6') 
axs[2].set_ylabel(r'Ia [$\mu$A]')

axs[2].set_xlabel(r'Iteration')


# fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)   
# fig.suptitle('I7 = {}uA; I6 = {}uA; I4 = {}uA; Ia = {}uA'.format(I7,I6,I4,Ia))

# axs[0].plot(iteration_vec,np.abs( (I6_vec-I6_vec[-1])/I6_vec[-1] ), '-', color = colors['blue3'], label = 'I6') 
# axs[0].set_ylabel(r'$\Delta I6/I6[-1]$') 

# axs[1].plot(iteration_vec,np.abs( (I4_vec-I4_vec[-1])/I4_vec[-1] ), '-', color = colors['blue3'], label = 'I7') 
# axs[1].set_ylabel(r'$\Delta I4/I4[-1]$')

# axs[2].plot(iteration_vec,np.abs( (Ia_vec-Ia_vec[-1])/Ia_vec[-1] ), '-', color = colors['blue3'], label = 'I6') 
# axs[2].set_ylabel(r'$\Delta Ia [$\mu$A]$')

# axs[2].set_xlabel(r'Iteration')
