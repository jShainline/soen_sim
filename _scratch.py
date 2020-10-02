import numpy as np
from util import color_dictionary, physical_constants
from _functions import dendrite_current_splitting
colors = color_dictionary()

p = physical_constants()

#%%

dt = 1
tf = 100
time_vec = np.arange(0,tf+dt,dt)
state = 'off'
for ii in range(len(time_vec)):
    

#%%

def nTron(I_gate,I_on,I_off,r_gate,r_channel,state):
    
    if I_gate >= I_on:
        state = 'on'
    elif I_gate < I_on and I_gate >= I_off:
        if state == 'on':
            state = 'latched'
    elif I_gate < I_off:
        if state == 'on' or state == 'latched':
            state = 'off'
    
    if state == 'on' or state == 'latched':
        r1 = r_gate
        r2 = r_channel
    elif state == 'off':
        r1 = 0
        r2 = 0
            
    return r1, r2


#%%

#%% drive
I_bias = 35 # uA
L_drive = 200 # pH
L1 = 20 # pH
L2 = 20 # pH
r = 1 # pH ns

# pwl = [[0,0],[1e-9,0],[101e-9,20e-6],[102e-9,0]]
pwl = [[0,0],[1,0],[2,20],[7,20],[8,0]]
I_drive = dendritic_drive__piecewise_linear(time_vec,pwl)
M = -np.sqrt(L_drive*L1)
Lt = L1+L2

#%% jj
Ic = 40 # uA
rj = 6.25e3 # mOhm

Phi0 = 1e18*p['Phi0'] # uA pH

#%% step through time
I1 = np.zeros([len(time_vec)])
I1[0] = I_bias
I2 = np.zeros([len(time_vec)])

state = 'subthreshold'
for ii in range(len(time_vec)-1):
    
    # if ii < 100:
    # print('ii = {:d} of {:d}; time_vec[ii] = {:7.2f}ns; I1[ii] = {:5.2f}uA; I2[ii] = {:5.2f}uA; Ic = {:5.2f}uA'.format(ii+1,len(time_vec),time_vec[ii]*1e9,I1[ii]*1e6,I2[ii]*1e6,Ic*1e6))
    # if Vj(I_bias-I2[ii],Ic,r) > 0:
    #     print('I2[ii] = {:5.2f}uA; Ic = {:5.2f}uA; Vj = {}uV'.format(I2[ii]*1e6,Ic*1e6,Vj(I_bias-I2[ii],Ic,r)*1e6))
        
    # I2[ii+1] = (1-r*dt/Lt)*I2[ii] - ( M/Lt )*( I_drive[ii+1] - I_drive[ii] ) + ( dt/Lt )*Vj(I1[ii],Ic,rj)
    Ltt = Lt+Ljj(Ic,I1[ii])
    # if ii < 20:
    #     print('Ltt = {}pH'.format(Ltt*1e12))
    if state == 'subthreshold':
        V_j = 0
    elif state == 'spiking':
        # print('spiking')
        V_j = Phi0
        state = 'subthreshold'
    I2[ii+1] = (1-r*dt/Ltt)*I2[ii] + ( M/Ltt )*( I_drive[ii+1] - I_drive[ii] ) + ( 1/Ltt )*V_j
    I1[ii+1] = I_bias-I2[ii+1]
    if I1[ii+1] >= Ic:
        state = 'spiking'