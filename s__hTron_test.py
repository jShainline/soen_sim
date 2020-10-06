import numpy as np
from matplotlib import pyplot as plt
import time

from util import color_dictionary, physical_constants
from _functions import dendritic_drive__piecewise_linear, dendritic_drive__exp_pls_train__LR
colors = color_dictionary()

p = physical_constants()

plt.close('all')

#%% time parameters

dt = 0.1 # ns
tf = 100 # ns
time_vec = np.arange(0,tf+dt,dt)
_nt = len(time_vec)

#%% passive circuit parameters

# see pg 34-35 of green lab notebook started 20200514

L1 = 200 # pH
L2 = 200 # pH
L3 = 200 # pH
L4 = 10 # pH
L5 = 100e3 # pH

M = np.sqrt(L1*L2)
La = L2+L3
Lb = L4+L5

tau_nt = 10 # ns
r1 = La/tau_nt # mOhm

#%% hTron parameters

r_gate = 100e3 # mOhm
r_channel = 5e6 # mOhm

tau_nf = 20 # ns
r4 = Lb/tau_nf # mOhm

Ib = 20 # uA
I_gate_on = 3 # uA, hTron gate threshold current
I_gate_off = 0.1*I_gate_on # uA, hTron gate reset current
I_channel_on = 0.8*Ib # uA, hTron channel activation current
I_channel_off = 0.2*I_channel_on # uA, hTron channel reset current

#%% drive

isi = 50
# pwl = [[0,0],[1,0],[2,20],[7,20],[8,0],[1+isi,0],[2+isi,20],[7+isi,20],[8+isi,0]]
# I_d = dendritic_drive__piecewise_linear(time_vec,pwl)
    
Id = 20
drv_params = dict()
drv_params['t_r1_start'] = 5
drv_params['t_r1_rise'] = 0.1
drv_params['t_r1_pulse'] = 0.2
drv_params['t_r1_fall'] = 1
drv_params['t_r1_period'] = isi
drv_params['value_r1_off'] = 0
drv_params['value_r1_on'] = 5e6
drv_params['L1'] = 100e3
drv_params['L2'] = 100e3
drv_params['r2'] = 200e3/50
drv_params['Ib'] = Id
    
I_d = dendritic_drive__exp_pls_train__LR(time_vec,drv_params)

#%% ode loop
Lt = L3+L4
state_previous = 'off'
state_next = 'off'
I_1_vec = np.zeros([_nt])
I_3_vec = np.zeros([_nt])
ii_vec = np.arange(1,_nt-1,1)
for ii in ii_vec:

    I_gate = I_1_vec[ii]
    I_channel = Ib-I_3_vec[ii]
        
    if state_previous == 'off':
        if I_gate >= I_gate_on and I_channel > I_channel_on:
            state_next = 'on'
            print('ii = {} (t = {:7.0}ns); state = on from off'.format(ii,time_vec[ii]))            
    elif state_previous == 'on':
        if I_gate < I_gate_off and I_channel < I_channel_off:
            state_next = 'off'
    
    if state_next == 'on':
        r2 = r_gate
        r3 = r_channel
    elif state_next == 'off':
        r2 = 0
        r3 = 0
    
    ra = r1+r2
    rb = r3+r4       
    I_1_vec[ii+1] = (1-dt*(ra/La))*I_1_vec[ii] + (M/La)*( I_d[ii+1]-I_d[ii] )
    I_3_vec[ii+1] = (1-dt*(rb/Lb))*I_3_vec[ii] + dt*(r3/Lb)*Ib
    
    # if state_previous != state_next:
    #     print('state_previous = {}; state_next = {}'.format(state_previous,state_next))
        
    state_previous = state_next
      
#%% plot
fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)
# fig.suptitle('num_jjs = {}'.format(num_jjs))

ax[0].plot(time_vec,I_d, '-', color = colors['red3'])
ax[0].set_ylabel(r'$I_{drive}$ [$\mu$A]')

ax[1].plot(time_vec,I_1_vec, '-', color = colors['blue3'])
ax[1].set_ylabel(r'$I_{gate}$ [$\mu$A]')
     
ax[2].plot(time_vec,I_3_vec, '-', color = colors['green3'])
ax[2].set_ylabel(r'$I_{th}$ [$\mu$A]')
        
ax[2].set_xlabel(r'Time [ns]')
# ax[2].set_xlim([0,100])

plt.show()


