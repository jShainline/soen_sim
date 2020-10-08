import numpy as np
from matplotlib import pyplot as plt

from _functions import Ljj_pH as Lj
from _functions import ind_in_par as ip
from util import physical_constants, color_dictionary

colors = color_dictionary()
p = physical_constants()

#%% 

# units of:
# pH
# uA
# ns
# mohm
# nV

#%% inputs

Ia_desired = 35 # desired current in each branch of DR squid loop
I6_desired = 35 # desired current in final jj in DI loop
I4_desired = 35 # desired current in jj in JTL

rb1 = 7.2e5 # 7.2e6  # right-most
rb2 = 800e2 # 800e3  # middle
rb3 = 1.9e5 # 1.9e6  # left-most

L1 = 20 # inductance of each branch of DR squid loop
L2 = 67.5 # next inductor over
L3 = 67.5 # next one
L4 = 77.5e3 # DI loop
L5 = 10 # driver side of MIs
L6 = 10 # receiver side of MIs

Ic = 40 # uA

V0 = 1e8 # 1e9

#%% time
dt = 0.0001
t0 = 1
t1 = 21
tf = 30
time_vec = np.arange(0,tf+dt,dt)

_nt = len(time_vec)

#%% calculated quantities

M = np.sqrt(L5*L6)

V = np.zeros([_nt])
V3 = np.zeros([_nt])
V2 = np.zeros([_nt])
# _t0_ind = ( np.abs( time_vec-t0 ) ).argmin()
# _t1_ind = ( np.abs( time_vec-t1 ) ).argmin()
for ii in range(_nt):
    if time_vec[ii] >= t0 and time_vec[ii] < t1:
        V[ii] = (time_vec[ii]-t0)*V0/(t1-t0)
    elif time_vec[ii] >= t1:
        V[ii] = V0
        
#%%
Ia = np.zeros([_nt])
I2 = np.zeros([_nt])
I3 = np.zeros([_nt])
I4 = np.zeros([_nt])
I5 = np.zeros([_nt])
I6 = np.zeros([_nt])
I7 = np.zeros([_nt])

Ib1 = np.zeros([_nt])
Ib2 = np.zeros([_nt])
Ib3 = np.zeros([_nt])

ii_vec = np.arange(0,_nt-1,1)
for ii in ii_vec:
    La = L1+Lj(Ic,Ia[ii])
    Lb = L6+Lj(Ic,I4[ii])
    Lc = L6+Lj(Ic,I6[ii])
    I7[ii+1] = I7[ii] + (1/rb1)*( V[ii+1]-V[ii] )
    V3[ii+1] = -(L4/dt)*( I7[ii+1]-I7[ii] )
    Ib2[ii+1] = (1-(dt*rb2/L5))*Ib2[ii] + (dt/L5)*V[ii+1]
    I6[ii+1] = I6[ii] + (dt/Lc)*V3[ii+1] - (M/Lc)*Ib2[ii+1]
    I5[ii+1] = I5[ii] + (I7[ii+1]-I7[ii]) - (I6[ii+1]-I6[ii])
    V2[ii+1] = V3[ii+1] - (L3/dt)*( I5[ii+1]-I5[ii] )
    Ib3[ii+1] = (1-dt*rb3/L5)*Ib3[ii] + (dt/L5)*V[ii+1]
    I4[ii+1] = I4[ii] + (dt/Lb)*V2[ii+1] - (M/Lb)*( Ib3[ii+1]-Ib3[ii] )
    I3[ii+1] = I3[ii] + ( I5[ii+1]-I5[ii] ) - ( I4[ii+1]-I4[ii] )
    Ia[ii+1] = Ia[ii] + (dt/La)*V2[ii+1] - (L2/La)*( I3[ii+1]-I3[ii] )
    
#%% plot

fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)   

axs[0].plot(time_vec,V, '-', color = colors['blue3'], label = 'V(t)') 
axs[0].set_ylabel(r'V [nV]') 

axs[1].plot(time_vec,I7, '-', color = colors['blue3'], label = 'I7') 
axs[1].set_ylabel(r'I7 [$\mu$A]')

axs[2].plot(time_vec,I6, '-', color = colors['blue3'], label = 'I6') 
axs[2].set_ylabel(r'I6 [$\mu$A]')

axs[2].set_xlabel(r'Time [ns]')
    
# # neuron
# nr_flux__wr = (I_si__wr*neuron_instance.synapses[name_1].M - I_threshold_2*neuron_instance.refraction_M)/Phi_0
# axs[0].plot(time_vec__wr,nr_flux__wr, '-', color = colors['yellow3'], label = 'wr; $M^{ni|si} = $'+'{:5.2f}pH'.format(neuron_instance.synapses[name_1].M))
# # axs[0].plot(time_vec__wr,I_si__wr*neuron_instance.synapses[name_1].M/Phi_0, '-', color = colors['red2'], label = 'wr; no ref')
# # axs[0].plot(time_vec__wr,I_threshold_2*neuron_instance.refraction_M/Phi_0, '-', color = colors['red4'], label = 'wr; just ref')
# axs[0].plot(time_vec,neuron_instance.influx_vec/Phi_0, '-', color = colors['green3'], label = neuron_instance.name+'; Isy = {:5.2f}$\mu$A'.format(neuron_instance.synapses[name_1].bias_currents[0])+' $I_{ne}$ = '+'{:5.2f}uA'.format(neuron_instance.bias_currents[0]))
# # axs[0].plot(time_vec,neuron_instance.influx_vec__no_refraction/Phi_0, '-', color = colors['blue2'], label = neuron_instance.name+'; no ref')
# # axs[0].plot(time_vec,neuron_instance.influx_vec__just_refraction/Phi_0, '-', color = colors['blue4'], label = neuron_instance.name+'; just ref')

# x1 = time_vec[0]
# x2 = time_vec[-1]
# y1 = neuron_instance.dendrites['{}__d'.format(neuron_instance.name)].influx_list__dend[1]/Phi_0
# ylims = axs[0].get_ylim()
# axs[0].plot([x1,x2],[y1,y1], ':', color = colors['greengrey5'], linewidth = pp['fine_linewidth'], label = 'ne__d thresh')
# axs[0].set_ylim(ylims)

# y1 = neuron_instance.dendrites['{}__d'.format(neuron_instance.name)].influx_list__dend[1]/Phi_0
# # y2 = -y1
# ylims = axs[0].get_ylim()
# # axs[2].plot([x1,x2],[y1,y1], '-.', color = colors['red1'], linewidth = pp['fine_linewidth'], label = 'th')
# # axs[2].plot([x1,x2],[y2,y2], '-.', color = colors['red1'], linewidth = pp['fine_linewidth'], label = '-th')   
# axs[1].plot(time_vec__wr,I_ni__wr, '-', color = colors['yellow3'], label = 'wr')    
# axs[1].plot(time_vec,neuron_instance.I_ni_vec, '-', color = colors['green3'], label = neuron_instance.name+'; $tau_{ni}$ = '+'{:4.0f}ns'.format(neuron_instance.integration_loop_time_constant))    
# # axs[3].plot(neuron_instance.spike_times*1e6,neuron_instance.output_voltage[neuron_instance.spike_times], 'x', markersize = pp['nominal_markersize'], color = colors['blue5'])

# axs[0].set_ylabel(r'$\Phi^{nr}_{a}/\Phi_0$')
# axs[0].set_ylim(ylims)
# axs[1].set_ylabel(r'$I^{ni}$ [$\mu$A]')