import numpy as np
from matplotlib import pyplot as plt

from _functions import Ljj_pH as Ljj
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

rb1 = 7.2e6 # right-most
rb2 = 800e3 # middle
rb3 = 1.9e6 # left-most

L1 = 20 # inductance of each branch of DR squid loop
L2 = 67.5 # next inductor over
L3 = 67.5 # next one
L4 = 77.5e3 # DI loop
L5 = 10 # driver side of MIs
L6 = 10 # receiver side of MIs

Ic = 40 # uA

Vb = 1e9

#%% calculated quantities
M = np.sqrt(L5*L6)

#%% iterate
Ia = 0
I2 = 0
I3 = 0

I4 = 0
I4a = 0
I4b = 0

I5 = 0
I5a = 0
I5b = 0

I6 = 0
I6a = 0
I6b = 0

I7 = 0
I7a = 0
I7b = 0

Ib1 = Vb/rb1
Ib2 = Vb/rb2
Ib3 = Vb/rb3

num_it = 10
I6_vec = np.zeros([num_it])
I4_vec = np.zeros([num_it])
Ia_vec = np.zeros([num_it])
for ii in range(num_it):
    
    print('ii = {} of {}'.format(ii+1,num_it))
    
    La = (L1+Ljj(Ic,Ia))/2
    Lb = L2+La
    Lc = ip(Lb,L6+Ljj(Ic,I4))
    Ld = L3+Lc
    Le = ip(Ld,Ljj(Ic,I6))
    Lf = ip(Ld,L4)
    Lg = Lf+L6+Ljj(Ic,I6)
    Lh = ip(L4,L6+Ljj(Ic,I6))
    Li = L3+Lh
    Lj = ip(Lb,Li)
    Lk = Lj+L6+Ljj(Ic,I4)
    Lm = Ld+L6+Ljj(Ic,I6)
    Ln = Ld+L4
    Lp = Lb+L6+Ljj(Ic,I4)
    
    I7 = Ib1
    
    I6a = (Ld/Lm)*I7 + (L4/(L4+Lm))*I5b
    I6b = (M/Lg)*Ib2
    I6 = I6a-I6b
    I6_vec[ii] = I6
           
    I5a = ((L6+Ljj(Ic,I6))/Lm)*I7 + (L4/Ln)*I6b
    I5b = (Lb/(Lb+Li))*I4b
    I5 = I5a-I5b
    
    I4a = (Lb/Lp)*I5a
    I4b = (M/Lk)*Ib3
    I4 = I4a-I4b
    I4_vec[ii] = I4
    
    I3 = ((L6+Ljj(Ic,I4))/Lp)*I5a + (Li/(Lb+Li))*I4b
    Ia = I3/2
    Ia_vec[ii] = Ia
    
print('I7 = {}uA; I6 = {}uA; I4 = {}uA; Ia = {}uA'.format(I7,I6,I4,Ia))    
    
#%% plot
iteration_vec = np.arange(1,num_it+1,1)
fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)   

axs[0].plot(iteration_vec,I6_vec, '-', color = colors['blue3'], label = 'I6') 
axs[0].set_ylabel(r'I6 [$\mu$A]') 

axs[1].plot(iteration_vec,I4_vec, '-', color = colors['blue3'], label = 'I7') 
axs[1].set_ylabel(r'I4 [$\mu$A]')

axs[2].plot(iteration_vec,Ia_vec, '-', color = colors['blue3'], label = 'I6') 
axs[2].set_ylabel(r'Ia [$\mu$A]')

axs[2].set_xlabel(r'Iteration')
    
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