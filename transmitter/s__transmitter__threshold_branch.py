import numpy as np
from matplotlib import pyplot as plt
import time
import copy

from _util import color_dictionary, physical_constants
from _functions import dendritic_drive__piecewise_linear
colors = color_dictionary()

p = physical_constants()

# plt.close('all')

#%% units

# all SI units

#%% semiconductor parameters for transistor(nmos)

# mobility, diffusion, capacitance
mu_n = 0.12 # mobility of electrons in channel; m^2/Vs
T_mosfet = 300 # K; need to investigate low temp
D_n = (p['kB']*T_mosfet/p['e'])*mu_n
epsilon_r_gate = 1.45**2
c_i = epsilon_r_gate*p['epsilon0']/20e-9 # capacitance of gate per unit area; F/m^2

# geometrical
h_soi = 70e-9 # thickness of soi/mosfet channel
w_mosfet = 1e-6 # width of mosfet channel
A_mosfet_cond = h_soi*w_mosfet
L_mosfet = 1e-6 # length of channel
A_mosfet = w_mosfet*L_mosfet

# dopants
N_a = 1e17*1e6 # dopants per m^3 in mosfet channel
n_i = 1.5e10*1e6 # intrinsic carrier concentration at T = 300K; need to investigate low T
n_p = n_i**2/N_a

# current prefactor in subthreshold expressions
I_0 = p['e']*A_mosfet_cond*D_n*n_p/L_mosfet


r_s = 1e4 # resistance in series with mosfet gate # ohms
C_m = c_i*A_mosfet

#%% semiconductor parameters for diode (p-n junction, see pg 195 of Streetman and Banerjee)
# T = 300 # temperature of operation; need to investigate low T

# L_junc = 5e-6
# h_junc = 200e-9
# A_junc = L_junc*h_junc

# Na = 5e19*1e6
# Nd = 5e19*1e6
# ni = 1.5e10*1e6 # need to investigate low T
# p_n = ni**2/Nd
# n_p = ni**2/Na

# tau_np = 40e-9;
# tau_pn = 40e-9;

# mu_pp = 100e-4 # 200e-4;%mobility of holes on p side
# mu_pn = 100e-4 # 450e-4;%mobility of holes on n side
# mu_nn = 250e-4 # 1300e-4;%mobility of electrons on n side
# mu_np = 250e-4 # 700e-4;%mobility of electrons on p side

# Dp = (p.kB*T/p.eE)*mu_pn;
# Dn = (p.kB*T/p.eE)*mu_np;

# Lp = np.sqrt(Dp*tau_pn);
# Ln = np.sqrt(Dn*tau_np);

# Ipn = p['eE']*A_junc*( (Dp/Lp)*p_n + (Dn/Ln)*n_p )*( np.exp(p['eE']*V/(p['kB']*T)) - 1 );


#%% hTron circuit parameters
num_squares = 20
r_sq = 500 # ohms per square
L_sq = 200e-12 # henries per square
L_h_extra = 100e-9 # henries

r_h_off = 0
r_h_on = r_sq*num_squares
L_h = L_sq*num_squares+L_h_extra

I_b = 100e-6 # current bias to circuit # amps

#%% time parameters

dt = 0.001e-9 # s
tf = 50e-9 # s
time_vec = np.arange(0,tf+dt,dt)
_nt = len(time_vec)

#%% drive

isi = 100e-9
t1 = 5e-9
dt_rise = 1e-9
pwl = [[0,r_h_off],[t1,r_h_off],[t1+dt_rise,r_h_on]]
r_h = dendritic_drive__piecewise_linear(time_vec,pwl)  

# Id = 20
# drv_params = dict()
# drv_params['t_r1_start'] = 5
# drv_params['t_r1_rise'] = 0.1
# drv_params['t_r1_pulse'] = 0.2
# drv_params['t_r1_fall'] = 1
# drv_params['t_r1_period'] = isi
# drv_params['value_r1_off'] = 0
# drv_params['value_r1_on'] = 5e6
# drv_params['L1'] = 100e3
# drv_params['L2'] = 100e3
# drv_params['r2'] = 200e3/50
# drv_params['Ib'] = Id
    
# I_d = dendritic_drive__exp_pls_train__LR(time_vec,drv_params)

#%% ode loop

V_2_vec = np.zeros([_nt])
Q_vec = np.zeros([_nt])

ii_vec = np.arange(1,_nt-1,1)
for ii in ii_vec:
    Q_vec[ii+1] = ( 1 - dt*(r_s+r_h[ii])/L_h )*Q_vec[ii] - (dt/(L_h*C_m))*V_2_vec[ii] + ((dt*r_h[ii])/(L_h*C_m))*I_b
    V_2_vec[ii+1] = V_2_vec[ii] + dt*Q_vec[ii+1]

    
#%% plot
_fs = 24 # font size
plt.rcParams['axes.labelsize'] = _fs
plt.rcParams['xtick.labelsize'] = _fs
plt.rcParams['ytick.labelsize'] = _fs
plt.rcParams['legend.fontsize'] = 16

fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)
# fig.suptitle('num_jjs = {}'.format(num_jjs))

ax[0].plot(time_vec,r_h*1e-3, '-', color = colors['red3'])
ax[0].set_ylabel(r'hTron resistance [k$\Omega$]')

ax[1].plot(time_vec,V_2_vec, '-', color = colors['green3'])
ax[1].set_ylabel(r'$V_{gate}$ [V]')

        
ax[1].set_xlabel(r'Time [ns]')
# ax[2].set_xlim([0,100])

plt.show()


#temporal zoom
# tp = 20
# fig2 = copy.deepcopy(fig)
# ax[1].set_xlim([tp,time_vec[-1]])
# ax[1].set_ylim([0.9,1.1])

# plt.show()