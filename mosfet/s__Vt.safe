import numpy as np

from _util import physical_constants

p = physical_constants()

# plt.close('all')

#%% units

# all SI units

#%% semiconductor parameters for transistor (nmos)

# temperature 
T = 300 # kelvin # need to investigate low temp

# dopants
N_a = 3e16*1e6 # dopants per m^3 in mosfet channel
n_i = 1.5e10*1e6 # intrinsic carrier concentration at T = 300K; need to investigate low T

# voltages
# http://ecee.colorado.edu/~bart/book/book/chapter6/ch6_3.htm
V_th = p['kB']*T/p['e'] # thermal voltage
phi_b = V_th*np.log(N_a/n_i)
V_fb = 4.1-4.05-0.56-phi_b # -0.34 # flat-band voltage with aluminum gate # volts (somewhere in the range -0.79V to -0.34V)

# capacitance
epsilon_r_gate = 3.9 # 1.45**2 # relative permittivity of gate oxide
epsilon_gate = epsilon_r_gate*p['epsilon0']
c_i = epsilon_r_gate*p['epsilon0']/20e-9 # capacitance of gate per unit area; F/m^2
epsilon_r_s = 11.9 # 3.48**2 # relative permittivity of silicon
epsilon_s = epsilon_r_s*p['epsilon0']

#%% threshold voltage
print('V_fb = {:5.3f}V'.format(V_fb))
V_T = V_fb + 2*phi_b + np.sqrt(4*epsilon_s*p['e']*N_a*phi_b)/c_i
print('V_T = {:5.3f}V'.format(V_T))