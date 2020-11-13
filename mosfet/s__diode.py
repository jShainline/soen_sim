import numpy as np

from _util import physical_constants

p = physical_constants()

# plt.close('all')

#%% units

# all SI units

#%% parameters for LED (p-n junction, see pg 195 of Streetman and Banerjee)

T = 300 # temperature of operation; need to investigate low T

# geometrical
L_junc = 5e-6
h_junc = 70e-9
A_junc = L_junc*h_junc

# capacitance
L_intrinsic = 300e-9
epsilon_r_s = 3.48**2 # relative permittivity of silicon
epsilon_s = epsilon_r_s*p['epsilon0']
C_junc = epsilon_s*A_junc/L_intrinsic # capacitance of gate per unit area; F/m^2

Na = 5e19*1e6
Nd = 5e19*1e6
ni = 1.5e10*1e6 # need to investigate low T
p_n = ni**2/Nd
n_p = ni**2/Na

tau_np = 40e-9
tau_pn = 40e-9

mu_pp = 50e-4 # 200e-4;%mobility of holes on p side
mu_pn = 50e-4 # 450e-4;%mobility of holes on n side
mu_nn = 50e-4 # 1300e-4;%mobility of electrons on n side
mu_np = 50e-4 # 700e-4;%mobility of electrons on p side

Dp = (p['kB']*T/p['e'])*mu_pn
Dn = (p['kB']*T/p['e'])*mu_np

Lp = np.sqrt(Dp*tau_pn)
Ln = np.sqrt(Dn*tau_np)

#%% 
# current prefactor
I_s = p['e']*A_junc*( (Dp/Lp)*p_n + (Dn/Ln)*n_p )
print('I_s = {:5.3e}A'.format(I_s))

V_bias = 1
I_pn = I_s*( np.exp(p['e']*V_bias/(p['kB']*T)) - 1 )
print('I_pn = {:5.3e}A at V_bias = {:4.2f}V'.format(I_pn,V_bias))

print('C_junc = {:5.3e}F at 0V'.format(C_junc))
