from scipy.integrate import odeint

from _functions import low_pass_filter

#%%
dt = 100e-12
t0 = 0
tf = 100e-9

low_pass_filter(y,t,r,L1,L2,C,M,dIdrive_dt)1p