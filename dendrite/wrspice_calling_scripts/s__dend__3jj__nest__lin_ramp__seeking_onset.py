import numpy as np

from _util import physical_constants
p = physical_constants()
Phi0 = p['Phi0__pH_ns']

import WRSpice


#%%

# tempCirFile only netlist
# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
# VERY IMPORTTANT Leave a blank line in the start and end of the  cir file

#%% inputs

# dendritic series bias current
# dIb = 10
# Ib_vec = np.arange(85,105+dIb,dIb) # uA
Ib = 85 # uA

# inductor in plasticity loop
Lp = 35 # 25 # 12.5 # 18.75 # 17.5 #  20 # pH

# dendritic plasticity bias current
num_steps = 21 # num steps on either side of zero
max_flux = Phi0/2
flux_resolution = max_flux/num_steps
Mp = np.sqrt(200*Lp)
Ip_max = max_flux/Mp
Ip_vec = np.linspace(0,Ip_max,num_steps)
Ip_vec = np.concatenate((np.flipud(-Ip_vec)[0:-1],Ip_vec)) 

# excitatory flux input
Mex = np.sqrt(400*12.5)
Iex_max = max_flux/Mex


#%%
rate = WRSpice.WRSpice(tempCirFile = 'dend__3jj__nest__lin_ramp', OutDatFile = 'dend_3jj_nest_lin_ramp_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'i(L4) v(2)' # list all vectors you want to save
rate.pathCir = ''

# FileFormat = 'Ib{:05.2f}uA_Ip{:05.2f}uA_rmp{:1d}'
FileFormat = 'Ib{:05.2f}uA_Lp{:05.2f}pH_Ip{:09.6f}uA'
# for aa in [-1,1]: # don't want both sides of in activity flux; negative gives undesirable response shape
# for ii in range(len(Ib_vec)):
#     Ib = Ib_vec[ii]
for jj in range(len(Ip_vec)):
    Ip = Ip_vec[jj]
    
    # if aa == -1:
    #     _tn = 1
    # elif aa == 1:
    #     _tn = 2
    # print('aa = {} of {}; ii = {} of {}; jj = {} of {}'.format(_tn,2,ii+1,len(Ib_vec),jj+1,len(Ip_vec)))

    # print('ii = {} of {}; jj = {} of {}'.format(ii+1,len(Ib_vec),jj+1,len(Ip_vec)))            
    print('jj = {} of {}'.format(jj+1,len(Ip_vec)))            
        
    FilePrefix = FileFormat.format(Ib,Lp,Ip)
    rate.FilePrefix = FilePrefix
             
    Params = {          
    'Ip':'{:9.6f}u'.format(np.round(Ip,6)),
    'Ib':'{:6.2f}u'.format(np.round(Ib,2)),
    'Iex':'{:9.6f}u'.format(np.round(Iex_max,6)),
    'Lp':'{:4.2f}p'.format(Lp)
    }

    rate.Params = Params
    rate.stepTran = '10p'
    rate.stopTran = '102n'
    rate.doAll()
