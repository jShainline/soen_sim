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
dIb = 2
Ib_vec = np.arange(130,134+dIb,dIb) # uA

# dendritic plasticity bias current
num_steps = 11 # num steps on either side of zero
max_flux = Phi0/2
flux_resolution = max_flux/num_steps
Mp = 77.5
Ip_max = max_flux/Mp
Ip_vec = np.linspace(0,Ip_max,num_steps)
Ip_vec = np.concatenate((np.flipud(-Ip_vec)[0:-1],Ip_vec))

# activity flux input
Ma = np.sqrt(400*12.5)
Ia_max = max_flux/Ma

#%%
rate = WRSpice.WRSpice(tempCirFile = 'dend__4jj__one_bias__plasticity__mid_betaL__lin_ramp', OutDatFile = 'dend_4jj_one_bias_plstc_lin_ramp_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'L8#branch v(2)' # list all vectors you want to save
rate.pathCir = ''


FileFormat = 'Ib{:05.2f}uA_Ip{:05.2f}uA_rmp{:1d}'
for aa in [-1,1]:
    for ii in range(len(Ib_vec)):
        Ib = Ib_vec[ii]
        for jj in range(len(Ip_vec)):
            Ip = Ip_vec[jj]
            
            if aa == -1:
                _tn = 1
            elif aa == 1:
                _tn = 2
            print('aa = {} of {}; ii = {} of {}; jj = {} of {}'.format(_tn,2,ii+1,len(Ib_vec),jj+1,len(Ip_vec)))
                
            FilePrefix = FileFormat.format(Ib,Ip,aa)
            rate.FilePrefix = FilePrefix
                     
            Params = {          
            'Ip':'{:9.6f}u'.format(np.round(Ip,6)),
            'Ib':'{:6.2f}u'.format(np.round(Ib,2)),
            'Ia':'{:4.2f}u'.format(Ia_max*aa)
            }
        
            rate.Params = Params
            rate.stepTran = '10p'
            rate.stopTran = '101n'
            rate.doAll()
