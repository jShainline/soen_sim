import numpy as np

import WRSpice

#%%

# tempCirFile only netlist
# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
# VERY IMPORTTANT Leave a blank line in the start and end of the  cir file

#%%
dI_de = 1
I_de_vec = np.arange(56,85+dI_de,dI_de)

#%%
rate = WRSpice.WRSpice(tempCirFile = 'dend__4jj__lin_ramp', OutDatFile = 'dend_4jj_lin_ramp_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'L3#branch L2#branch v(12) v(5)' # list all vectors you want to save
rate.pathCir = ''

L_si = 77.5e-9

FileFormat = 'I_de{:05.2f}uA'
for ii in range(len(I_de_vec)):
    I_de = I_de_vec[ii]
    
    print('ii = {} of {}'.format(ii+1,len(I_de_vec)))
        
    FilePrefix = FileFormat.format(I_de)
    rate.FilePrefix = FilePrefix
                
    Params = {          
    'Ide':'{:4.2f}u'.format(np.round(I_de,2))
    }

    rate.Params = Params
    rate.stepTran = '1p'
    rate.stopTran = '101n'
    rate.doAll()
