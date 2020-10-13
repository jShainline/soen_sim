import numpy as np

import WRSpice

#%%

# tempCirFile only netlist
# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
# VERY IMPORTTANT Leave a blank line in the start and end of the  cir file

#%% inputs
dI_de = 1
I_de_vec = np.arange(56,93+dI_de,dI_de) # uA

dI_sc = 5
I_sc_vec = np.arange(20,30+dI_sc,dI_sc) # uA

L_di = 77.5 # nH

#%%
rate = WRSpice.WRSpice(tempCirFile = 'dend__3jj__lin_ramp', OutDatFile = 'dend_3jj_lin_ramp_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'L1#branch L2#branch v(10) v(4)' # list all vectors you want to save
rate.pathCir = ''


FileFormat = 'Ide{:05.2f}uA_Isc{:05.2f}uA_Ldi{:07.2f}nH'
for jj in range(len(I_sc_vec)):
    I_sc = I_sc_vec[jj]
    for ii in range(len(I_de_vec)):
        I_de = I_de_vec[ii]
        
        print('ii = {} of {}; jj = {} of {}'.format(ii+1,len(I_de_vec),jj+1,len(I_sc_vec)))
            
        FilePrefix = FileFormat.format(I_de,I_sc,L_di)
        rate.FilePrefix = FilePrefix
                    
        Params = {          
        'Ide':'{:4.2f}u'.format(np.round(I_de,2)),
        'Isc':'{:4.2f}u'.format(np.round(I_sc,2)),
        'Ldi':'{:4.2f}u'.format(np.round(L_di,2))
        }
    
        rate.Params = Params
        rate.stepTran = '1p'
        rate.stopTran = '101n'
        rate.doAll()
