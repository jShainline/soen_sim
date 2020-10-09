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
rate = WRSpice.WRSpice(tempCirFile = 'ne__4jj__single_pulse__alt_readout__no_rd', OutDatFile = 'ne_4jj_1pls_alt_read_no_rd_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'L4#branch L7#branch L2#branch L10#branch L3#branch' # list all vectors you want to save
rate.pathCir = ''

L_si = 50e-9

critical_current = 40e-6
current = 0
norm_current = np.max([np.min([current/critical_current,1]),1e-9])
L_jj = (3.2910596281416393e-16/critical_current)*np.arcsin(norm_current)/(norm_current)
I_de_vec = [76,82,88]
I_sy_vec = [76,78,80]
tau_si_vec = [500,1000]
FileFormat = 'Isy{:05.2f}uA_Ide{:05.2f}uA_tausi{:07.2f}ns'
for ii in range(len(I_de_vec)):
    Ide = I_de_vec[ii]
    for jj in range(len(I_sy_vec)):
        Isy = I_sy_vec[jj]
        for kk in range(len(tau_si_vec)):
            tau_si = tau_si_vec[kk]            
            r_si = (L_si+L_jj)/(tau_si*1e-9)
    
            print('ii = {} of {}; jj = {} of {}; kk = {} of {};'.format(ii+1,len(I_de_vec),jj+1,len(I_sy_vec),kk+1,len(tau_si_vec)))
                
            FilePrefix = FileFormat.format(Isy,Ide,tau_si)
            rate.FilePrefix = FilePrefix
                        
            Params = {          
            'Ide':str(np.round(Ide,2))+'u',
            'Isy':str(np.round(Isy,2))+'u',
            'rsi':'R=v(1)+'+str(np.round(r_si,4))
            }
        
            rate.Params = Params
            rate.stepTran = '10p'
            rate.stopTran = '300n'
            rate.doAll()
