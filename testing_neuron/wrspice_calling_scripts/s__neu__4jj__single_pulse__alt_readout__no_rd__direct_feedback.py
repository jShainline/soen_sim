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
I_nf_vec = [72,76,80] # [76] # 
I_sy_vec = [72,76,80] # [76] # 
tau_si_vec = [500,1000,2000] # [1000] # 

#%%
rate = WRSpice.WRSpice(tempCirFile = 'ne__4jj__single_pulse__alt_readout__no_rd__direct_feedback', OutDatFile = 'ne_4jj_1pls_alt_read_no_rd_dir_fb_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'L8#branch L3#branch L6#branch L13#branch L14#branch v(12) v(33)' # list all vectors you want to save
rate.pathCir = ''

L_si = 77.5e-9

critical_current = 40e-6
current = 0
norm_current = np.max([np.min([current/critical_current,1]),1e-9])
L_jj = (3.2910596281416393e-16/critical_current)*np.arcsin(norm_current)/(norm_current)

FileFormat = 'I_sy{:05.2f}uA_I_nf{:05.2f}uA_tau_si{:07.2f}ns'
for ii in range(len(I_nf_vec)):
    I_nf = I_nf_vec[ii]
    for jj in range(len(I_sy_vec)):
        I_sy = I_sy_vec[jj]
        for kk in range(len(tau_si_vec)):
            tau_si = tau_si_vec[kk]            
            r_si = (L_si+L_jj)/(tau_si*1e-9)
    
            print('ii = {} of {}; jj = {} of {}; kk = {} of {};'.format(ii+1,len(I_nf_vec),jj+1,len(I_sy_vec),kk+1,len(tau_si_vec)))
                
            FilePrefix = FileFormat.format(I_sy,I_nf,tau_si)
            rate.FilePrefix = FilePrefix
                        
            Params = {          
            'Isy':'{:4.2f}u'.format(np.round(I_sy,2)),
            'Inf':'{:4.2f}u'.format(np.round(I_nf,2)),
            'rsi':'R=v(1)+{:6.4e}'.format(np.round(r_si,4))
            }
        
            rate.Params = Params
            rate.stepTran = '10p'
            rate.stopTran = '1u'
            rate.doAll()
