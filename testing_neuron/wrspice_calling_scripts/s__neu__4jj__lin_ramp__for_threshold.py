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
rate = WRSpice.WRSpice(tempCirFile = 'ne__4jj__direct_drive__lin_ramp__alt_ind', OutDatFile = 'ne_4jj_direct_drive_lin_ramp_alt_ind_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'L2#branch L3#branch' # list all vectors you want to save
rate.pathCir = ''

I_de_vec = np.arange(74,101,1)
FileFormat = 'Ldrv200pH_Lnr20pH20pH_Ide{:05.2f}uA_taunf50.00ns_dt01.0ps'
for ii in range(len(I_de_vec)):
    
    print('ii = {} of {}'.format(ii+1,len(I_de_vec)))
    
    Ide = I_de_vec[ii]    
    FilePrefix = FileFormat.format(Ide)
    rate.FilePrefix = FilePrefix
                
    Params = {          
    'Ide':str(np.round(Ide,2))+'u',
    }

    rate.Params = Params
    rate.stepTran = '1p'
    rate.stopTran = '101n'
    rate.doAll()

# data_dict = rate.read_wr_data()
