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
rate = WRSpice.WRSpice(tempCirFile = 'squid__demo', OutDatFile = 'sq_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'L0#branch L3#branch v(3)' # list all vectors you want to save
rate.pathCir = ''

dt = 0.1 # ps
I_sq_vec = np.arange(40,85,10)
FileFormat = 'Isq{:05.2f}uA_dt{:04.1f}ps'
for ii in range(len(I_sq_vec)):
    
    print('ii = {} of {}'.format(ii+1,len(I_sq_vec)))
    
    Isq = I_sq_vec[ii]    
    FilePrefix = FileFormat.format(Isq,dt)
    rate.FilePrefix = FilePrefix
                
    Params = {          
    'Isq':'{}u'.format(np.round(Isq,2)),
    't1':'{:6.3f}n'.format(0.999),
    't2':'{:6.3f}n'.format(1),
    't3':'{:6.3f}n'.format(101),
    't4':'{:6.3f}n'.format(101),
    't5':'{:6.3f}n'.format(201),
    't6':'{:6.3f}n'.format(201.001),
    }

    rate.Params = Params
    rate.stepTran = '{:4.1f}p'.format(dt)
    rate.stopTran = '202n'
    rate.doAll()

# data_dict = rate.read_wr_data()
