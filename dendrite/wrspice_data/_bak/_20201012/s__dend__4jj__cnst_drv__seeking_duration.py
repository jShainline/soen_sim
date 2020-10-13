import numpy as np
import time

from util import physical_constants
p = physical_constants()

import WRSpice

#%%

# tempCirFile only netlist
# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
# VERY IMPORTTANT Leave a blank line in the start and end of the  cir file

#%%
num_jjs = 4
M = np.sqrt(200*20)

#%% load flux onset vector

# with open('dend_{:1d}jj_flux_onset.soen'.format(num_jjs), 'rb') as data_file:         
#     data_array_imported = pickle.load(data_file)

# I_de_list = data_array_imported['I_de_list']
# Phi_a_on = data_array_imported['Phi_a_on']

if num_jjs == 2:

    # dendritic firing junction bias current
    dI_de = 1
    I_de_0 = 52
    I_de_f = 80

elif num_jjs == 4:
    
    # dendritic firing junction bias current
    dI_de = 1
    I_de_0 = 56
    I_de_f = 85
    
I_de_list = np.arange(I_de_0,I_de_f+dI_de,dI_de)
num_I_de = len(I_de_list)

I_drive_on__vec = [16.1648,15.7114,15.2568,14.8008,14.3432,13.8840,13.4228,12.9596,12.4940,12.0260,11.5552,11.0812,10.6038,10.1226, 9.6372, 9.1470, 8.6510, 8.1490, 7.6398, 7.1220, 6.5942, 6.0544, 5.4998, 4.9266, 4.3284, 3.6958, 3.0102, 2.2290, 1.1768, 0.0000]

num_steps = 200

max_flux = p['Phi0__pH_ns']/2
flux_resolution = max_flux/num_steps
I_drive_off = np.ceil(max_flux/flux_resolution)*flux_resolution/M
dI_drive = flux_resolution/M

#%%
rate = WRSpice.WRSpice(tempCirFile = 'dend__4jj__cnst_drv', OutDatFile = 'dend_4jj_cnst_drv_seek_dur_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'v(5)' # list all vectors you want to save
rate.pathCir = ''

L_di = 77.5e3 # pH

FileFormat = 'Ide{:05.2f}uA_Idrive{:08.5f}uA'
for ii in range(len(I_de_list)):
    I_drive_vec = np.arange(I_drive_on__vec[ii],I_drive_off,dI_drive)
    # print(I_drive_vec)
    # time.sleep(10)
    for jj in range(len(I_drive_vec)):
    
        print('ii = {} of {}; jj = {} of {}'.format(ii+1,len(I_de_list),jj+1,len(I_drive_vec)))
            
        FilePrefix = FileFormat.format(I_de_list[ii],I_drive_vec[jj])
        rate.FilePrefix = FilePrefix
                    
        Params = {          
        'Ide':'{:6.4f}u'.format(np.round(I_de_list[ii],4)),
        'Ia':'{:6.4f}u'.format(np.round(I_drive_vec[jj],4)),
        'Ldi':'{:6.4f}n'.format(np.round(L_di*1e-3,4))
        }
    
        rate.Params = Params
        rate.stepTran = '10p'
        rate.stopTran = '500n'
        rate.doAll()
