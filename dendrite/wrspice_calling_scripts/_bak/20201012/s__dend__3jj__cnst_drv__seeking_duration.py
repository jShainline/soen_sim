import numpy as np
import time

from _util import physical_constants
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
num_jjs = 3
M = np.sqrt(200*20)

L_di = 77.5 # nH

#%% load flux onset vector

# DI loop saturation capacity current
dI_sc = 5
# I_sc_vec = np.arange(20,30+dI_sc,dI_sc) # uA
I_sc_vec = np.arange(30,30+dI_sc,dI_sc) # uA
num_I_sc = len(I_sc_vec)

# dendritic firing junction bias current
dI_de = 1
# I_de_0__list = [65,64,60] # uA
# I_de_f__list = [93,92,89] # uA
I_de_0__list = [60] # uA
I_de_f__list = [89] # uA

L_di = 77.5 # nH

I_de_list__array = []
for jj in range(num_I_sc): 
    I_de_list__array.append(np.arange(I_de_0__list[jj],I_de_f__list[jj]+dI_de,dI_de))

# I_drive_on__array = [[16.1085,15.6547,15.1998,14.7434,14.2854,13.8255,13.3638,12.8999,12.4337,11.9650,11.4936,11.0189,10.5405,10.0584, 9.5720, 9.0804, 8.5833, 8.0796, 7.5686, 7.0486, 6.5183, 5.9753, 5.4166, 4.8380, 4.2326, 3.5889, 2.8845, 2.0608, 0.0000],[16.1278,15.6742,15.2194,14.7632,14.3053,13.8458,13.3845,12.9210,12.4550,11.9866,11.5154,11.0410,10.5631,10.0813, 9.5951, 9.1039, 8.6072, 8.1040, 7.5935, 7.0740, 6.5446, 6.0024, 5.4450, 4.8677, 4.2643, 3.6237, 2.9245, 2.1125, 0.8605],[16.1941,15.7430,15.2909,14.8375,14.3829,13.9268,13.4693,13.0101,12.5489,12.0858,11.6201,11.1522,10.6813,10.2072, 9.7299, 9.2486, 8.7628, 8.2723, 7.7763, 7.2739, 6.7640, 6.2456, 5.7169, 5.1759, 4.6195, 4.0435, 3.4414, 2.8017, 2.1021, 1.2830]]
I_drive_on__array = [[16.1941,15.7430,15.2909,14.8375,14.3829,13.9268,13.4693,13.0101,12.5489,12.0858,11.6201,11.1522,10.6813,10.2072, 9.7299, 9.2486, 8.7628, 8.2723, 7.7763, 7.2739, 6.7640, 6.2456, 5.7169, 5.1759, 4.6195, 4.0435, 3.4414, 2.8017, 2.1021, 1.2830]]

num_steps = 200

max_flux = p['Phi0__pH_ns']/2
flux_resolution = max_flux/num_steps
I_drive_off = np.ceil(max_flux/flux_resolution)*flux_resolution/M
dI_drive = flux_resolution/M

#%%
rate = WRSpice.WRSpice(tempCirFile = 'dend__{:d}jj__cnst_drv'.format(num_jjs), OutDatFile = 'dend_{:d}jj_cnst_drv_seek_dur_'.format(num_jjs)) # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'v(4)' # list all vectors you want to save
rate.pathCir = ''


FileFormat = 'Ide{:05.2f}uA_Isc{:05.2f}uA_Ldi{:07.2f}nH_Idrive{:08.5f}uA'

for qq in range(len(I_sc_vec)):
    I_de_list = I_de_list__array[qq]
    # print('I_de_list = {}'.format(I_de_list))
    for ii in range(len(I_de_list)):
        I_drive_on = I_drive_on__array[qq][ii]
        I_drive_vec = np.arange(I_drive_on,I_drive_off,dI_drive)
        # print('I_drive_vec = {}'.format(I_drive_vec))
        # time.sleep(1)
        
        for jj in range(len(I_drive_vec)):
        
            print('qq = {} of {}; ii = {} of {}; jj = {} of {}'.format(qq+1,len(I_sc_vec),ii+1,len(I_de_list),jj+1,len(I_drive_vec)))
                
            FilePrefix = FileFormat.format(I_de_list[ii],I_sc_vec[qq],L_di,I_drive_vec[jj])
            rate.FilePrefix = FilePrefix
                        
            Params = {          
            'Ide':'{:6.4f}u'.format(np.round(I_de_list[ii],4)),
            'Isc':'{:6.4f}u'.format(np.round(I_sc_vec[qq],4)),
            'Ia':'{:6.4f}u'.format(np.round(I_drive_vec[jj],4)),
            'Ldi':'{:6.4f}n'.format(np.round(L_di,4))
            }
        
            rate.Params = Params
            rate.stepTran = '10p'
            rate.stopTran = '500n'
            rate.doAll()
