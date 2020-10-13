import numpy as np
# import pickle

import WRSpice

#%%

# tempCirFile only netlist
# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
# VERY IMPORTTANT Leave a blank line in the start and end of the  cir file

#%% load I_drive_array
# file_name = 'I_drive_array_2jj_Llft20.0_Lrgt20.0_Lnf77.5.soen.txt'
# with open('soen_sim_data/'+file_name, 'rb') as data_file:         
#     data_array = pickle.load(data_file)
#     I_de_vec = data_array['I_de_vec']
#     I_drive_array = data_array['I_drive_array']
    
# the below two lines are here because grumpy runs python 2.7 and wont import pickle
I_drive_array = [[16.80],[16.34,16.80],[15.88,16.80],[15.41,16.41,16.80],[14.95,15.95,16.80],[14.48,15.48,16.48,16.80],[14.02,15.02,16.02,16.80],[13.55,14.55,15.55,16.55,16.80],[13.07,14.07,15.07,16.07,16.80],[12.59,13.59,14.59,15.59,16.59,16.80],[12.11,13.11,14.11,15.11,16.11,16.80],[11.62,12.62,13.62,14.62,15.62,16.62,16.80],[11.13,12.13,13.13,14.13,15.13,16.13,16.80],[10.63,11.63,12.63,13.63,14.63,15.63,16.63,16.80],[10.13,11.13,12.13,13.13,14.13,15.13,16.13,16.80],[9.62,10.62,11.62,12.62,13.62,14.62,15.62,16.62,16.80],[9.10,10.10,11.10,12.10,13.10,14.10,15.10,16.10,16.80],[8.58,9.58,10.58,11.58,12.58,13.58,14.58,15.58,16.58,16.80],[8.04,9.04,10.04,11.04,12.04,13.04,14.04,15.04,16.04,16.80],[7.49,8.49,9.49,10.49,11.49,12.49,13.49,14.49,15.49,16.49,16.80],[6.93,7.93,8.93,9.93,10.93,11.93,12.93,13.93,14.93,15.93,16.80],[6.35,7.35,8.35,9.35,10.35,11.35,12.35,13.35,14.35,15.35,16.35,16.80],[5.75,6.75,7.75,8.75,9.75,10.75,11.75,12.75,13.75,14.75,15.75,16.75,16.80],[5.11,6.11,7.11,8.11,9.11,10.11,11.11,12.11,13.11,14.11,15.11,16.11,16.80],[4.43,5.43,6.43,7.43,8.43,9.43,10.43,11.43,12.43,13.43,14.43,15.43,16.43,16.80],[3.68,4.68,5.68,6.68,7.68,8.68,9.68,10.68,11.68,12.68,13.68,14.68,15.68,16.68,16.80],[2.79,3.79,4.79,5.79,6.79,7.79,8.79,9.79,10.79,11.79,12.79,13.79,14.79,15.79,16.79,16.80],[1.21,2.21,3.21,4.21,5.21,6.21,7.21,8.21,9.21,10.21,11.21,12.21,13.21,14.21,15.21,16.21,16.80]]
I_de_vec = [53.00,54.00,55.00,56.00,57.00,58.00,59.00,60.00,61.00,62.00,63.00,64.00,65.00,66.00,67.00,68.00,69.00,70.00,71.00,72.00,73.00,74.00,75.00,76.00,77.00,78.00,79.00,80.00]

#%%
rate = WRSpice.WRSpice(tempCirFile = 'ne__2jj__direct_drive__cnst_drv', OutDatFile = 'ne_2jj_direct_drive_cnst_drv_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'L0#branch' # list all vectors you want to save
rate.pathCir = ''

FileFormat = 'Lnr20pH20pH_Ide{:05.2f}uA_Ldrv200pH_Idrv{:05.2f}_taunf50.00ns_dt01.0ps'
for ii in range(len(I_de_vec)):
    I_drive_vec = I_drive_array[ii]
    Ide = I_de_vec[ii] 
    
    for jj in range(len(I_drive_vec)):
        Idrive = I_drive_vec[jj] 
    
        print('ii = {} of {}; jj = {} of {}'.format(ii+1,len(I_de_vec),jj+1,len(I_drive_vec)))
       
        FilePrefix = FileFormat.format(Ide,Idrive)
        rate.FilePrefix = FilePrefix
                    
        Params = {          
        'Ide':str(np.round(Ide,6))+'u',
        'Idrive':str(np.round(Idrive,6))+'u'
        }
    
        rate.Params = Params
        rate.stepTran = '10p'
        rate.stopTran = '1000n'
        rate.doAll()

