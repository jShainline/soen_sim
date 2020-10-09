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
I_drive_array = [[13.67],[13.20,13.67],[12.73,13.67],[12.27,13.27,13.67],[11.80,12.80,13.67],[11.32,12.32,13.32,13.67],[10.85,11.85,12.85,13.67],[10.37,11.37,12.37,13.37,13.67],[9.88,10.88,11.88,12.88,13.67],[9.39,10.39,11.39,12.39,13.39,13.67],[8.90,9.90,10.90,11.90,12.90,13.67],[8.40,9.40,10.40,11.40,12.40,13.40,13.67],[7.89,8.89,9.89,10.89,11.89,12.89,13.67],[7.38,8.38,9.38,10.38,11.38,12.38,13.38,13.67],[6.85,7.85,8.85,9.85,10.85,11.85,12.85,13.67],[6.32,7.32,8.32,9.32,10.32,11.32,12.32,13.32,13.67],[5.77,6.77,7.77,8.77,9.77,10.77,11.77,12.77,13.67],[5.20,6.20,7.20,8.20,9.20,10.20,11.20,12.20,13.20,13.67],[4.61,5.61,6.61,7.61,8.61,9.61,10.61,11.61,12.61,13.61,13.67],[3.99,4.99,5.99,6.99,7.99,8.99,9.99,10.99,11.99,12.99,13.67],[3.33,4.33,5.33,6.33,7.33,8.33,9.33,10.33,11.33,12.33,13.33,13.67],[2.58,3.58,4.58,5.58,6.58,7.58,8.58,9.58,10.58,11.58,12.58,13.58,13.67],[0.99,1.99,2.99,3.99,4.99,5.99,6.99,7.99,8.99,9.99,10.99,11.99,12.99,13.67],[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,13.67],[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,13.67],[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,13.67],[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,13.67]]
I_de_vec = [74.00,75.00,76.00,77.00,78.00,79.00,80.00,81.00,82.00,83.00,84.00,85.00,86.00,87.00,88.00,89.00,90.00,91.00,92.00,93.00,94.00,95.00,96.00,97.00,98.00,99.00,100.00]

#%%
rate = WRSpice.WRSpice(tempCirFile = 'ne__4jj__direct_drive__cnst_drv__alt_ind', OutDatFile = 'ne_4jj_direct_drive_cnst_drv_alt_ind_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'L2#branch' # list all vectors you want to save
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
        rate.stepTran = '1p'
        rate.stopTran = '100n'
        rate.doAll()

