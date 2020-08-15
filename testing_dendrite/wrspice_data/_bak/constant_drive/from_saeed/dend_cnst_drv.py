import WRSpice
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
from IPython import get_ipython
import csv
from os import path
get_ipython().run_line_magic('matplotlib', 'qt')

# tempCirFile only netlist
# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
# VERY IMPORTTANT Leave a blank line in the start and the end of the cir file
rate=WRSpice.WRSpice(tempCirFile='dend_cnst_drv',OutDatFile='dend_cnst_drv') # no file extentions
rate.pathWRSSpice='/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
#rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save='v(4) i(L4)' # list all vectors you want to save


RedoSim=False 

L_left_vector=np.arange(10-3,10+4,1) # pH
I_de_vector=np.arange(70,80+2,2) # uA
I_drive_vector=np.arange(1,30+1,1) # uA

sTime=0.1 #psec
fTime=100 #nsec
FileFromat="_Llft{Lleft:05.2f}pH_Lrgt{LRight:05.2f}pH_Ide{Ide:05.2f}uA_Idrv{Idrive:05.2f}uA_Ldi0077.50nH_taudi0775ms_dt00.1ps"
#%%
for iL in np.arange(len(L_left_vector)):
    for iIde in np.arange(len(I_de_vector)):
        for iIdrive in np.arange(len(I_drive_vector)):
            
            #%%
#            iL=0
#            iIde=0
#            iIdrive=25
            L_left=L_left_vector[iL]
            L_Right=30-L_left
            I_de=I_de_vector[iIde]
            I_drive=I_drive_vector[iIdrive]
            
            FilePrefix=FileFromat.format(Lleft=L_left,LRight=L_Right,Ide=I_de,Idrive=I_drive)
    
            rate.FilePrefix=FilePrefix
            Params= {
    
              "L5": str(L_left)+'p', 
              "L6": str(L_Right)+'p', 
              "Ide":str(np.round(I_de,6))+'u',
              "Idrive":str(np.round(I_drive,6))+'u',
            }
            rate.Params=Params
            rate.stepTran=str(np.round(sTime,6))+'p'
            rate.stopTran=str(np.round(fTime,6))+'n'
    
            if RedoSim==False and path.exists(rate.OutDatFile+FilePrefix+'.dat'):
                print('Simulation File already exist')
            else:
                rate.doAll()
                
#            data_dict=rate.read_wr_data()
#            time_vec = data_dict['time']
#            v4=data_dict['v(4)']
#            iL4=data_dict['i(L4)']
##            plt.plot(time_vec,v4, 'r')
#            plt.plot(time_vec,iL4, 'b')
#    


  
    

