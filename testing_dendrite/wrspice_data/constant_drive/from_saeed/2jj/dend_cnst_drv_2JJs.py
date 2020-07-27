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
rate=WRSpice.WRSpice(tempCirFile='dend_cnst_drv',OutDatFile='dend_cnst_drv_1jj') # no file extentions
rate.pathWRSSpice='/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
#rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save='v(2) i(L2)' # list all vectors you want to save


RedoSim=False 

L_left_vector=np.arange(10-3,10+4,1) # pH
I_de_vector=np.arange(70,80+2,2) # uA
I_drive_vector=np.arange(1,30+1,1) # uA

sTime=0.1 #psec
Spikes_start=np.zeros((len(L_left_vector),len(I_de_vector)),dtype=float)
FileFromat="_Llft{Lleft:05.2f}pH_Lrgt{LRight:05.2f}pH_Ide{Ide:05.2f}uA_Idrv{Idrive:05.2f}uA_Ldi0077.50nH_taudi0775ms_dt00.1ps"
#%%
for iL in np.arange(len(L_left_vector)):
    for iIde in np.arange(len(I_de_vector)):
        minDrive=False
        FirstSpike=True
        
        fTime=20 #nsec
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
    
              "L3": str(L_left)+'p', 
              "L4": str(L_Right)+'p', 
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
                
            data_dict=rate.read_wr_data()
            time_vec = data_dict['time']
            v2=data_dict['v(2)']
            iL2=data_dict['i(L2)']

            len_v2=len(v2)
            time_vec=time_vec[round(19*len_v2/20):len_v2]
            v2=v2[round(19*len_v2/20):len_v2]
            
            j_peaks, _ = find_peaks(v2, distance=200,height =150e-6)
            

            while len(j_peaks)>1:
                print('Redo : '+FilePrefix)
                fTime=fTime*(1+0.25)
                print('Sim Time : '+ str(np.round(fTime,6))+'n')
                rate.stepTran=str(np.round(sTime,6))+'p'
                rate.stopTran=str(np.round(fTime,6))+'n'
                rate.doAll()
                
                data_dict=rate.read_wr_data()
                time_vec = data_dict['time']
                v2=data_dict['v(2)']
                iL2=data_dict['i(L2)']
    
                len_v2=len(v2)
                time_vec=time_vec[round(19*len_v2/20):len_v2]
                v2=v2[round(19*len_v2/20):len_v2]
                
                j_peaks, _ = find_peaks(v2, distance=200,height =150e-6)
                
            v2_full=data_dict['v(2)']
            j_peaks_full, _ = find_peaks(v2_full, distance=200,height =150e-6)
            if len(j_peaks_full)>1 :
                print('Spikes : '+FilePrefix)
#                Spikes_start[iL][iIde]=I_drive
            else:
                print('No spikes : '+FilePrefix)    
            if len(j_peaks_full)>1 and minDrive==False:
                minDrive=True
                print(FilePrefix)
                print('min IDrive is : '+str(I_drive_vector[iIdrive-1]))
                I_drive_vector2=np.arange(I_drive_vector[iIdrive]-1,I_drive_vector[iIdrive]+0.1,0.1) # uA
                for iIdrive2 in np.arange(len(I_drive_vector2)):
                    
#                    fTime=20 #nsec
                    L_left=L_left_vector[iL]
                    L_Right=30-L_left
                    I_de=I_de_vector[iIde]
                    I_drive=I_drive_vector2[iIdrive2]
                    
                    FilePrefix=FileFromat.format(Lleft=L_left,LRight=L_Right,Ide=I_de,Idrive=I_drive)
            
                    rate.FilePrefix=FilePrefix
                    Params= {
            
                      "L3": str(L_left)+'p', 
                      "L4": str(L_Right)+'p', 
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
                        
                    data_dict=rate.read_wr_data()
                    time_vec = data_dict['time']
                    v2=data_dict['v(2)']
                    iL2=data_dict['i(L2)']
        
                    len_v2=len(v2)
                    time_vec=time_vec[round(19*len_v2/20):len_v2]
                    v2=v2[round(19*len_v2/20):len_v2]
                    
                    j_peaks, _ = find_peaks(v2, distance=200,height =150e-6)
                    
        
                    while len(j_peaks)>1:
                        print('Redo : '+FilePrefix)
                        fTime=fTime*(1+0.25)
                        print('Sim Time : '+ str(np.round(fTime,6))+'n')
                        rate.stepTran=str(np.round(sTime,6))+'p'
                        rate.stopTran=str(np.round(fTime,6))+'n'
                        rate.doAll()
                        
                        data_dict=rate.read_wr_data()
                        time_vec = data_dict['time']
                        v2=data_dict['v(2)']
                        iL2=data_dict['i(L2)']
            
                        len_v2=len(v2)
                        time_vec=time_vec[round(19*len_v2/20):len_v2]
                        v2=v2[round(19*len_v2/20):len_v2]
                        
                        j_peaks, _ = find_peaks(v2, distance=200,height =150e-6)
                    v2_full=data_dict['v(2)']
                    j_peaks_full, _ = find_peaks(v2_full, distance=200,height =150e-6)
                    if len(j_peaks_full)>1 :
                        print('Spikes : '+FilePrefix)
                        if FirstSpike==True:
                            Spikes_start[iL][iIde]=I_drive
                            print(Spikes_start)
                            FirstSpike=False
                    else:
                        print('No spikes : '+FilePrefix) 
                        
            with open('./SpikeStart.dat', 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerow(Spikes_start)

            np.savetxt('SpikeStart.dat',Spikes_start)
            