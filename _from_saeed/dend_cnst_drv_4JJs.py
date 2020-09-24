import WRSpice
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
from IPython import get_ipython
import csv
from os import path
get_ipython().run_line_magic('matplotlib', 'qt')
from util import physical_constants 

# tempCirFile only netlist# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
# VERY IMPORTTANT Leave a blank line in the start and the end of the cir file
rate=WRSpice.WRSpice(tempCirFile='dend',OutDatFile='dend_cnst_drv_4jj') # no file extentions
rate.pathWRSSpice='/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
#rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save='v(5) i(L2) v(12)' # list all vectors you want to save


RedoSim=False 
init_clip_time=1e-9
L_left_vector=[20] #np.arange(20-3,20+4,3) # pH
I_de_vector=np.arange(50,90+1,1) # uA


sTime=1 #psec
if path.exists('SpikeStart.dat'):
    Spikes_start=np.loadtxt('SpikeStart.dat')
else:
    Spikes_start=np.zeros((len(I_de_vector)),dtype=float)
    
#%%
p = physical_constants()
Phi_vec = np.linspace(0,p['Phi0']/2,50)
M = np.sqrt(200e-12*20e-12)
I_drive_vec = Phi_vec/M
resolution = 10e-9
I_drive_vec_round = np.round((Phi_vec/M)/resolution)*resolution
I_drive_vector=I_drive_vec_round*1e6
#%%
print('min(I_drive_vec) = {}uA; max(I_drive_vec = {}uA'.format(np.min(I_drive_vec)*1e6,np.max(I_drive_vec)*1e6))


FileFromat="_Llft{Lleft:05.2f}pH_Lrgt{LRight:05.2f}pH_Ide{Ide:05.2f}uA_Idrv{Idrive:05.2f}uA_Ldi0775.0nH_taudi0775ms_dt01.0ps"
#%%
for iL in np.arange(len(L_left_vector)):
    L_left=L_left_vector[iL]
    L_Right=L_left
    
#    M = np.sqrt(200e-12*L_left*1e-12)
#    Phi0=2.067833848e-15
#    IdriveMax=np.round(Phi0/2/M*1e6,3)
#    I_drive_vector=np.arange(0,IdriveMax+0.25,0.25) # uA
    for iIde in np.arange(len(I_de_vector)):
        minDrive=False
        FirstSpike=True
        
        fTime=100 #nsec
#        L_left_vector[iL]=20
#        I_de_vector[iIde]=76
        for iIdrive in np.arange(len(I_drive_vector)):
            
            if iIdrive==0 and (Spikes_start[iIde]!=0) :
                print('L_left='+str(L_left_vector[iL]) +'; Ide='+str(I_de_vector[iIde])+' Already done')
                break
            
            I_de=I_de_vector[iIde]
            I_drive=I_drive_vector[iIdrive]
            
            FilePrefix=FileFromat.format(Lleft=L_left,LRight=L_Right,Ide=I_de,Idrive=I_drive)
    
            rate.FilePrefix=FilePrefix
            Params= {
    
              "L4": str(L_left)+'p', 
              "L5": str(L_Right)+'p', 
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
            v2=data_dict['v(5)']
            iL2=data_dict['i(L2)']

            len_v2=len(v2)
            time_vec=time_vec[round(9*len_v2/10):len_v2]
            v2=v2[round(9*len_v2/10):len_v2]
            
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
                v2=data_dict['v(5)']
                iL2=data_dict['i(L2)']
    
                len_v2=len(v2)
                time_vec=time_vec[round(9*len_v2/10):len_v2]
                v2=v2[round(9*len_v2/10):len_v2]
                
                j_peaks, _ = find_peaks(v2, distance=200,height =150e-6)
              
            initial_ind = (np.abs(time_vec-init_clip_time)).argmin()
            final_ind = (np.abs(time_vec-time_vec.max())).argmin()
        
            v2_full=data_dict['v(5)']
            v2_full=v2_full[initial_ind:final_ind]
            j_peaks_full, _ = find_peaks(v2_full, distance=200,height =150e-6)
            if len(j_peaks_full)>1 :
                print('Spikes : '+FilePrefix)
#                Spikes_start[iIde]=I_drive
            else:
                print('No spikes : '+FilePrefix)    
                
                
            if len(j_peaks_full)>1 and minDrive==False:
                minDrive=True
                print(FilePrefix)
                print('min IDrive is : '+str(I_drive_vector[iIdrive-1]))
                print('mac IDrive is : '+str(I_drive_vector[iIdrive]))
                I_drive_vector2=list(np.arange(I_drive_vector[iIdrive-1],I_drive_vector[iIdrive],0.1)) # uA
                I_drive_vector2.append(I_drive_vector[iIdrive])
                print(I_drive_vector2)
        
                for iIdrive2 in np.arange(len(I_drive_vector2)):
                    
#                    fTime=20 #nsec
                    L_left=L_left_vector[iL]
                    L_Right=L_left
                    I_de=I_de_vector[iIde]
                    I_drive=I_drive_vector2[iIdrive2]
                    
                    FilePrefix=FileFromat.format(Lleft=L_left,LRight=L_Right,Ide=I_de,Idrive=I_drive)
            
                    rate.FilePrefix=FilePrefix
                    Params= {
            
                      "L4": str(L_left)+'p', 
                      "L5": str(L_Right)+'p', 
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
                    v2=data_dict['v(5)']
                    iL2=data_dict['i(L2)']
        
                    len_v2=len(v2)
                    time_vec=time_vec[round(9*len_v2/10):len_v2]
                    v2=v2[round(9*len_v2/10):len_v2]
                    
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
                        v2=data_dict['v(5)']
                        iL2=data_dict['i(L2)']
            
                        len_v2=len(v2)
                        time_vec=time_vec[round(9*len_v2/10):len_v2]
                        v2=v2[round(9*len_v2/10):len_v2]
                        
                        j_peaks, _ = find_peaks(v2, distance=200,height =150e-6)

                    initial_ind = (np.abs(time_vec-init_clip_time)).argmin()
                    final_ind = (np.abs(time_vec-time_vec.max())).argmin()
                
                    v2_full=data_dict['v(5)']
                    v2_full=v2_full[initial_ind:final_ind]
                    j_peaks_full, _ = find_peaks(v2_full, distance=200,height =150e-6)
                    if len(j_peaks_full)>1 :
                        print('Spikes : '+FilePrefix)
                        if FirstSpike==True:
                            Spikes_start[iIde]=I_drive
                            print(Spikes_start)
                            FirstSpike=False
#                            if RedoSim==False:
#                                break
                    else:
                        print('No spikes : '+FilePrefix) 
                        


        np.savetxt('SpikeStart.dat',Spikes_start) 
                        
            