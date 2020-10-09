
import numpy as np
from util import physical_constants 
from os import path
OutDatFile='dend_cnst_drv_4jj'
FileFromat="_Llft{Lleft:05.2f}pH_Lrgt{LRight:05.2f}pH_Ide{Ide:05.2f}uA_Idrv{Idrive:05.2f}uA_Ldi0775.0nH_taudi0775ms_dt01.0ps"
L_left_vector=[20] #np.arange(20-3,20+4,3) # pH
I_de_vector=np.arange(50,80+1,1) # uA


#%%
p = physical_constants()
Phi_vec = np.linspace(0,p['Phi0']/2,50)
M = np.sqrt(200e-12*20e-12)
I_drive_vec = Phi_vec/M
resolution = 10e-9
I_drive_vec_round = np.round((Phi_vec/M)/resolution)*resolution
I_drive_vector=I_drive_vec_round*1e6


SpikeStart=np.loadtxt('SpikeStart.dat')


print(SpikeStart)
I_drive_array=[]
for iIde in np.arange(len(I_de_vector)):
    I_de=I_de_vector[iIde]
    temp_I_drive=[]
    I_drive=np.round(SpikeStart[iIde],3)
    temp_I_drive.append(I_drive)
    
    index=(np.abs(I_drive_vector-I_drive)).argmin()

    if I_drive>I_drive_vector[index]:
        index=index+1
        
    while I_drive< I_drive_vector[index]:
        I_drive=I_drive+0.1
        temp_I_drive.append(round(I_drive,3))
    while I_drive< I_drive_vector[-1]:
        index=index+1
        I_drive=I_drive_vector[index]
        temp_I_drive.append(round(I_drive,3))

    for iDrive in np.arange(len(temp_I_drive)):
        I_drive=temp_I_drive[iDrive]
        FilePrefix=FileFromat.format(Lleft=L_left_vector[0],LRight=L_left_vector[0],Ide=I_de,Idrive=I_drive)
        if path.exists(OutDatFile+FilePrefix+'.dat'):
            print('File exists. Great1')
            continue
            
        else:
            print('Oops. No file')
            print(FilePrefix)
            break
    if path.exists(OutDatFile+FilePrefix+'.dat')==False:
            
            break    
    I_drive_array.append(temp_I_drive)
print(I_drive_array)
  

          