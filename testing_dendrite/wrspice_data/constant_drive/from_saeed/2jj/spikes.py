
import numpy as np


RedoSim=False 

L_left_vector=np.arange(10-3,10+4,1) # pH
I_de_vector=np.arange(70,80+2,2) # uA

SpikeStart=np.loadtxt('SpikeStart.dat')



print(SpikeStart)
strI_drive='I_drive_array = ['
for iL in np.arange(len(L_left_vector)):
    if iL==0:
        strI_drive=strI_drive+'['
    else:
        strI_drive=strI_drive+'],\n['
    for iIde in np.arange(len(I_de_vector)):
        I_drive=np.round(SpikeStart[iL][iIde],3)
        
#      cv(18.3,18.9,d1,19,30,d2),
#        if np.ceil(I_drive)-0.1>I_drive:
        strI_drive=strI_drive+'cv('+str(np.round(I_drive,6))+','+str(np.round(np.ceil(I_drive)-0.1,6))+',d1,'+str(np.round(np.ceil(I_drive),6))+',30,d2),'
#        print(I_drive)
#        print('cv('+str(np.round(I_drive,6))+','+str(np.round(np.ceil(I_drive)-0.1,6))+',d1,'+str(np.round(np.ceil(I_drive),6))+',30,d2),')
#        else:
#            strI_drive=strI_drive+'cv('+str(np.round(I_drive,6))+','+str(np.round(np.ceil(I_drive)+1-0.1,6))+',d1,'+str(np.round(np.ceil(I_drive)+1,6))+',30,d2),'
strI_drive=strI_drive+']]'
print(strI_drive)
  