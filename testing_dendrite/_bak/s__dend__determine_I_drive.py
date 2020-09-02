import numpy as np
from util import physical_constants 


L_left_vector=[20] #np.arange(20-3,20+4,3) # pH
I_de_vector = np.arange(50,80+1,1) # uA


#%%
p = physical_constants()
Phi_vec = np.linspace(0,p['Phi0']/2,50)
M = np.sqrt(200e-12*20e-12)
I_drive_vec = Phi_vec/M
resolution = 10e-9
I_drive_vec_round = np.round((Phi_vec/M)/resolution)*resolution
I_drive_vector = I_drive_vec_round*1e6


# SpikeStart = np.loadtxt('wrspice_data/4jj/SpikeStart.dat')
SpikeStart = np.loadtxt('wrspice_data/2jj/SpikeStart.dat')


print('SpikeStart = {}\n\n'.format(SpikeStart))
I_drive_array=[]
for iIde in np.arange(len(I_de_vector)):
    temp_I_drive=[]
    I_drive = np.round(SpikeStart[iIde],3)
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
    
    I_drive_array.append(temp_I_drive)
print('I_drive_array = [ {} ]'.format(I_drive_array))
