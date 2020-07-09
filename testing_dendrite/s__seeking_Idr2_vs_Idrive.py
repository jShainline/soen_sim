#%%
import numpy as np
from matplotlib import pyplot as plt
import time

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_error_mat
from _functions import dendrite_current_splitting, Ljj, I_di_sat_of_I_dr_2

plt.close('all')

#%% circuit parameters

Ic = 40e-6
Ldr1 = 10e-12
Ldr2 = 26e-12
L1 = 200e-12
L2 = 77.5e-12
L3 = 77.5e-9
Ib = [71.5e-6, 36e-6, 35e-6]
Lm2 = 10e-12
M = np.sqrt(200e-12*Lm2)

#%% calculate

I_drive_vec = np.arange(19e-6,27e-6,1e-6)
Idr1_vec = np.zeros([len(I_drive_vec),1])
Idr2_vec = np.zeros([len(I_drive_vec),1])

Lj0 = Ljj(Ic,0)
Iflux = 0
Idr2_prev = ((Lm2+Ldr1+Lj0)*Ib[0]+M*Iflux)/( Lm2+Ldr1+Ldr2+2*Lj0 + (Lm2+Ldr1+Lj0)*(Ldr2+Lj0)/L1 )
Idr1_prev = Ib[0]-( 1 + (Ldr2+Lj0)/L1 )*Idr2_prev

for ii in range(len(I_drive_vec)):
    for jj in range(10):
        Idr1_next, Idr2_next = dendrite_current_splitting(Ic,I_drive_vec[ii],Ib[0],Ib[1],Ib[2],M,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev)
        Idr1_prev = Idr1_next
        Idr2_prev = Idr2_next
    Idr1_vec[ii] = Idr1_next
    Idr2_vec[ii] = Idr2_next

Idr2_fit = np.polyfit(I_drive_vec,Idr2_vec,1)
I_drive_vec_dense = np.linspace(0,I_drive_vec[-1],100)
Idr2_vec_dense = np.polyval(Idr2_fit,I_drive_vec_dense)
threshold_ind = (np.abs(Idr2_vec_dense-Ic)).argmin()
Idrive_threshold = I_drive_vec_dense[threshold_ind]

max_valid_ind = (np.abs(I_drive_vec_dense-26e-6)).argmin()
max_valid_Idr2 = Idr2_vec_dense[max_valid_ind]

I_drive_fit = np.polyfit(Idr2_vec[:,0],I_drive_vec,1)
Idr2_vec_dense = np.linspace(0,Idr2_vec[-1],100)
I_drive_vec_dense = np.polyval(I_drive_fit,Idr2_vec_dense)

title_string = 'Idr2 = {}*Iflux+{}; Iflux = {}*Idr2+{}\nIflux_threshold = {}uA'.format(Idr2_fit[0],Idr2_fit[1],I_drive_fit[0],I_drive_fit[1],Idrive_threshold*1e6)
    
#%% plot

fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('Fitting Idr2 vs Iflux')
plt.title(title_string)
ax.plot(I_drive_vec*1e6,Idr2_vec*1e6, 'o-', label = 'data')   
ax.plot(I_drive_vec_dense*1e6,Idr2_vec_dense*1e6, '-', label = 'fit') 
ax.plot([0,Idrive_threshold*1e6], [Ic*1e6,Ic*1e6], '-.', label = 'Idr2 threshold')            
ax.plot([Idrive_threshold*1e6,Idrive_threshold*1e6], [0,Ic*1e6], '-.', label = 'Iflux at threshold')            
ax.set_xlim([0,30])
ax.set_ylim([30,50])
ax.set_xlabel(r'$I_{flux}$ [$\mu$A]')
ax.set_ylabel(r'$I_{dr2}$ [$\mu$A]')
ax.legend()    
plt.show()

#%%
Idr2_vec = np.linspace(Ic,50e-6,100)
I_di_sat_vec = np.zeros([len(Idr2_vec),1])
for ii in range(len(I_di_sat_vec)):
    I_di_sat_vec[ii] = I_di_sat_of_I_dr_2(Idr2_vec[ii])

fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('I_di_sat vs Idr2')
# plt.title(title_string)
ax.plot(Idr2_vec*1e6,I_di_sat_vec*1e6, '-')   
ax.set_xlabel(r'$I_{dr2}$ [$\mu$A]')
ax.set_ylabel(r'$I_{di}^{sat}$ [$\mu$A]')
# ax.legend()    
plt.show()