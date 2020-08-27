from matplotlib import pyplot as plt
import numpy as np

from util import color_dictionary, physical_constants
colors = color_dictionary()
p = physical_constants()

# plt.close('all')

#%%
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['figure.figsize'] = (15,15/1.618)

#%%
current_to_flux = 1e12*np.sqrt(200e-12*20e-12)
current_to_Phi0 = 1e-6*np.sqrt(200e-12*20e-12)/p['Phi0']  

#%% data from spice

# 2jj, L_left = 20 pH, L_right = 20 pH
I_de_vec__2jj__Llft20pH = [40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100] # uA
I_drive_min_vec__2jj__Llft20pH__minus = [0,0,0,0,0,0,-16.0,-15.0,-14.5,-13.5,-12.5,-11.5,-10.5,-9.5,-8.5,-7.5,-6.5,-5.5,-4.0,-3.0,-0.5,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0]
I_drive_min_vec__2jj__Llft20pH__plus = [0.0,0.0,0.0,0.0,0.0,0.0,16.0,15.0,14.5,13.5,12.5,11.5,10.5,9.5,8.5,7.5,6.5,5.5,4.0,3.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
I_drive_max_vec__2jj__Llft20pH__minus = [0,0,0,0,0,0,-17.0,-18.0,-19.0,-20.0,-20.5,-21.5,-22.5,-23.5,-24.5,-25.5,-26.5,-28.0,-29.0,-30.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5]
I_drive_max_vec__2jj__Llft20pH__plus = [0.0,0.0,0.0,0.0,0.0,0.0,17.0,18.0,19.0,20.0,20.5,21.5,22.5,23.5,24.5,25.5,26.5,28.0,29.0,30.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

# 2jj, L_left = 17 pH, L_right = 23 pH
I_de_vec__2jj__Llft17pH = [40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100] # uA
I_drive_min_vec__2jj__Llft17pH__minus = [0.0,0.0,0.0,0.0,0.0,0.0,-15.0,-13.5,-12.5,-11.5,-10.5,-9.5,-8.0,-7.0,-6.0,-4.5,-3.5,-2.0,-0.5,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0]
I_drive_min_vec__2jj__Llft17pH__plus = [0.0,0.0,0.0,0.0,0.0,0.0,20.0,19.5,18.5,17.5,16.5,15.5,14.5,14.0,13.0,12.0,11.0,9.5,8.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
I_drive_max_vec__2jj__Llft17pH__minus = [0.0,0.0,0.0,0.0,0.0,0.0,-16.0,-16.5,-17.5,-18.5,-19.5,-20.5,-21.0,-22.0,-23.0,-24.0,-25.0,-26.5,-27.5,-29.0,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5]
I_drive_max_vec__2jj__Llft17pH__plus = [0.0,0.0,0.0,0.0,0.0,0.0,21.0,22.0,23.5,24.5,25.5,26.5,28.0,29.0,30.0,31.5,32.5,34.0,35.5,1.5,79.5,79.5,79.5,79.5,79.5,79.5,79.5,79.5,79.5,79.5,79.5]

# 4jj, L_left = 20 pH, L_right = 20 pH
I_de_vec__4jj__Llft20pH = [40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100] # uA
I_drive_min_vec__4jj__Llft20pH__minus = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-15.5,-14.5,-13.5,-13.0,-12.0,-11.0,-10.5,-9.5,-8.5,-8.0,-7.0,-6.0,-5.0,-4.0,-2.5,-1.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0]
I_drive_min_vec__4jj__Llft20pH__plus = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,17.5,16.5,15.5,14.5,13.5,12.5,12.0,11.0,10.0,9.0,8.0,7.0,5.5,4.5,3.0,1.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
I_drive_max_vec__4jj__Llft20pH__minus = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-16.0,-17.0,-17.5,-18.5,-19.5,-20.5,-21.5,-22.5,-23.5,-24.5,-25.5,-26.5,-27.5,-28.5,-30.0,-32.0,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5]
I_drive_max_vec__4jj__Llft20pH__plus = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,18.0,18.5,19.5,20.5,21.0,22.0,23.0,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5,32.0,79.5,79.5,79.5,79.5,79.5,79.5,79.5,79.5]

# 4jj, L_left = 17 pH, L_right = 23 pH
I_de_vec__4jj__Llft17pH = [40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100] # uA
I_drive_min_vec__4jj__Llft17pH__minus = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-14.0,-13.0,-12.0,-11.0,-10.0,-9.0,-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-0.5,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0,-0.0]
I_drive_min_vec__4jj__Llft17pH__plus = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,21.5,20.5,19.5,19.0,18.0,17.0,16.0,15.0,14.5,13.5,12.5,11.5,10.0,9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
I_drive_max_vec__4jj__Llft17pH__minus = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-14.5,-15.5,-16.0,-17.0,-18.0,-19.0,-20.0,-21.0,-21.5,-22.5,-23.5,-24.5,-25.5,-27.0,-28.0,-30.0,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5,-79.5]
I_drive_max_vec__4jj__Llft17pH__plus = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.5,1.0,3.0,79.5,79.5,79.5,79.5,79.5,79.5,79.5,79.5]
     
#%% plot
  
#2jj, L_left = 20pH  
fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('2jj, L_left = 20pH')  
 
axs[0].plot(I_de_vec__2jj__Llft20pH, np.asarray(I_drive_min_vec__2jj__Llft20pH__minus)*current_to_flux, '-', color = colors['green3'], label = 'min') 
axs[0].plot(I_de_vec__2jj__Llft20pH, np.asarray(I_drive_min_vec__2jj__Llft20pH__plus)*current_to_flux, '-', color = colors['green3'], label = 'min') 
axs[0].plot(I_de_vec__2jj__Llft20pH, np.asarray(I_drive_max_vec__2jj__Llft20pH__minus)*current_to_flux, '-', color = colors['red3'], label = 'max') 
axs[0].plot(I_de_vec__2jj__Llft20pH, np.asarray(I_drive_max_vec__2jj__Llft20pH__plus)*current_to_flux, '-', color = colors['red3'], label = 'max')
axs[0].legend()

axs[1].plot(I_de_vec__2jj__Llft20pH, np.asarray(I_drive_min_vec__2jj__Llft20pH__minus)*current_to_Phi0, '-', color = colors['green3'], label = 'min') 
axs[1].plot(I_de_vec__2jj__Llft20pH, np.asarray(I_drive_min_vec__2jj__Llft20pH__plus)*current_to_Phi0, '-', color = colors['green3'], label = 'min')
axs[1].plot(I_de_vec__2jj__Llft20pH, np.asarray(I_drive_max_vec__2jj__Llft20pH__minus)*current_to_Phi0, '-', color = colors['red3'], label = 'max') 
axs[1].plot(I_de_vec__2jj__Llft20pH, np.asarray(I_drive_max_vec__2jj__Llft20pH__plus)*current_to_Phi0, '-', color = colors['red3'], label = 'max')

axs[0].set_ylabel(r'$\Phi^{dr}$ [$\mu$A pH]')
axs[1].set_ylabel(r'$\Phi^{dr}/\Phi_0$')
axs[1].set_xlabel(r'$I_{de}$ [$\mu$A]')
axs[1].set_xlim([50,80])

plt.show()
  
#2jj, L_left = 17pH  
fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('2jj, L_left = 17pH')  
 
axs[0].plot(I_de_vec__2jj__Llft17pH, np.asarray(I_drive_min_vec__2jj__Llft17pH__minus)*current_to_flux, '-', color = colors['green3'], label = 'min') 
axs[0].plot(I_de_vec__2jj__Llft17pH, np.asarray(I_drive_min_vec__2jj__Llft17pH__plus)*current_to_flux, '-', color = colors['green3'], label = 'min') 
axs[0].plot(I_de_vec__2jj__Llft17pH, np.asarray(I_drive_max_vec__2jj__Llft17pH__minus)*current_to_flux, '-', color = colors['red3'], label = 'max') 
axs[0].plot(I_de_vec__2jj__Llft17pH, np.asarray(I_drive_max_vec__2jj__Llft17pH__plus)*current_to_flux, '-', color = colors['red3'], label = 'max')
axs[0].legend()

axs[1].plot(I_de_vec__2jj__Llft17pH, np.asarray(I_drive_min_vec__2jj__Llft17pH__minus)*current_to_Phi0, '-', color = colors['green3'], label = 'min') 
axs[1].plot(I_de_vec__2jj__Llft17pH, np.asarray(I_drive_min_vec__2jj__Llft17pH__plus)*current_to_Phi0, '-', color = colors['green3'], label = 'min')
axs[1].plot(I_de_vec__2jj__Llft17pH, np.asarray(I_drive_max_vec__2jj__Llft17pH__minus)*current_to_Phi0, '-', color = colors['red3'], label = 'max') 
axs[1].plot(I_de_vec__2jj__Llft17pH, np.asarray(I_drive_max_vec__2jj__Llft17pH__plus)*current_to_Phi0, '-', color = colors['red3'], label = 'max')

axs[0].set_ylabel(r'$\Phi^{dr}$ [$\mu$A pH]')
axs[1].set_ylabel(r'$\Phi^{dr}/\Phi_0$')
axs[1].set_xlabel(r'$I_{de}$ [$\mu$A]')
axs[1].set_xlim([50,80])

plt.show()
  
#4jj, L_left = 20pH  
fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('4jj, L_left = 20pH')  
 
axs[0].plot(I_de_vec__4jj__Llft20pH, np.asarray(I_drive_min_vec__4jj__Llft20pH__minus)*current_to_flux, '-', color = colors['green3'], label = 'min') 
axs[0].plot(I_de_vec__4jj__Llft20pH, np.asarray(I_drive_min_vec__4jj__Llft20pH__plus)*current_to_flux, '-', color = colors['green3'], label = 'min') 
axs[0].plot(I_de_vec__4jj__Llft20pH, np.asarray(I_drive_max_vec__4jj__Llft20pH__minus)*current_to_flux, '-', color = colors['red3'], label = 'max') 
axs[0].plot(I_de_vec__4jj__Llft20pH, np.asarray(I_drive_max_vec__4jj__Llft20pH__plus)*current_to_flux, '-', color = colors['red3'], label = 'max')
axs[0].legend()

axs[1].plot(I_de_vec__4jj__Llft20pH, np.asarray(I_drive_min_vec__4jj__Llft20pH__minus)*current_to_Phi0, '-', color = colors['green3'], label = 'min') 
axs[1].plot(I_de_vec__4jj__Llft20pH, np.asarray(I_drive_min_vec__4jj__Llft20pH__plus)*current_to_Phi0, '-', color = colors['green3'], label = 'min')
axs[1].plot(I_de_vec__4jj__Llft20pH, np.asarray(I_drive_max_vec__4jj__Llft20pH__minus)*current_to_Phi0, '-', color = colors['red3'], label = 'max') 
axs[1].plot(I_de_vec__4jj__Llft20pH, np.asarray(I_drive_max_vec__4jj__Llft20pH__plus)*current_to_Phi0, '-', color = colors['red3'], label = 'max')

axs[0].set_ylabel(r'$\Phi^{dr}$ [$\mu$A pH]')
axs[1].set_ylabel(r'$\Phi^{dr}/\Phi_0$')
axs[1].set_xlabel(r'$I_{de}$ [$\mu$A]')
axs[1].set_xlim([50,90])

plt.show()
  
#4jj, L_left = 17pH  
fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
fig.suptitle('4jj, L_left = 17pH')  
 
axs[0].plot(I_de_vec__4jj__Llft17pH, np.asarray(I_drive_min_vec__4jj__Llft17pH__minus)*current_to_flux, '-', color = colors['green3'], label = 'min') 
axs[0].plot(I_de_vec__4jj__Llft17pH, np.asarray(I_drive_min_vec__4jj__Llft17pH__plus)*current_to_flux, '-', color = colors['green3'], label = 'min') 
axs[0].plot(I_de_vec__4jj__Llft17pH, np.asarray(I_drive_max_vec__4jj__Llft17pH__minus)*current_to_flux, '-', color = colors['red3'], label = 'max') 
axs[0].plot(I_de_vec__4jj__Llft17pH, np.asarray(I_drive_max_vec__4jj__Llft17pH__plus)*current_to_flux, '-', color = colors['red3'], label = 'max')
axs[0].legend()

axs[1].plot(I_de_vec__4jj__Llft17pH, np.asarray(I_drive_min_vec__4jj__Llft17pH__minus)*current_to_Phi0, '-', color = colors['green3'], label = 'min') 
axs[1].plot(I_de_vec__4jj__Llft17pH, np.asarray(I_drive_min_vec__4jj__Llft17pH__plus)*current_to_Phi0, '-', color = colors['green3'], label = 'min')
axs[1].plot(I_de_vec__4jj__Llft17pH, np.asarray(I_drive_max_vec__4jj__Llft17pH__minus)*current_to_Phi0, '-', color = colors['red3'], label = 'max') 
axs[1].plot(I_de_vec__4jj__Llft17pH, np.asarray(I_drive_max_vec__4jj__Llft17pH__plus)*current_to_Phi0, '-', color = colors['red3'], label = 'max')

axs[0].set_ylabel(r'$\Phi^{dr}$ [$\mu$A pH]')
axs[1].set_ylabel(r'$\Phi^{dr}/\Phi_0$')
axs[1].set_xlabel(r'$I_{de}$ [$\mu$A]')
axs[1].set_xlim([50,90])

plt.show()

