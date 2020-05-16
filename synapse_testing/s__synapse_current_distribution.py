#%%
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.signal import find_peaks

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_error_mat, plot_fq_peaks, plot_fq_peaks__dt_vs_bias, plot_wr_data__currents_and_voltages
from _functions import Ljj, synapse_current_distribution, read_wr_data
from _functions__more import dendrite_model__parameter_sweep
from util import physical_constants
p = physical_constants()

plt.close('all')

#%% compare initial currents in synapse circuit

I_drive = 0

Ic = 40
I_bias_list = [28,36,35]
L_jtl1 = 77.5
L_jtl2 = 77.5
L_si = 77.5e3


#%% no fluxons in SI loop (I_si = 0)

Ijsf = I_bias_list[0]
Ijtl = I_bias_list[1]
Ijsi = I_bias_list[2]
for ii in range(5):
    # print('Ijsf = {}'.format(Ijsf))
    I1, I2, I3, Ijsf, Ijtl, Ijsi, Ljsf, Ljtl, Ljsi = synapse_current_distribution(Ic,L_jtl1,L_jtl2,L_si,I_bias_list,Ijsf,Ijtl,Ijsi)

Ijsf_wr = 29.0600
Ijtl_wr = 34.9491
Ijsi_wr = 34.9909
I1_wr = -1.0600
I2_wr = -9.1300e-3
Isi_wr = 0
print('\n\nbefore flux:')
print('Ijsf_soen = {:7.4f}uA, Ijsf_wr = {:7.4f}uA; Ijsf_soen-Ijsf_wr = {:3.0f}nA'.format(Ijsf,Ijsf_wr,(Ijsf-Ijsf_wr)*1e3))
print('Ijtl_soen = {:7.4f}uA, Ijtl_wr = {:7.4f}uA; Ijtl_soen-Ijtl_wr = {:3.0f}nA'.format(Ijtl,Ijtl_wr,(Ijtl-Ijtl_wr)*1e3))
print('Ijsi_soen = {:7.4f}uA, Ijsi_wr = {:7.4f}uA; Ijsi_soen-Ijsi_wr = {:3.0f}nA'.format(Ijsi,Ijsi_wr,(Ijsi-Ijsi_wr)*1e3))
print('I1_soen = {:7.4f}uA, I1_wr = {:7.4f}uA; I1_soen-I1_wr = {:3.0f}nA'.format(I1,I1_wr,(I1-I1_wr)*1e3))
print('I2_soen = {:7.4f}uA, I2_wr = {:7.4f}uA; I2_soen-I2_wr = {:3.0f}nA'.format(I2,I2_wr,(I2-I2_wr)*1e3))
 

#%% after flux added to SI loop (I_si = 5.7890 uA)

Isi = 5.7890

I_bias_list[2] -= Isi
Ijsf = I_bias_list[0]
Ijtl = I_bias_list[1]
Ijsi = I_bias_list[2]
for ii in range(5):
    # print('Ijsf = {}'.format(Ijsf))
    I1, I2, I3, Ijsf, Ijtl, Ijsi, Ljsf, Ljtl, Ljsi = synapse_current_distribution(Ic,L_jtl1,L_jtl2,L_si,I_bias_list,Ijsf,Ijtl,Ijsi)

# I_loop2_from_si = ( Ljsi/(L_jtl2+Ljtl) )*Isi
# I_loop1_from_loop2 = ( Ljtl/(L_jtl1+Ljsf) )*I_loop2_from_si
# Ijsf -= I_loop1_from_loop2
# Ijtl += I_loop1_from_loop2-I_loop2_from_si
# Ijsi -= Isi-I_loop2_from_si

Ijsf_wr = 28.9386
Ijtl_wr = 34.2875
Ijsi_wr = 29.9851
print('\nafter flux:')
print('Ijsf_soen = {:7.4f}uA, Ijsf_wr = {:7.4f}uA; Ijsf_soen-Ijsf_wr = {:3.0f}nA'.format(Ijsf,Ijsf_wr,(Ijsf-Ijsf_wr)*1e3))
print('Ijtl_soen = {:7.4f}uA, Ijtl_wr = {:7.4f}uA; Ijtl_soen-Ijtl_wr = {:3.0f}nA'.format(Ijtl,Ijtl_wr,(Ijtl-Ijtl_wr)*1e3))
print('Ijsi_soen = {:7.4f}uA, Ijsi_wr = {:7.4f}uA; Ijsi_soen-Ijsi_wr = {:3.0f}nA'.format(Ijsi,Ijsi_wr,(Ijsi-Ijsi_wr)*1e3))

