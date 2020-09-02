import numpy as np
from matplotlib import pyplot as plt
import pickle

from _plotting import plot_dend_rate_array__norm_to_phi0

from util import physical_constants

p = physical_constants()

#%% 3D plot of rate array
num_jjs = 4
L_left = 20
L_right = 20
I_de = 74

file_name = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}.soen'.format(num_jjs,L_left,L_right,I_de)
# plot_dend_rate_array(file_name = file_name)

plot_dend_rate_array__norm_to_phi0(file_name = file_name)
        
# plt.close('all')
# for pp in range(num_L):
#     for qq in range(num_I_de):
        
#         file_name = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}.soen'.format(num_jjs,L_left_list[pp],L_right_list[pp],I_de_list[qq])
#         plot_dend_rate_array(file_name = file_name)
        # plot_dend_rate_array__norm_to_phi0(file_name = file_name)
        
#%% squid response curves of rate vs applied flux for I_di = 0
num_jjs = 2
L_left = 20
L_right = 20

dI_de = 2
I_de_vec = np.arange(70,80+dI_de,dI_de)
num_I_de = len(I_de_vec)


fig = plt.figure()
for ii in range(len(I_de_vec)):
    file_name = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}.soen'.format(num_jjs,L_left,L_right,I_de_vec[ii])
    with open('../_circuit_data/'+file_name, 'rb') as data_file:         
        data_array = pickle.load(data_file)
        # data_array = load_session_data(kwargs['file_name'])
        rate_array = data_array['rate_array']
        influx_list = data_array['influx_list']
        I_di_array = data_array['I_di_array']
    
    rate_vec__temp = np.zeros([len(influx_list)])
    for qq in range(len(influx_list)):
        rate_vec__temp[qq] = rate_array[qq][1]
    rate_vec__temp = np.insert(np.insert(rate_vec__temp,0,0),0,0)
    influx_list.insert(0,influx_list[0])
    influx_list.insert(0,0)
     
    ax = fig.gca()    
    ax.plot(1e-18*np.asarray(influx_list)/p['Phi0'],np.asarray(rate_vec__temp)*1e-3, '-o', label = 'Ide = {:2.0f}uA'.format(I_de_vec[ii])) # , label = legend_text
         
ax.set_xlim([0,1/2])   
ax.set_xlabel(r'$\Phi_a/\Phi_0$')
ax.set_ylabel(r'$R_{fq}$ [fluxons per ns]')
ax.legend()
plt.show()
        
#%% I_di_sat versus applied flux
num_jjs = 4
L_left = 20
L_right = 20

dI_de = 2
I_de_vec = np.arange(70,80+dI_de,dI_de)
num_I_de = len(I_de_vec)


fig = plt.figure()
for ii in range(len(I_de_vec)):
    file_name = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}.soen'.format(num_jjs,L_left,L_right,I_de_vec[ii])
    with open('../_circuit_data/'+file_name, 'rb') as data_file:         
        data_array = pickle.load(data_file)
        # data_array = load_session_data(kwargs['file_name'])
        rate_array = data_array['rate_array']
        influx_list = data_array['influx_list']
        I_di_array = data_array['I_di_array']
    
    I_di_sat_vec = np.zeros([len(influx_list)])
    for qq in range(len(influx_list)):
        I_di_sat_vec[qq] = I_di_array[qq][-1]
     
    ax = fig.gca()    
    ax.plot(1e-18*np.asarray(influx_list)/p['Phi0'],I_di_sat_vec, '-o', label = 'Ide = {:2.0f}uA'.format(I_de_vec[ii])) # , label = legend_text
         
ax.set_xlim([0,1/2])   
ax.set_xlabel(r'$\Phi_a/\Phi_0$')
ax.set_ylabel(r'$I^{di}_{sat}$ [$\mu$A]')
ax.legend()
plt.show()

#%% Phi_th versus I_de
num_jjs_vec = [2,4]
L_left = 20
L_right = 20

dI_de = 2
I_de_vec = np.arange(70,80+dI_de,dI_de)
num_I_de = len(I_de_vec)


fig = plt.figure()
for qq in range(len(num_jjs_vec)):
    num_jjs = num_jjs_vec[qq]
    
    Phi_th_vec = np.zeros([len(I_de_vec)])
    for ii in range(len(I_de_vec)):
        file_name = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}.soen'.format(num_jjs,L_left,L_right,I_de_vec[ii])
        with open('../_circuit_data/'+file_name, 'rb') as data_file:         
            data_array = pickle.load(data_file)
            # data_array = load_session_data(kwargs['file_name'])
            rate_array = data_array['rate_array']
            influx_list = data_array['influx_list']
            I_di_array = data_array['I_di_array']
        
        Phi_th_vec[ii] = influx_list[1]
         
    ax = fig.gca()    
    ax.plot(I_de_vec,1e-18*np.asarray(Phi_th_vec)/p['Phi0'], '-o', label = 'num_jjs = {:1d}'.format(num_jjs_vec[qq])) # , label = legend_text
         
# ax.set_xlim([0,1/2])   
ax.set_xlabel(r'$I^{de}$ [$\mu$A]')
ax.set_ylabel(r'$\Phi^{dr}_{th}/\Phi_0$')
ax.legend()
plt.show()
        