import numpy as np
from matplotlib import pyplot as plt
import pickle
import copy

from _plotting import plot_dend_rate_array__norm_to_phi0

from util import physical_constants

p = physical_constants()

plt.close('all')

#%%

num_jjs_list = [2,4]
L_left = 20
L_right = 20

do__3d_rate_arrays = True
do__squid_response = True

for num_jjs in num_jjs_list:

    if num_jjs == 2:
        dI_de = 2
        I_de_vec = np.arange(70,80+dI_de,dI_de)
        num_I_de = len(I_de_vec)
    if num_jjs == 4:
        dI_de = 2
        I_de_vec = np.arange(70,80+dI_de,dI_de)
        num_I_de = len(I_de_vec)
    
    fig_sq = plt.figure()
    fig_sq.suptitle('num_jjs = {}'.format(num_jjs)) 
    fig_sat = plt.figure()       
    fig_sat.suptitle('num_jjs = {}'.format(num_jjs)) 
    fig_th = plt.figure()
    fig_th.suptitle('num_jjs = {}'.format(num_jjs)) 
    Phi_th_vec = np.zeros([len(I_de_vec)])
    for ii in range(len(I_de_vec)):
        I_de = I_de_vec[ii]        
        
        file_name = 'master_dnd_rate_array_{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}.soen'.format(num_jjs,L_left,L_right,I_de)
        with open('../_circuit_data/'+file_name, 'rb') as data_file:         
            data_array = pickle.load(data_file)
            rate_array = data_array['rate_array']
            influx_list = data_array['influx_list']
            I_di_array = data_array['I_di_array']
            
        # 3D plot of rate array            
        if do__3d_rate_arrays == True:
            plot_dend_rate_array__norm_to_phi0(file_name = file_name)
            # plot_dend_rate_array(file_name = file_name)
          
            
        # squid response curves of rate vs applied flux for I_di = 0
        if do__squid_response == True:
            rate_vec__temp = np.zeros([len(influx_list)])
            for qq in range(len(influx_list)):
                rate_vec__temp[qq] = rate_array[qq][1]
            rate_vec__temp = np.insert(np.insert(rate_vec__temp,0,0),0,0)
            influx_list__2 = copy.deepcopy(influx_list)
            influx_list__2.insert(0,influx_list__2[0])
            influx_list__2.insert(0,0)
             
            ax_sq = fig_sq.gca()    
        ax_sq.plot(1e-18*np.asarray(influx_list__2)/p['Phi0'],np.asarray(rate_vec__temp)*1e-3, '-o', label = 'Ide = {:2.0f}uA'.format(I_de)) # , label = legend_text

        # I_di_sat versus applied flux
        I_di_sat_vec = np.zeros([len(influx_list)])
        for qq in range(len(influx_list)):
            I_di_sat_vec[qq] = I_di_array[qq][-1]
         
        ax_sat = fig_sat.gca()    
        ax_sat.plot(1e-18*np.asarray(influx_list)/p['Phi0'],I_di_sat_vec, '-o', label = 'Ide = {:2.0f}uA'.format(I_de)) # , label = legend_text
                    
        # Phi_th versus I_de    
        Phi_th_vec[ii] = influx_list[1]
        ax_th = fig_th.gca()    
        ax_th.plot(I_de_vec,1e-18*np.asarray(Phi_th_vec)/p['Phi0'], '-o', label = 'num_jjs = {:1d}'.format(num_jjs)) # , label = legend_text
    
    # squid response curves of rate vs applied flux for I_di = 0         
    ax_sq.set_xlim([0,1/2])   
    ax_sq.set_xlabel(r'$\Phi_a/\Phi_0$')
    ax_sq.set_ylabel(r'$R_{fq}$ [fluxons per ns]')
    ax_sq.legend()
    plt.show()
         
    # I_di_sat versus applied flux
    ax_sat.set_xlim([0,1/2])   
    ax_sat.set_xlabel(r'$\Phi_a/\Phi_0$')
    ax_sat.set_ylabel(r'$I^{di}_{sat}$ [$\mu$A]')
    ax_sat.legend()
    plt.show()
             
    # Phi_th versus I_de     
    ax_th.set_xlabel(r'$I^{de}$ [$\mu$A]')
    ax_th.set_ylabel(r'$\Phi^{dr}_{th}/\Phi_0$')
    ax_th.legend()
    plt.show()
        