import numpy as np
from matplotlib import pyplot as plt
import pickle
import copy

from _plotting import plot_dend_rate_array__norm_to_phi0

from util import physical_constants

p = physical_constants()

plt.close('all')

#%%

num_jjs_list = [2,4] # [2,4]
L_left = 20
L_right = 20

do__3d_rate_arrays = True
do__squid_response = False
do__saturation = False
do__threshold = False

if do__threshold == True:
    fig_th = plt.figure()
    fig_th.suptitle('DR loop flux threshold')
for num_jjs in num_jjs_list:

    if num_jjs == 2:       
        # dI_de = 1
        # I_de_0 = 52
        # I_de_f = 80
        dI_de = 5
        I_de_0 = 63
        I_de_f = 78
    
    if num_jjs == 4:
        # dI_de = 1
        # I_de_0 = 56
        # I_de_f = 90
        dI_de = 5
        I_de_0 = 63
        I_de_f = 78
    
    I_de_vec = np.arange(I_de_0,I_de_f+dI_de,dI_de)
    num_I_de = len(I_de_vec)
    
    if do__squid_response == True:
        fig_sq = plt.figure()
        # fig_sq.suptitle('squid response; num_jjs = {}'.format(num_jjs))
    if do__saturation == True:
        fig_sat = plt.figure()       
        fig_sat.suptitle('DI loop saturation; num_jjs = {}'.format(num_jjs))     
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
            ax_sq.plot(1e-18*np.asarray(influx_list__2)/p['Phi0'],np.asarray(rate_vec__temp)*1e-3, '-', label = 'Ide = {:2.0f}uA'.format(I_de)) # , label = legend_text

        # I_di_sat versus applied flux
        if do__saturation == True:
            I_di_sat_vec = np.zeros([len(influx_list)])
            for qq in range(len(influx_list)):
                I_di_sat_vec[qq] = I_di_array[qq][-1]
             
            ax_sat = fig_sat.gca()    
            ax_sat.plot(1e-18*np.asarray(influx_list)/p['Phi0'],I_di_sat_vec, '-o', label = 'Ide = {:2.0f}uA'.format(I_de)) # , label = legend_text
                    
        # Phi_th versus I_de    
        if do__threshold == True:
            Phi_th_vec[ii] = influx_list[1]
    
    # squid response curves of rate vs applied flux for I_di = 0  
    if do__squid_response == True:       
        ax_sq.set_xlim([0,1/2])   
        ax_sq.set_xlabel(r'$\Phi_a/\Phi_0$')
        ax_sq.set_ylabel(r'$R_{fq}$ [fluxons per ns]')
        # ax_sq.legend()
        if num_jjs == 2:
            ax_sq.set_ylim([0,65])
        if num_jjs == 4:
            ax_sq.set_ylim([0,65])
        plt.show()
         
    # I_di_sat versus applied flux
    if do__saturation == True:
        ax_sat.set_xlim([0,1/2])   
        ax_sat.set_xlabel(r'$\Phi_a/\Phi_0$')
        ax_sat.set_ylabel(r'$I^{di}_{sat}$ [$\mu$A]')
        # ax_sat.legend()
        plt.show()
             
    # Phi_th versus I_de  
    if do__threshold == True:
        ax_th = fig_th.gca()    
        ax_th.plot(I_de_vec,1e-18*np.asarray(Phi_th_vec)/p['Phi0'], '-o', label = 'num_jjs = {:d}'.format(num_jjs)) # , label = legend_text
        ax_th.set_xlabel(r'$I^{de}$ [$\mu$A]')
        ax_th.set_ylabel(r'$\Phi^{dr}_{th}/\Phi_0$')
        # ax_th.legend()
        plt.show()
        