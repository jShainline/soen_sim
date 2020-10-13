import numpy as np
from matplotlib import pyplot as plt
import pickle
import copy

from _plotting import plot_dend_rate_array__norm_to_phi0

from util import physical_constants, color_dictionary

p = physical_constants()
colors = color_dictionary()

plt.close('all')

#%%
Phi0 = p['Phi0__pH_ns']

#%%

num_jjs_list = [4] # [2,4]
L_left = 20
L_right = 20

do__3d_rate_arrays = False
do__squid_response = False
do__threshold = False
do__squid_threshold_composite = True
do__functional_range = True
do__saturation = False

if do__threshold == True:
    fig_th = plt.figure()
    fig_th.suptitle('DR loop flux threshold')
if do__functional_range == True:
    fig_range, axs_range = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)     
    fig_range.suptitle('DR functional range')
     
for num_jjs in num_jjs_list:

    if num_jjs == 2:       
        dI_de = 1
        I_de_0 = 52
        I_de_f = 80
        # dI_de = 5
        # I_de_0 = 63
        # I_de_f = 78
    
    if num_jjs == 4:
        dI_de = 1
        I_de_0 = 56
        I_de_f = 90
        # dI_de = 5
        # I_de_0 = 63
        # I_de_f = 78
    
    I_de_vec = np.arange(I_de_0,I_de_f+dI_de,dI_de)
    num_I_de = len(I_de_vec)
    
    if do__squid_response == True:
        fig_sq = plt.figure()
        # fig_sq.suptitle('squid response; num_jjs = {}'.format(num_jjs))
    if do__saturation == True:
        fig_sat = plt.figure()       
        fig_sat.suptitle('DI loop saturation; num_jjs = {}'.format(num_jjs))   
    if do__squid_threshold_composite == True:
        fig_comp, axs_comp = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)     
        fig_comp.suptitle('num_jjs = {}'.format(num_jjs))  
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
        if do__squid_response == True or do__squid_threshold_composite == True:
            # rate_vec__temp = np.zeros([len(influx_list)])
            rate_vec__temp = np.zeros([len(influx_list)])
            for qq in range(len(influx_list)):
                rate_vec__temp[qq] = np.asarray(rate_array[qq][1])
            # rate_vec__temp = np.insert(np.insert(rate_vec__temp,0,0),0,0)
            # influx_list__2 = copy.deepcopy(influx_list)
            # influx_list__2.insert(0,influx_list__2[0])
            # influx_list__2.insert(0,0)
            tn = 0
            if num_jjs == 4:
                if I_de > 84:
                    tn = 1
            # if num_jjs == 2:
            #     if I_de > 82:
            #         tn = 1
            rate_vec__temp = np.concatenate((np.flipud(rate_vec__temp[tn:]),rate_vec__temp[tn:]))
            influx_list__2 = np.concatenate((-np.flipud(np.asarray(influx_list[tn:])),np.asarray(influx_list[tn:])))
            # rate_vec__temp = np.concatenate((np.flipud(rate_vec__temp),rate_vec__temp))
            # influx_list__2 = np.concatenate((np.flipud(np.asarray(influx_list)),-np.asarray(influx_list)))
            
            if do__squid_response == True :
                ax_sq = fig_sq.gca()    
                ax_sq.plot(influx_list__2/p['Phi0__pH_ns'],np.asarray(rate_vec__temp), '-', label = 'Ide = {:2.0f}uA'.format(I_de)) # , label = legend_text
            if do__squid_threshold_composite == True:
                axs_comp[0].plot(influx_list__2/p['Phi0__pH_ns'],np.asarray(rate_vec__temp), '-', label = 'Ide = {:2.0f}uA'.format(I_de))
        
        # Phi_th versus I_de    
        if do__threshold == True or do__squid_threshold_composite == True or do__functional_range == True:
            Phi_th_vec[ii] = influx_list[1]
            
        # I_di_sat versus applied flux
        if do__saturation == True:
            I_di_sat_vec = np.zeros([len(influx_list)])
            for qq in range(len(influx_list)):
                I_di_sat_vec[qq] = I_di_array[qq][-1]
             
            ax_sat = fig_sat.gca()    
            ax_sat.plot(np.asarray(influx_list)/p['Phi0__pH_ns'],I_di_sat_vec, '-o', label = 'Ide = {:2.0f}uA'.format(I_de)) # , label = legend_text
    
    # squid response curves of rate vs applied flux for I_di = 0  
    if do__squid_response == True:       
        ax_sq.set_xlim([-1/2,1/2])   
        ax_sq.set_xlabel(r'$\Phi_a/\Phi_0$')
        ax_sq.set_ylabel(r'$R_{fq}$ [fluxons per ns]')
        # ax_sq.legend()
        if num_jjs == 2:
            ax_sq.set_ylim([0,65])
        if num_jjs == 4:
            ax_sq.set_ylim([0,65])
        plt.show()
             
    # Phi_th versus I_de  
    if do__threshold == True or do__squid_threshold_composite == True:
        Phi_th_vec__two_sided = np.concatenate((-np.flipud(np.asarray(Phi_th_vec)),np.asarray(Phi_th_vec)))
        I_de_vec__two_sided = np.concatenate((np.flipud(I_de_vec),I_de_vec))  
        # ax_th.plot(I_de_vec,np.asarray(Phi_th_vec)/p['Phi0__pH_ns'], '-o', label = 'num_jjs = {:d}'.format(num_jjs)) # , label = legend_text
        # ax_th.set_xlabel(r'$I^{de}$ [$\mu$A]')
        # ax_th.set_ylabel(r'$\Phi^{dr}_{th}/\Phi_0$')
        
        if do__threshold == True:
            ax_th = fig_th.gca()
            ax_th.plot(np.asarray(Phi_th_vec__two_sided)/Phi0,I_de_vec__two_sided, '-o', label = 'num_jjs = {:d}'.format(num_jjs)) # , label = legend_text
            ax_th.set_ylabel(r'$I^{de}$ [$\mu$A]')
            ax_th.set_xlabel(r'$\Phi^{dr}_{th}/\Phi_0$')
            # ax_th.legend()
            ax_th.set_xticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            plt.show()
         
    # I_di_sat versus applied flux
    if do__saturation == True:
        ax_sat.set_xlim([0,1/2])   
        ax_sat.set_xlabel(r'$\Phi_a/\Phi_0$')
        ax_sat.set_ylabel(r'$I^{di}_{sat}$ [$\mu$A]')
        # ax_sat.legend()
        plt.show()
        
    # two-panel plot
    if do__squid_threshold_composite == True:        
        axs_comp[0].set_ylabel(r'$R_{fq}$ [fluxons per ns]')
        
        axs_comp[1].plot(np.asarray(Phi_th_vec__two_sided)/Phi0,I_de_vec__two_sided, '-o', color = colors['blue3'], label = '$\Phi^{dr}_{th}$') # , label = legend_text  
        axs_comp[1].legend()
        axs_comp[1].set_ylabel(r'$I^{de}$ [$\mu$A]')        
        
        axs_comp[1].set_xlim([-1/2,1/2])   
        axs_comp[1].set_xlabel(r'$\Phi_a/\Phi_0$')
        axs_comp[1].set_xticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
        
        if num_jjs == 2:
            axs_comp[0].set_ylim([0,65])
            axs_comp[1].set_ylim([51,81])
        if num_jjs == 4:
            axs_comp[0].set_ylim([0,65])
            axs_comp[1].set_ylim([55,86])
        
        plt.show()
            
    if do__functional_range == True:   
        area_list = []
        I_de_list = []
        kk_last = 0
        for kk in range(len(I_de_vec)):
            Phi_ex_vec = [Phi_th_vec[kk]/Phi0,1/2,1/2,Phi_th_vec[kk]/Phi0,Phi_th_vec[kk]/Phi0]
            Phi_ih_vec = [1/2-Phi_th_vec[kk]/Phi0,1/2-Phi_th_vec[kk]/Phi0,2*Phi_th_vec[kk]/Phi0,2*Phi_th_vec[kk]/Phi0,1/2-Phi_th_vec[kk]/Phi0]
            if 1/2-Phi_th_vec[kk]/Phi0 < 2*Phi_th_vec[kk]/Phi0:
                area_list.append( (1/2-Phi_th_vec[kk]/Phi0)*(2*Phi_th_vec[kk]/Phi0-(1/2-Phi_th_vec[kk]/Phi0)) )
                I_de_list.append(I_de_vec[kk])
                axs_range[0].plot(Phi_ex_vec,Phi_ih_vec,'-', label = 'Ide = {:5.2f}'.format(I_de_vec[kk]))
                kk_last = kk 
                Phi_ex_vec__use = Phi_ex_vec
        tn = 1/2-Phi_th_vec[kk_last]/Phi0 + (2*Phi_th_vec[kk_last]/Phi0-(1/2-Phi_th_vec[kk_last]/Phi0))/2
        axs_range[0].plot([np.min(Phi_ex_vec__use),np.max(Phi_ex_vec__use)],[tn,tn],':', color = colors['greengrey3'], label = 'op = {:06.4f}'.format(tn))
        axs_range[0].margins(0.05,0.05)
            
        # ax_range.set_xlim([0,1/2])
        axs_range[0].set_xlabel('$\Phi_{ex}/\Phi_0$')
        # ax_range.set_ylim([0,1/2])        
        axs_range[0].set_ylabel('$\Phi_{ih}/\Phi_0$')
        axs_range[0].legend()
        
        axs_range[1].plot(I_de_list,area_list,'o-', color = colors['blue3'], label = 'max at Ide = {:5.2f}uA'.format( I_de_list[(np.asarray(area_list)).argmax()] ))
        axs_range[1].set_xlabel('$I_{de}$ [$\mu$A]')
        axs_range[1].set_ylabel('$\Delta \Phi^{ex} \Delta \Phi^{ih}/\Phi_0^2$')
        axs_range[1].legend()
            
        plt.show()
        