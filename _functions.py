#%%
import numpy as np
import pickle
import time
from matplotlib import pyplot as plt
from pylab import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from util import physical_constants
from _plotting import plot_wr_comparison, plot_error_mat # plot_dendritic_drive, 

#%%

def neuron_time_stepper(time_vec,neuron_object):

    print('running neuron simulation ...\n') 
    t_init = time.time()

    current_conversion = 1e6
    inductance_conversion = 1e12
    time_conversion = 1e6
    
    n = neuron_object
    nt = len(time_vec)
    dt = time_conversion*(time_vec[1]-time_vec[0])
    # print('dt = {}'.format(dt))
    
    p = physical_constants()
    Phi0 = p['Phi0']
    
    Ic = 40 # so dumb to hard-code this here
    # I_sf_hyst = 1.1768 # hysteresis of junctions with Ic = 40uA and beta_c = 0.95
    # I_reset = Ic - I_sf_hyst
    
    #------------------
    # load synapse data
    #------------------
    print('loading spd response data')
    num_jjs = 3
    if num_jjs == 1:
        file_string__spd = 'master__syn__spd_response__1jj__dt{:04.0f}ps.soen'.format(dt*1e6)
        file_string__rate_array = 'master__syn__rate_array__1jj__Isipad0010nA.soen'
    elif num_jjs == 2:
        file_string__spd = 'master__syn__spd_response__2jj__dt{:04.0f}ps.soen'.format(dt*1e6)
        file_string__rate_array = 'master__syn__rate_array__2jj__Isipad0010nA.soen'
    elif num_jjs == 3:
        file_string__spd = 'master__syn__spd_response__3jj__dt{:04.0f}ps.soen'.format(dt*1e6)
        file_string__rate_array = 'master__syn__rate_array__3jj__Isipad0010nA.soen'
    
    with open('../_circuit_data/{}'.format(file_string__spd), 'rb') as data_file:         
        data_array__spd = pickle.load(data_file)
        
    spd_response_array = data_array__spd['spd_response_array'] # entries have units of uA
    I_sy_list__spd = data_array__spd['I_sy_list'] # entries have units of uA
    # print('I_sy_list__spd = {}'.format(I_sy_list__spd))
    spd_t = data_array__spd['time_vec'] # entries have units of us
    spd_duration = spd_t[-1]
    
    print('loading synapse rate array')
    with open('../_circuit_data/{}'.format(file_string__rate_array), 'rb') as data_file:         
        data_array__rate = pickle.load(data_file)                        
        
    I_si_array = data_array__rate['I_si_array'] # entries have units of uA
    I_drive_list__syn = data_array__rate['I_drive_list'] # entries have units of uA
    rate_array__syn = data_array__rate['rate_array'] # entries have units of fluxons per microsecond
    #----------------------
    # end load synapse data
    #----------------------    
           
    #-------------------
    # load dendrite data
    #-------------------
    print('loading dendrite rate array')
    with open('../_circuit_data/master__dnd__rate_array.soen', 'rb') as data_file:         
        data_array_imported = pickle.load(data_file)
    
    I_di_array = data_array_imported['I_di_array']
    # I_drive_list__dend = data_array_imported['I_drive_list']
    influx_list__dend = data_array_imported['influx_list']
    rate_array__dend = data_array_imported['rate_array']   
    
    # print('min(influx_list__dend) = {}'.format(min(influx_list__dend)))
    # print('max(influx_list__dend) = {}'.format(max(influx_list__dend)))
    #-----------------------
    # end load dendrite data
    #-----------------------
    
    #initialize all direct synapses
    for sy_name in n.input_synaptic_connections:
        # print('sy_name = {}'.format(sy_name))
        
        #change units to microamps and picohenries
        n.synapses[sy_name].I_sy = current_conversion*n.synapses[sy_name].I_sy
        n.synapses[sy_name].I_fq = current_conversion*Phi0/(n.synapses[sy_name].L_si)
        n.synapses[sy_name].L_si = inductance_conversion*n.synapses[sy_name].L_si
        n.synapses[sy_name].tau_si = time_conversion*n.synapses[sy_name].tau_si
        n.synapses[sy_name].M = inductance_conversion*n.dendrites['{}__d'.format(n.name)].input_synaptic_inductances[sy_name][1]*np.sqrt(n.synapses[sy_name].integration_loop_output_inductance*n.dendrites['{}__d'.format(n.name)].input_synaptic_inductances[sy_name][0])
        
        n.synapses[sy_name].I_si_vec = np.zeros([nt])
        n.synapses[sy_name].I_spd_vec = np.zeros([nt])
        n.synapses[sy_name].I_sf_vec = np.zeros([nt])
        n.synapses[sy_name].j_sf_state = ['below_Ic']
        n.synapses[sy_name].I_sy_ind_spd = (np.abs(I_sy_list__spd[:] - current_conversion*n.synapses[sy_name].I_sy)).argmin()
        n.synapses[sy_name].spd_i = spd_response_array[n.synapses[sy_name].I_sy_ind_spd]
        n.synapses[sy_name].st_ind_last = 0
        n.synapses[sy_name].spd_current_memory = 0
    
    #initialize all dendrites and their synapses
    for de_name in n.input_dendritic_connections:
        # print('de_name = {}'.format(de_name))
        
        n.dendrites[de_name].I_fq = current_conversion*Phi0/n.dendrites[de_name].integration_loop_total_inductance
        n.dendrites[de_name].I_di_vec = np.zeros([nt])
        n.dendrites[de_name].I_drive_vec = np.zeros([nt])
        n.dendrites[de_name].influx_vec = np.zeros([nt])
        n.dendrites[de_name].tau_di = current_conversion*n.dendrites[de_name].integration_loop_time_constant
        n.dendrites[de_name].M = inductance_conversion*n.dendrites['{}__d'.format(n.name)].input_dendritic_inductances[de_name][1]*np.sqrt(n.dendrites[de_name].integration_loop_output_inductance*n.dendrites['{}__d'.format(n.name)].input_dendritic_inductances[de_name][0])        
        
        # t1 = n.dendrites[de_name].input_dendritic_inductances['{}__d'.format(n.name)][1]
        # print('t1 = {}'.format(t1))
        # t2 = n.dendrites['{}__d'.format(n.name)].integration_loop_output_inductance
        # print('t2 = {}'.format(t2))
        # print('n.dendrites[de_name].M = {}'.format(n.dendrites[de_name].M))
        
        for de__sy_name in n.dendrites[de_name].input_synaptic_connections:
            # print('de__sy_name = {}'.format(de__sy_name))
            
            #change units to microamps and picohenries
            n.synapses[de__sy_name].I_sy = current_conversion*n.synapses[sy_name].I_sy
            n.synapses[de__sy_name].I_fq = current_conversion*Phi0/(n.synapses[sy_name].L_si)
            n.synapses[de__sy_name].L_si = inductance_conversion*n.synapses[sy_name].L_si
            n.synapses[de__sy_name].tau_si = time_conversion*n.synapses[sy_name].tau_si
            n.synapses[de__sy_name].M = inductance_conversion*n.dendrites['{}__d'.format(n.name)].input_synaptic_inductances[de__sy_name][1]*np.sqrt(n.synapses[de__sy_name].integration_loop_output_inductance*n.dendrites['{}__d'.format(n.name)].input_synaptic_inductances[de__sy_name][0])
            
            n.synapses[de__sy_name].I_si_vec = np.zeros([nt])
            n.synapses[de__sy_name].I_spd_vec = np.zeros([nt])
            n.synapses[de__sy_name].I_sf_vec = np.zeros([nt])
            n.synapses[de__sy_name].j_sf_state = ['below_Ic']
            n.synapses[de__sy_name].I_sy_ind_spd = (np.abs(I_sy_list__spd[:] - current_conversion*n.synapses[sy_name].I_sy)).argmin()
            n.synapses[de__sy_name].spd_i = spd_response_array[n.synapses[sy_name].I_sy_ind_spd]
            n.synapses[de__sy_name].st_ind_last = 0
            n.synapses[de__sy_name].spd_current_memory = 0
            
        for de__de_name in n.dendrites[de_name].input_dendritic_connections:
            # print('de__de_name = {}'.format(de__de_name))
            
            n.dendrites[de__de_name].I_fq = current_conversion*Phi0/n.dendrites[de_name].integration_loop_total_inductance
            n.dendrites[de__de_name].I_di_vec = np.zeros([nt])
            n.dendrites[de__de_name].I_drive_vec = np.zeros([nt])
            n.dendrites[de__de_name].influx_vec = np.zeros([nt])
            n.dendrites[de__de_name].tau_di = current_conversion*n.dendrites[de_name].integration_loop_time_constant
            n.dendrites[de__de_name].M = inductance_conversion*n.dendrites[de_name].input_dendritic_inductances[de__de_name][1]*np.sqrt(n.dendrites[de__de_name].integration_loop_output_inductance*n.dendrites[de_name].input_dendritic_inductances[de__de_name][0])
            
    #initialize neuron
    n.I_fq = current_conversion*Phi0/n.integration_loop_total_inductance
    n.I_ni_vec = np.zeros([nt])
    n.I_drive_vec = np.zeros([nt])
    n.influx_vec = np.zeros([nt])
    n.tau_ni = current_conversion*n.integration_loop_time_constant
    n.state = 'quiescent'
    n.spike_times = []
    
    # step through time
    print('starting time stepping ...')
    current_to_flux = inductance_conversion*np.sqrt(200e-12*10e-12)
    for ii in range(nt-1):
        
        _pt = time_vec[ii] # present time 
        # print('ii = {} of {}; t = {}'.format(ii+1,len(range(nt-1)),_pt))
        
        # step through synapses
        for sy_name in n.input_synaptic_connections:
            
            # print('sy_name = {}'.format(sy_name))
            
            spike_times = n.synapses[sy_name].input_signal.spike_times

            # print('tau_si = {}'.format(n.synapses[sy_name].tau_si))
            # print('L_jtl1 = {}'.format(n.synapses[sy_name].L_jtl1))
            # print('L_jtl2 = {}'.format(n.synapses[sy_name].L_jtl1))
            # print('L_si = {}'.format(n.synapses[sy_name].L_si))
            # print('I_sy = {}'.format(n.synapses[sy_name].I_sy))  
            # print('I_fq = {}'.format(n.synapses[sy_name].I_fq))
            # pause(2)
    
            if len(spike_times) > 0:
                                  
                # find most recent spike time 
                n.synapses[sy_name].st_ind = (np.abs(spike_times[:] - _pt)).argmin()
                gf = 0
                if n.synapses[sy_name].st_ind == 0 and spike_times[n.synapses[sy_name].st_ind] > _pt:
                    gf = 0 # growth factor
                    # print('code 1: st_ind == 0 and spike_times[st_ind] > _pt')
                if n.synapses[sy_name].st_ind > 0 and spike_times[n.synapses[sy_name].st_ind] > _pt:
                    n.synapses[sy_name].st_ind -= 1
                    # print('code 2: st_ind > 0 and spike_times[st_ind] > _pt')
                if _pt - spike_times[n.synapses[sy_name].st_ind] > spd_duration:
                    gf = 0 # growth factor
                    # print('code 3: _pt - spike_times[st_ind] > spd_duration')
                if spike_times[n.synapses[sy_name].st_ind] <= _pt and time_conversion*(_pt - spike_times[n.synapses[sy_name].st_ind]) < spd_duration:
                    # print('code 4')
                
                    n.synapses[sy_name].dt_spk = _pt - spike_times[n.synapses[sy_name].st_ind]
                    n.synapses[sy_name].spd_t_ind  = (np.abs(spd_t[:] - time_conversion*n.synapses[sy_name].dt_spk)).argmin()
                    
                    # this block to avoid spd drive going too low at the onset of each spike 
                    if n.synapses[sy_name].st_ind - n.synapses[sy_name].st_ind_last == 1:
                        n.synapses[sy_name].spd_current = np.max([n.synapses[sy_name].I_spd_vec[ii-1],n.synapses[sy_name].spd_i[n.synapses[sy_name].spd_t_ind]])
                        n.synapses[sy_name].spd_current_memory = n.synapses[sy_name].spd_current
                    if n.synapses[sy_name].spd_current_memory > 0 and n.synapses[sy_name].spd_i[n.synapses[sy_name].spd_t_ind] < n.synapses[sy_name].spd_current_memory:
                        n.synapses[sy_name].spd_current = n.synapses[sy_name].spd_current_memory
                    else:
                        n.synapses[sy_name].spd_current = n.synapses[sy_name].spd_i[n.synapses[sy_name].spd_t_ind]
                        n.synapses[sy_name].spd_current_memory = 0
                                    
                    n.synapses[sy_name].I_spd_vec[ii] = n.synapses[sy_name].spd_current
                    n.synapses[sy_name].st_ind_last = n.synapses[sy_name].st_ind
                    I_tot = n.synapses[sy_name].spd_current+n.synapses[sy_name].I_sy
                    I_drive = I_tot-Ic # all data so far is with 40uA JJs
                    if I_drive < np.min(I_drive_list__syn):
                        gf = 0
                    else:                    
                        I_drive_ind = (np.abs(I_drive_list__syn[:] - I_drive)).argmin()
                        I_si_ind = (np.abs(np.asarray(I_si_array[I_drive_ind][:]) - n.synapses[sy_name].I_si_vec[ii])).argmin()                        
                        
                        #no interpolation
                        gf = dt*n.synapses[sy_name].I_fq*rate_array__syn[I_drive_ind][I_si_ind] # growth factor
                    
                        # linear interpolation
                        # gf = dt*I_fq*np.interp(spd_current,I_drive_list,rate_array[:][I_si_ind])
                                
                n.synapses[sy_name].I_si_vec[ii+1] = gf + (1-dt/n.synapses[sy_name].tau_si)*n.synapses[sy_name].I_si_vec[ii]
                
        # step through dendrites
        for de_name in n.input_dendritic_connections:
            # if ii == 0:
            #     print('de_name = {}'.format(de_name))
            
            # calculate drive current from synapses
            # I_syn = 0
            # for de__sy_name in n.dendrites[de_name].input_synaptic_connections:
            #     if n.synapses[de__sy_name].inhibitory_or_excitatory == 'inhibitory':
            #         _prefactor = -1
            #     elif n.synapses[de__sy_name].inhibitory_or_excitatory == 'excitatory':
            #         _prefactor = 1
            #     I_syn += _prefactor*n.synapses[de__sy_name].I_si_vec[ii+1]
                
            # # calculate drive current from other dendrites
            # I_dend = 0
            # for de__de_name in n.dendrites[de_name].input_dendritic_connections:
            #     if n.dendrites[de__de_name].inhibitory_or_excitatory == 'inhibitory':
            #         _prefactor = -1
            #     elif n.dendrites[de__de_name].inhibitory_or_excitatory == 'excitatory':
            #         _prefactor = 1                    
            #     I_dend += _prefactor*n.dendrites[de__de_name].I_di_vec[ii+1] 
            
            # # total drive to this dendrite
            # n.dendrites[de_name].I_drive_vec[ii] = I_syn+I_dend
                              
            # if n.dendrites[de_name].I_drive_vec[ii] > 18.6:
            #     _ind1 = (np.abs(I_drive_list__dend-n.dendrites[de_name].I_drive_vec[ii])).argmin()
            #     _ind2 = (np.abs(I_di_array[ind1]-n.dendrites[de_name].I_di_vec[ii])).argmin()
            #     _rate = rate_array__dend[ind1][ind2]
            #     # linear interpolation
            #     # rate = np.interp(I_drive[ii],I_drive_vec__imported,master_rate_matrix__imported[:,ind2])            
            # else:
            #     _rate = 0
            
            # calculate flux drive from synapses
            syn_flux = 0
            for de__sy_name in n.dendrites[de_name].input_synaptic_connections:
                if n.synapses[de__sy_name].inhibitory_or_excitatory == 'inhibitory':
                    _prefactor = -1
                elif n.synapses[de__sy_name].inhibitory_or_excitatory == 'excitatory':
                    _prefactor = 1
                syn_flux += _prefactor*n.synapses[de__sy_name].M*n.synapses[de__sy_name].I_si_vec[ii+1]
                
            # calculate flux drive from other dendrites
            dend_flux = 0
            for de__de_name in n.dendrites[de_name].input_dendritic_connections:
                # if ii == 0:
                # print('de__de_name = {}'.format(de__de_name))
                if n.dendrites[de__de_name].inhibitory_or_excitatory == 'inhibitory':
                    _prefactor = -1
                elif n.dendrites[de__de_name].inhibitory_or_excitatory == 'excitatory':
                    _prefactor = 1  
                # print('_prefactor = {}'.format(_prefactor))
                # print('n.dendrites[de__de_name].M = {}'.format(n.dendrites[de__de_name].M))
                # print('_prefactor = {}'.format(_prefactor))
                dend_flux += _prefactor*n.dendrites[de__de_name].M*n.dendrites[de__de_name].I_di_vec[ii] 
            
            # total drive to this dendrite
            n.dendrites[de_name].influx_vec[ii] = syn_flux+dend_flux
                              
            # if n.dendrites[de_name].I_drive_vec[ii] > 18.6:
            if n.dendrites[de_name].influx_vec[ii] > 18.6*current_to_flux:
                ind1 = (np.abs(influx_list__dend-n.dendrites[de_name].influx_vec[ii])).argmin()
                ind2 = (np.abs(I_di_array[ind1]-n.dendrites[de_name].I_di_vec[ii])).argmin()
                rate = rate_array__dend[ind1][ind2]
                # linear interpolation
                # rate = np.interp(I_drive[ii],I_drive_vec__imported,master_rate_matrix__imported[:,ind2])            
            else:
                rate = 0
    
            n.dendrites[de_name].I_di_vec[ii+1] = rate*n.dendrites[de_name].I_fq*dt + (1-dt/n.dendrites[de_name].tau_di)*n.dendrites[de_name].I_di_vec[ii]  
        
        # the neuron itself        
        # I_syn = 0
        # for sy_name in n.input_synaptic_connections:
        #     if n.synapses[sy_name].inhibitory_or_excitatory == 'inhibitory':
        #         _prefactor = -1
        #     elif n.synapses[sy_name].inhibitory_or_excitatory == 'excitatory':
        #         _prefactor = 1
        #     I_syn += _prefactor*n.synapses[sy_name].I_si_vec[ii+1]            
        # I_dend = 0
        # for de_name in n.input_dendritic_connections:
        #     if n.dendrites[de_name].inhibitory_or_excitatory == 'inhibitory':
        #         _prefactor = -1
        #     elif n.dendrites[de_name].inhibitory_or_excitatory == 'excitatory':
        #         _prefactor = 1                    
        #     I_dend += _prefactor*n.dendrites[de_name].I_di_vec[ii+1]
            
        # total drive current to neuron
        # n.I_drive_vec[ii] = I_syn+I_dend
        # print('n.I_drive_vec[ii] = {}'.format(n.I_drive_vec[ii]))
        # print('n.I_ni_vec[ii+1] = {}'.format(n.I_ni_vec[ii+1]))
        # print('np.max(np.asarray(I_drive_vec__imported__dend)) = {}'.format(np.max(np.asarray(I_drive_vec__imported__dend))))
                
        syn_flux = 0
        for sy_name in n.input_synaptic_connections:
            if n.synapses[sy_name].inhibitory_or_excitatory == 'inhibitory':
                _prefactor = -1
            elif n.synapses[sy_name].inhibitory_or_excitatory == 'excitatory':
                _prefactor = 1
            syn_flux += _prefactor*n.synapses[sy_name].M*n.synapses[sy_name].I_si_vec[ii+1]            
        dend_flux = 0
        for de_name in n.input_dendritic_connections:
            if n.dendrites[de_name].inhibitory_or_excitatory == 'inhibitory':
                _prefactor = -1
            elif n.dendrites[de_name].inhibitory_or_excitatory == 'excitatory':
                _prefactor = 1                    
            dend_flux += _prefactor*n.dendrites[de_name].M*n.dendrites[de_name].I_di_vec[ii+1]
        
        # total flux drive to neuron        
        n.influx_vec[ii] = syn_flux+dend_flux
        
        # if n.I_drive_vec[ii] > 18.6:
        if n.influx_vec[ii] > 18.6*current_to_flux:
            # _ind1 = (np.abs(I_drive_list__dend-n.I_drive_vec[ii])).argmin()
            ind1 = (np.abs(influx_list__dend-n.influx_vec[ii])).argmin()
            ind2 = (np.abs(I_di_array[ind1]-n.I_ni_vec[ii])).argmin()
            # print('ind1 = {}; ind2 = {}; size(master_rate_matrix__imported__dend) = {}'.format(ind1,ind2,master_rate_matrix__imported__dend.shape))
            rate = rate_array__dend[ind1][ind2]            
            if n.state == 'quiescent':
                n.spike_times.append(_pt)
            n.state = 'excited'
            # linear interpolation
            # rate = np.interp(I_drive[ii],I_drive_vec__imported,master_rate_matrix__imported[:,ind2]) 
            # print('_ind1 = {}; ind1 = {}'.format(_ind1,ind1))           
        else:
            n.state = 'quiescent'
            rate = 0

        n.I_ni_vec[ii+1] = rate*n.I_fq*dt + (1-dt/n.tau_ni)*n.I_ni_vec[ii]
        n.dendrites['{}__d'.format(n.name)].I_di_vec[ii+1] = n.I_ni_vec[ii+1]

    print('\ndone running neuron simulation. total time was {:.3}s\n'.format(time.time()-t_init))        
    
    return n #I_di_vec


def synapse_time_stepper__2jj__ode(time_vec,spike_times,L_list,r_list,I_bias_list):
    
    L_spd = L_list[0]
    L_jtl = L_list[1]
    L_si = L_list[2]
        
    r_spd2 = r_list[0]
    r_si = r_list[1]
    
    I_spd = I_bias_list[0]
    I_sy = I_bias_list[1] 
    I_sc = I_bias_list[2]
    
    Ic = 40e-6
    I_reset = 38.80656547520097e-6
    
    # print('L_spd = {}'.format(L_spd))
    # print('L_jtl = {}'.format(L_jtl))
    # print('L_si = {}'.format(L_si))
    # print('r_spd2 = {}'.format(r_spd2))
    # print('r_si = {}'.format(r_si))
    # print('I_spd = {}'.format(I_spd))  
    # print('I_sy = {}'.format(I_sy)) 
    # print('I_sc = {}'.format(I_sc)) 
    # print('I_fq = {}'.format(I_fq))
    # pause(2)
    
    nt = len(time_vec)
    dt = time_vec[1]-time_vec[0]
            
    I_sf_vec = np.zeros([nt])
    I_sf_vec[0] = I_sy # initial condition
    V_sf_vec = np.zeros([nt])
    r_spd1_vec = np.zeros([nt])
    
    I_si1_vec = np.zeros([nt])
    I_si1_vec[0] = I_sc # initial condition
    I_si2_vec = np.zeros([nt])
    V_si_vec = np.zeros([nt])
    
    j_sf_state = ['below_Ic']
    j_si_state = ['below_Ic']
    print('starting time stepping ...')
    for ii in range(nt-1):
                   
        _pt = time_vec[ii] # present time
                
        # find most recent spike time  
        st_ind = (np.abs(spike_times[:] - _pt)).argmin()
        # print('\nii = {}'.format(ii))
        # print('_pt = {}'.format(_pt))
        # print('st_ind = {}'.format(st_ind))
        # print('spike_times[st_ind] = {}'.format(spike_times[st_ind]))
                
        if ii > 0:
                        
            if st_ind == 0 and spike_times[st_ind] > _pt: # first spike has not arrived
                j_sf_state.append('below_Ic')
                j_si_state.append('below_Ic')
                # print('case a1')
            if st_ind > 0 and spike_times[st_ind] > _pt: # go back to previous spike
                st_ind -= 1
                # print('case a2')
            if spike_times[st_ind] <= _pt: # the case that counts                  
                # print('case a3')
                
                # calculate r_spd1
                r_spd1_vec[ii+1] = r_spd1_form(_pt - spike_times[st_ind])
                
                #J_si
                I_si1_tot = I_si1_vec[ii] - I_si2_vec[ii]
                if I_si1_tot > Ic:
                    j_si_state.append('above_Ic')    
                    # print('case b1')
                if I_si1_tot <= Ic and I_si1_tot > I_reset:
                    # print('case b2')
                    if j_si_state[ii-1] == 'above_Ic' or j_si_state[ii-1] == 'latched':
                        j_si_state.append('latched')
                        # print('case b3')
                    else:
                        j_si_state.append('below_Ic')
                        # print('case b4')
                if I_si1_tot <= I_reset:
                    j_si_state.append('below_Ic')
                    # print('case b5')
                
                # calculate V_si
                if j_si_state[ii] == 'above_Ic' or j_si_state[ii] == 'latched':
                    V_si_vec[ii+1] = Vj_of_Ij(I_si1_tot)                    
                elif j_si_state[ii] == 'below_Ic':
                    V_si_vec[ii+1] = 0                       
                
                #J_sf
                I_sf_tot = I_sf_vec[ii]
                if I_sf_tot > Ic:
                    j_sf_state.append('above_Ic')    
                    # print('case b1')
                if I_sf_tot <= Ic and I_sf_tot > I_reset:
                    # print('case b2')
                    if j_sf_state[ii-1] == 'above_Ic' or j_sf_state[ii-1] == 'latched':
                        j_sf_state.append('latched')
                        # print('case b3')
                    else:
                        j_sf_state.append('below_Ic')
                        # print('case b4')
                if I_sf_tot <= I_reset:
                    j_sf_state.append('below_Ic')
                    # print('case b5')
                
                # V_sf                
                if j_sf_state[ii] == 'above_Ic' or j_sf_state[ii] == 'latched':
                    V_sf_vec[ii+1] = Vj_of_Ij(I_sf_tot)                    
                elif j_sf_state[ii] == 'below_Ic':
                    V_sf_vec[ii+1] = 0 
         
        #update I_si1
        I_si1_vec[ii+1] = I_si1_vec[ii] + (dt/L_jtl)*( V_sf_vec[ii+1]-V_si_vec[ii+1] )
                               
        # update I_si2
        I_si2_vec[ii+1] = (dt/L_si)*V_si_vec[ii+1] + (1-dt*r_si/L_si)*I_si2_vec[ii]
        
        # update I_sf
        I_sf_vec[ii+1] = ( I_sf_vec[ii] 
                          + dt*(r_spd1_vec[ii+1]/L_spd)*(I_spd+I_sc+I_sy-I_sf_vec[ii]-I_si1_vec[ii+1]) 
                          - dt*(r_spd2/L_spd)*(I_sf_vec[ii]+I_si1_vec[ii+1]-I_sy-I_sc)
                          + (dt/L_jtl)*V_si_vec[ii+1]
                          - dt*(1/L_jtl+1/L_spd)*V_sf_vec[ii+1] )
        # I_sf_vec[ii+1] = ( I_sf_vec[ii] 
        #                   + dt*(r_spd1_vec[ii+1]/L_spd)*(I_spd+I_sc+I_sy-I_sf_vec[ii]-I_si1_vec[ii+1]-I_si2_vec[ii+1]) 
        #                   - dt*(r_spd2/L_spd)*(I_sf_vec[ii]+I_si1_vec[ii+1]+I_si2_vec[ii+1]-I_sy-I_sc)
        #                   + (dt/L_jtl)*V_si_vec[ii+1]
        #                   - dt*(1/L_jtl+1/L_spd)*V_sf_vec[ii+1] )
        
    print('done time stepping')
    return I_si1_vec, I_si2_vec, I_sf_vec


def synapse_time_stepper__Isf_ode__spd_jj_test(time_vec,spike_times,L_list,r_list,I_bias_list):
    
    L_spd = L_list[0]
        
    r_spd2 = r_list[0]
    
    I_spd = I_bias_list[0]
    I_sy = I_bias_list[1] 
    
    # Ic = 40
    Ic = 40e-6
    I_reset = 38.80656547520097e-6 # 38.81773424470013e-6
    
    # print('L_spd = {}'.format(L_spd))
    # print('r_spd2 = {}'.format(r_spd2))
    # print('I_spd = {}'.format(I_spd))  
    # print('I_sy = {}'.format(I_sy))
    # pause(2)
    
    nt = len(time_vec)
    dt = time_vec[1]-time_vec[0]
        
    I_sf_vec = np.zeros([nt]) 
    I_sf_vec[0] = I_sy # initial condition
    V_sf_vec = np.zeros([nt])
    r_spd1_vec = np.zeros([nt])

    j_sf_state = ['below_Ic']
    print('starting time stepping ...')
    for ii in range(nt-1):
            
        _pt = time_vec[ii] # present time
        # print('_pt = {}'.format(_pt))
                
        # find most recent spike time  
        st_ind = (np.abs(spike_times[:] - _pt)).argmin()
        
        if ii > 0:
                        
            if st_ind == 0 and spike_times[st_ind] > _pt: # first spike has not arrived
                j_sf_state.append('below_Ic')
            if st_ind > 0 and spike_times[st_ind] > _pt: # go back to previous spike
                st_ind -= 1
            if spike_times[st_ind] <= _pt: # the case that counts                  
                
                if I_sf_vec[ii] > Ic:
                    j_sf_state.append('above_Ic')                    
                if I_sf_vec[ii] <= Ic and I_sf_vec[ii] > I_reset:
                    if j_sf_state[ii-1] == 'above_Ic' or j_sf_state[ii-1] == 'latched':
                        j_sf_state.append('latched')
                    else:
                        j_sf_state.append('below_Ic')
                if I_sf_vec[ii] <= I_reset:
                    j_sf_state.append('below_Ic')
                
                # step forward Isf
                r_spd1_vec[ii+1] = r_spd1_form(_pt - spike_times[st_ind])
                if j_sf_state[ii] == 'above_Ic' or j_sf_state[ii] == 'latched':
                    V_sf_vec[ii+1] = Vj_of_Ij(I_sf_vec[ii])
                elif j_sf_state[ii] == 'below_Ic':
                    V_sf_vec[ii+1] = 0
                
        I_sf_vec[ii+1] = I_sf_vec[ii] + dt*(r_spd1_vec[ii+1]/L_spd)*(I_spd+I_sy-I_sf_vec[ii]) - dt*(r_spd2/L_spd)*(I_sf_vec[ii]-I_sy) - dt*V_sf_vec[ii+1]/L_spd
                
    print('done time stepping')
    return I_sf_vec, V_sf_vec, r_spd1_vec, j_sf_state


def r_spd1_form(t):
    
    _dt = 100e-12 # units of s
    _Dt = 200e-12 # units of s
    _r_spd1_max = 5e3
    
    r_spd1 = 0
    if t <= _dt:
        r_spd1 = (_r_spd1_max/_dt)*t
    if _dt < t and t < _dt+_Dt:
        r_spd1 = _r_spd1_max
    if _dt+_Dt <= t and t <= 2*_dt+_Dt:
        r_spd1 = _r_spd1_max-(_r_spd1_max/_dt)*(t-(_dt+_Dt))
    
    return r_spd1


# def Vsf_of_Isf(Isf):
    
#     # print('Isf = {}'.format(Isf))
#     # return 235.19243476368464*( (Isf/38.81773424470013e-6)**3.4193613971219454 - 1 )**0.3083945546392435 # unit of uV
#     return 1e-6*236.878860808991*( (Isf/38.80656547520097e-6)**3.3589340685815574 - 1 )**0.310721713450461 # unit of V


def Vj_of_Ij(Ij):
    
    # print('Isf = {}'.format(Isf))
    # return 235.19243476368464*( (Isf/38.81773424470013e-6)**3.4193613971219454 - 1 )**0.3083945546392435 # unit of uV
    return 1e-6*236.878860808991*( (Ij/38.80656547520097e-6)**3.3589340685815574 - 1 )**0.310721713450461 # unit of V


# def synapse_time_stepper__1jj_ode(time_vec,spike_times,L_list,r_list,I_bias_list):
    
#     L_spd = L_list[0]
#     L_si = L_list[1]
        
#     r_spd2 = r_list[0]
#     r_si = r_list[1]
    
#     I_spd = I_bias_list[0]
#     I_sy = I_bias_list[1] 
    
#     Ic = 40e-6
#     I_reset = 38.80656547520097e-6
    
#     # print('L_spd = {}'.format(L_spd))
#     # print('L_si = {}'.format(L_si))
#     # print('r_spd2 = {}'.format(r_spd2))
#     # print('r_si = {}'.format(r_si))
#     # print('I_spd = {}'.format(I_spd))  
#     # print('I_sy = {}'.format(I_sy)) 
#     # print('I_fq = {}'.format(I_fq))
#     # pause(2)
    
#     nt = len(time_vec)
#     dt = time_vec[1]-time_vec[0]
        
#     I_si_vec = np.zeros([nt])
#     I_sf_vec = np.zeros([nt])
#     I_sf_vec[0] = I_sy # initial condition
#     V_sf_vec = np.zeros([nt])
#     r_spd1_vec = np.zeros([nt])
    
#     j_sf_state = ['below_Ic']
#     print('starting time stepping ...')
#     for ii in range(nt-1):
                   
#         _pt = time_vec[ii] # present time
                
#         # find most recent spike time  
#         st_ind = (np.abs(spike_times[:] - _pt)).argmin()
#         # print('\nii = {}'.format(ii))
#         # print('_pt = {}'.format(_pt))
#         # print('st_ind = {}'.format(st_ind))
#         # print('spike_times[st_ind] = {}'.format(spike_times[st_ind]))
                
#         if ii > 0:
                        
#             if st_ind == 0 and spike_times[st_ind] > _pt: # first spike has not arrived
#                 j_sf_state.append('below_Ic')
#                 # print('case a1')
#             if st_ind > 0 and spike_times[st_ind] > _pt: # go back to previous spike
#                 st_ind -= 1
#                 # print('case a2')
#             if spike_times[st_ind] <= _pt: # the case that counts                  
#                 # print('case a3')
                
#                 I_sf_tot = I_sf_vec[ii] - I_si_vec[ii]
#                 if I_sf_tot > Ic:
#                     j_sf_state.append('above_Ic')    
#                     # print('case b1')
#                 if I_sf_tot <= Ic and I_sf_tot > I_reset:
#                     # print('case b2')
#                     if j_sf_state[ii-1] == 'above_Ic' or j_sf_state[ii-1] == 'latched':
#                         j_sf_state.append('latched')
#                         # print('case b3')
#                     else:
#                         j_sf_state.append('below_Ic')
#                         # print('case b4')
#                 if I_sf_tot <= I_reset:
#                     j_sf_state.append('below_Ic')
#                     # print('case b5')
                
#                 # calculate r_spd1, V_sf
#                 r_spd1_vec[ii+1] = r_spd1_form(_pt - spike_times[st_ind])
                
#                 if j_sf_state[ii] == 'above_Ic' or j_sf_state[ii] == 'latched':
#                     V_sf_vec[ii+1] = Vj_of_Ij(I_sf_tot)                    
#                 elif j_sf_state[ii] == 'below_Ic':
#                     V_sf_vec[ii+1] = 0                       
           
#                 # update I_sf and I_si
#                 L_j = Ljj(Ic,I_sf_tot)
#                 L_tilde_sq = L_j*L_spd+L_si*L_spd+L_j*L_si+L_si*L_spd
#                 L_p = L_tilde_sq/(L_j+L_si)
                
#                 I_sf_vec[ii+1] = ( I_sf_vec[ii] 
#                                   + dt*r_spd1_vec[ii+1]*(I_spd+I_sy-I_si_vec[ii]-I_sf_vec[ii])/L_p 
#                                   - dt*r_spd2*(I_sf_vec[ii]+I_si_vec[ii]-I_sy)/L_p
#                                   + dt*r_si*I_si_vec[ii]*(L_spd-L_j)/L_tilde_sq
#                                   - dt*(L_spd+L_si)*V_sf_vec[ii+1]/L_tilde_sq )
            
#                 I_si_vec[ii+1] = ( I_si_vec[ii]
#                                   + dt*V_sf_vec[ii+1]/(L_j+L_si)
#                                   + (L_j/(L_j+L_si))*(I_sf_vec[ii+1]-I_sf_vec[ii])
#                                   - (dt*r_si/(L_j+L_si))*I_si_vec[ii] )
                
#     print('done time stepping')
#     return I_si_vec, I_sf_vec, j_sf_state

def synapse_time_stepper__1jj_ode(time_vec,spike_times,L_list,r_list,I_bias_list):
    
    L_spd = L_list[0]
    L_si = L_list[1]
        
    r_spd2 = r_list[0]
    r_si = r_list[1]
    
    I_spd = I_bias_list[0]
    I_sy = I_bias_list[1] 
    
    Ic = 40e-6
    I_reset = 38.80656547520097e-6
    
    # print('L_spd = {}'.format(L_spd))
    # print('L_si = {}'.format(L_si))
    # print('r_spd2 = {}'.format(r_spd2))
    # print('r_si = {}'.format(r_si))
    # print('I_spd = {}'.format(I_spd))  
    # print('I_sy = {}'.format(I_sy)) 
    # print('I_fq = {}'.format(I_fq))
    # pause(2)
    
    nt = len(time_vec)
    dt = time_vec[1]-time_vec[0]
        
    I_si_vec = np.zeros([nt])
    I_sf_vec = np.zeros([nt])
    I_sf_vec[0] = I_sy # initial condition
    V_sf_vec = np.zeros([nt])
    r_spd1_vec = np.zeros([nt])
    
    j_sf_state = ['below_Ic']
    print('starting time stepping ...')
    for ii in range(nt-1):
                   
        _pt = time_vec[ii] # present time
                
        # find most recent spike time  
        st_ind = (np.abs(spike_times[:] - _pt)).argmin()
        # print('\nii = {}'.format(ii))
        # print('_pt = {}'.format(_pt))
        # print('st_ind = {}'.format(st_ind))
        # print('spike_times[st_ind] = {}'.format(spike_times[st_ind]))
                
        if ii > 0:
                        
            if st_ind == 0 and spike_times[st_ind] > _pt: # first spike has not arrived
                j_sf_state.append('below_Ic')
                # print('case a1')
            if st_ind > 0 and spike_times[st_ind] > _pt: # go back to previous spike
                st_ind -= 1
                # print('case a2')
            if spike_times[st_ind] <= _pt: # the case that counts                  
                # print('case a3')
                
                I_sf_tot = I_sf_vec[ii] - I_si_vec[ii]
                if I_sf_tot > Ic:
                    j_sf_state.append('above_Ic')    
                    # print('case b1')
                if I_sf_tot <= Ic and I_sf_tot > I_reset:
                    # print('case b2')
                    if j_sf_state[ii-1] == 'above_Ic' or j_sf_state[ii-1] == 'latched':
                        j_sf_state.append('latched')
                        # print('case b3')
                    else:
                        j_sf_state.append('below_Ic')
                        # print('case b4')
                if I_sf_tot <= I_reset:
                    j_sf_state.append('below_Ic')
                    # print('case b5')
                
                # calculate r_spd1, V_sf
                r_spd1_vec[ii+1] = r_spd1_form(_pt - spike_times[st_ind])
                
                if j_sf_state[ii] == 'above_Ic' or j_sf_state[ii] == 'latched':
                    V_sf_vec[ii+1] = Vj_of_Ij(I_sf_tot)                    
                elif j_sf_state[ii] == 'below_Ic':
                    V_sf_vec[ii+1] = 0                       
           
        # update I_sf and I_si
        # I_sy = I_sy_0 - I_si_vec[ii]
        I_sf_vec[ii+1] = ( I_sf_vec[ii] 
                          + dt*(r_spd1_vec[ii+1]/L_spd)*(I_spd+I_sy-I_sf_vec[ii]) 
                          - dt*(r_spd2/L_spd)*(I_sf_vec[ii]-I_sy)
                          - dt*V_sf_vec[ii+1]/L_spd )
        # I_sf_vec[ii+1] = ( I_sf_vec[ii] 
        #                   + dt*(r_spd1_vec[ii+1]/L_spd)*(I_spd+I_sy-I_sf_vec[ii]-I_si_vec[ii]) 
        #                   - dt*(r_spd2/L_spd)*(I_sf_vec[ii]+I_si_vec[ii]-I_sy)
        #                   -dt*(r_si/L_si)*I_si_vec[ii]
        #                   -dt*V_sf_vec[ii+1]*(1/L_spd+1/L_si) )
        # I_sf_vec[ii+1] = ( I_sf_vec[ii] 
        #                   + dt*(r_spd1_vec[ii+1]/L_spd)*(I_spd-I_sf_vec[ii]+I_si_vec[ii]+I_sy) 
        #                   - dt*(r_spd2/L_spd)*(I_sf_vec[ii]-I_si_vec[ii]-I_sy)
        #                   -dt*(r_si/L_si)*I_si_vec[ii]
        #                   +dt*V_sf_vec[ii+1]*(1/L_si-1/L_spd) )
        
        I_si_vec[ii+1] = dt*V_sf_vec[ii+1]/L_si + (1-dt*r_si/L_si)*I_si_vec[ii]
            # print('ii = {}; I_si_vec[ii+1] = {}'.format(ii,I_si_flux_vec[ii+1]))
        
        # print('j_sf_state = {}'.format(j_sf_state[ii]))
        
    print('done time stepping')
    return I_si_vec, I_sf_vec, j_sf_state


def synapse_time_stepper__Isf_ode__spd_delta(time_vec,spike_times,L_list,r_list,I_bias_list):
    
    L_spd = L_list[0]
    L_si = L_list[1]
        
    r_spd2 = r_list[0]
    r_si = r_list[1]
    
    I_spd = I_bias_list[0]
    I_sy = I_bias_list[1] 
    
    Ic = 40e-6
    I_reset = 38.80656547520097e-6
    
    # print('L_spd = {}'.format(L_spd))
    # print('L_si = {}'.format(L_si))
    # print('r_spd2 = {}'.format(r_spd2))
    # print('r_si = {}'.format(r_si))
    # print('I_spd = {}'.format(I_spd))  
    # print('I_sy = {}'.format(I_sy)) 
    # print('I_fq = {}'.format(I_fq))
    # pause(2)
    
    nt = len(time_vec)
    dt = time_vec[1]-time_vec[0]
        
    I_si_vec = np.zeros([nt])
    I_sf_vec = np.zeros([nt])
    I_sf_vec[0] = I_sy # initial condition
    V_sf_vec = np.zeros([nt])
    r_spd1_vec = np.zeros([nt])
    
    j_sf_state = ['below_Ic']
    print('starting time stepping ...')    
    for ii in range(nt-1):
                   
        _pt = time_vec[ii] # present time
                
        # find most recent spike time  
        st_ind = (np.abs(spike_times[:] - _pt)).argmin()
        # print('\nii = {}'.format(ii))
        # print('_pt = {}'.format(_pt))
        # print('st_ind = {}'.format(st_ind))
        # print('spike_times[st_ind] = {}'.format(spike_times[st_ind]))
                
        if ii > 0:
                        
            if st_ind == 0 and spike_times[st_ind] > _pt: # first spike has not arrived
                j_sf_state.append('below_Ic')
                # print('case a1')
            if st_ind > 0 and spike_times[st_ind] > _pt: # go back to previous spike
                st_ind -= 1
                # print('case a2')
            if spike_times[st_ind] <= _pt: # the case that counts                  
                # print('case a3')
                
                # update I_sf
                if _pt - spike_times[st_ind] < 1.5*dt:
                    I_sf_vec[ii+1] = I_spd+I_sy
                else:   
                    I_sf_vec[ii+1] = ( I_sf_vec[ii] 
                                      + dt*(r_spd1_vec[ii+1]/L_spd)*(I_spd+I_sy-I_sf_vec[ii]) 
                                      - dt*(r_spd2/L_spd)*(I_sf_vec[ii]-I_sy)
                                      -dt*V_sf_vec[ii+1]/L_spd )
        
                # update V_sf
                I_sf_tot = I_sf_vec[ii+1] - I_si_vec[ii]
                if I_sf_tot > Ic:
                    j_sf_state.append('above_Ic')    
                    # print('case b1')
                if I_sf_tot <= Ic and I_sf_tot > I_reset:
                    # print('case b2')
                    if j_sf_state[ii-1] == 'above_Ic' or j_sf_state[ii-1] == 'latched':
                        j_sf_state.append('latched')
                        # print('case b3')
                    else:
                        j_sf_state.append('below_Ic')
                        # print('case b4')
                if I_sf_tot <= I_reset:
                    j_sf_state.append('below_Ic')
                    # print('case b5')                    
                
                if j_sf_state[ii] == 'above_Ic' or j_sf_state[ii] == 'latched':
                    V_sf_vec[ii+1] = Vj_of_Ij(I_sf_tot)                    
                elif j_sf_state[ii] == 'below_Ic':
                    V_sf_vec[ii+1] = 0                               
        
        # update I_si
        I_si_vec[ii+1] = dt*V_sf_vec[ii+1]/L_si + (1-dt*r_si/L_si)*I_si_vec[ii]
        
    print('done time stepping')
    return I_si_vec, I_sf_vec


def synapse_time_stepper(time_vec,spike_times,num_jjs,L_list,I_bias_list,tau_si): 
    
    # inductances in L_list = [L_jtl1,L_jtl2,L_si] with units of picohenries 
    
    I_bias_sy = I_bias_list[0]
    if num_jjs == 1:
        L_si = L_list[0]
    if num_jjs == 2:
        I_bias_si = I_bias_list[1]
        L_jtl = L_list[0]
        L_si = L_list[1] 
    if num_jjs == 3:
        I_bias_jtl = I_bias_list[1]
        I_bias_si = I_bias_list[2]
        L_jtl1 = L_list[0]
        L_jtl2 = L_list[1]
        L_si = L_list[2]    
    
    Ic = 40
    I_sf_hyst = 1.1768
    I_reset = Ic - I_sf_hyst
    
    st_ind_last = 0
    spd_current_memory = 0
        
    p = physical_constants()
    Phi0 = p['Phi0']
    I_fq = 1e6*Phi0/(L_si*1e-12)
    
    # print('tau_si = {}'.format(tau_si))
    # print('L_jtl1 = {}'.format(L_jtl1))
    # print('L_jtl2 = {}'.format(L_jtl1))
    # print('L_si = {}'.format(L_si))
    # print('I_sy = {}'.format(I_bias_sy))  
    # print('I_si = {}'.format(I_bias_si)) 
    # print('I_fq = {}'.format(I_fq))
    # pause(2)
    
    if len(spike_times) > 0:
        
        # print('spike_times = {}'.format(spike_times))
        
        nt = len(time_vec)
        dt = time_vec[1]-time_vec[0]
        
        # print('loading spd response data')
        if num_jjs == 1:
            file_string__spd = 'master__syn__spd_response__1jj__dt{:04.0f}ps.soen'.format(dt*1e6)
            file_string__rate_array = 'master__syn__rate_array__1jj__Isipad0010nA.soen'
        elif num_jjs == 2:
            file_string__spd = 'master__syn__spd_response__2jj__dt{:04.0f}ps.soen'.format(dt*1e6)
            file_string__rate_array = 'master__syn__rate_array__2jj__Isipad0010nA.soen'
        elif num_jjs == 3:
            file_string__spd = 'master__syn__spd_response__3jj__dt{:04.0f}ps.soen'.format(dt*1e6)
            file_string__rate_array = 'master__syn__rate_array__3jj__Isipad0010nA.soen'

        with open('../_circuit_data/{}'.format(file_string__spd), 'rb') as data_file:         
            data_array__spd = pickle.load(data_file)
            
        spd_response_array = data_array__spd['spd_response_array'] # entries have units of uA
        I_sy_list__spd = data_array__spd['I_sy_list'] # entries have units of uA
        # print('I_sy_list__spd = {}'.format(I_sy_list__spd))
        spd_t = data_array__spd['time_vec'] # entries have units of us
        spd_duration = spd_t[-1]
        
        # print('loading rate array')
        with open('../_circuit_data/{}'.format(file_string__rate_array), 'rb') as data_file:         
            data_array__rate = pickle.load(data_file)                        
            
        I_si_array = data_array__rate['I_si_array'] # entries have units of uA
        I_drive_list = data_array__rate['I_drive_list'] # entries have units of uA
        rate_array = data_array__rate['rate_array'] # entries have units of fluxons per microsecond                  
        
        
        I_si_vec = np.zeros([nt])
        I_spd_vec = np.zeros([nt])
        I_sf_vec = np.zeros([nt])
        
        j_sf_state = ['below_Ic']
        print('starting time stepping ...')
        
        if num_jjs == 1:
            
            I_sf = I_bias_sy
            for ii in range(nt-1):
                # dt = time_vec[ii+1]-time_vec[ii]
                
                _pt = time_vec[ii] # present time
                # print('ii = {:d}, present time = {:6.4f}us, j_sf_state = {}'.format(ii,_pt,j_sf_state[ii]))
                
                # find most recent spike time  
                # gf = 0
                st_ind = (np.abs(spike_times[:] - _pt)).argmin()
                
                if ii > 0:
                    
                    I_sf = I_bias_sy+I_spd_vec[ii-1]-I_si_vec[ii-1]
                    I_sf_vec[ii] = I_sf
                        
                    if st_ind == 0 and spike_times[st_ind] > _pt: # first spike has not arrived
                        gf = 0 # growth factor
                        j_sf_state.append('below_Ic')
                        # print('code 1: st_ind == 0 and spike_times[st_ind] > _pt')
                    if st_ind > 0 and spike_times[st_ind] > _pt: # go back to previous spike
                        st_ind -= 1
                        # print('code 2: st_ind > 0 and spike_times[st_ind] > _pt')
                    if _pt - spike_times[st_ind] > spd_duration: # outside SPD pulse range
                        gf = 0 # growth factor
                        j_sf_state.append('below_Ic')
                        # print('code 3: _pt - spike_times[st_ind] > spd_duration')
                    if spike_times[st_ind] <= _pt and _pt - spike_times[st_ind] < spd_duration: # the case that counts
                        # print('code 4')                    
                        
                        if I_sf > Ic:
                            j_sf_state.append('above_Ic')
                            bias_lower = I_si_vec[ii-1] # np.max([I_si_vec[ii-1]-I_sf_hyst,0])
                            
                        if I_sf <= Ic and I_sf >= I_reset:
                            if j_sf_state[ii-1] == 'above_Ic' or j_sf_state[ii-1] == 'latched':
                                j_sf_state.append('latched')
                                bias_lower = I_si_vec[ii-1] # np.max([I_si_vec[ii-1]-I_sf_hyst,0])
                            else:
                                j_sf_state.append('below_Ic')
                                bias_lower = I_si_vec[ii-1] # np.max([I_si_vec[ii-1]-I_sf_hyst,0])
                                
                        if I_sf < I_reset:
                            j_sf_state.append('below_Ic')
                            # if I_si_vec[ii-1] > 
                            bias_lower = I_si_vec[ii-1] # np.max([I_si_vec[ii-1]-I_sf_hyst,0]) # 0 # 
                            
                else:
                    bias_lower = 0
                        
                #this block to avoid spd drive going too low at the onset of each spike
                I_sy_ind_spd = (np.abs(I_sy_list__spd[:] - (I_bias_sy-bias_lower) )).argmin()
                spd_i = spd_response_array[I_sy_ind_spd]            
                dt_spk = _pt - spike_times[st_ind]                
                spd_t_ind  = (np.abs(spd_t[:] - dt_spk)).argmin()
                if st_ind - st_ind_last == 1:
                    spd_current = np.max([I_spd_vec[ii-1],spd_i[spd_t_ind]])
                    spd_current_memory = spd_current
                if spd_current_memory > 0 and spd_i[spd_t_ind] < spd_current_memory:
                    spd_current = spd_current_memory
                else:
                    spd_current = spd_i[spd_t_ind]
                    spd_current_memory = 0
                I_spd_vec[ii] = spd_current
                
                st_ind_last = st_ind
                
                I_drive = spd_current+I_bias_sy-Ic # -Ic in there because all data so far is with 40uA JJs and this is how I_drive is defined when making rate array
                # # I_sf = spd_current+I_sy-I_si_vec[ii-1]
                # I_drive = I_drive_real-I_c # -I_c in there because all data so far is with 40uA JJs and this is how I_drive is defined when making rate array
                
                if j_sf_state[ii] == 'latched' or j_sf_state[ii] == 'above_Ic':
                    
                    if I_drive < np.min(I_drive_list):
                        gf = 0
                    else:                    
                        
                        I_drive_ind = (np.abs(I_drive_list[:] - I_drive )).argmin() 
                        I_si_ind = (np.abs(np.asarray(I_si_array[I_drive_ind][:]) - I_si_vec[ii])).argmin()
                        r_sf = rate_array[I_drive_ind][I_si_ind]
                        # r_sf = syn_1jj_rate_vs_Isf(I_sf) 
                          
                        #no interpolation
                        gf = dt*I_fq*r_sf # growth factor
                        
                else:
                    gf = 0
                                                                    
                I_si_vec[ii+1] = gf + (1-dt/tau_si)*I_si_vec[ii]   
                              
        if num_jjs == 2:
            
            I_sy = I_bias_sy
            I_sy_ind_spd = (np.abs(I_sy_list__spd[:] - I_sy)).argmin()
            spd_i = spd_response_array[I_sy_ind_spd]
            for ii in range(nt-1):                
               
                # find most recent spike time
                _pt = time_vec[ii] # present time  
                st_ind = (np.abs(spike_times[:] - _pt)).argmin()
                gf = 0
                if st_ind == 0 and spike_times[st_ind] > _pt:
                    gf = 0 # growth factor
                    # print('code 1: st_ind == 0 and spike_times[st_ind] > _pt')
                if st_ind > 0 and spike_times[st_ind] > _pt:
                    st_ind -= 1
                    # print('code 2: st_ind > 0 and spike_times[st_ind] > _pt')
                if _pt - spike_times[st_ind] > spd_duration:
                    gf = 0 # growth factor
                    # print('code 3: _pt - spike_times[st_ind] > spd_duration')
                if spike_times[st_ind] <= _pt and _pt - spike_times[st_ind] < spd_duration:
                    # print('code 4')
                    
                    dt_spk = _pt - spike_times[st_ind]
                    spd_t_ind  = (np.abs(spd_t[:] - dt_spk)).argmin()
                    
                    # this block to avoid spd drive going too low at the onset of each spike 
                    if st_ind - st_ind_last == 1:
                        spd_current = np.max([I_spd_vec[ii-1],spd_i[spd_t_ind]])
                        spd_current_memory = spd_current
                    if spd_current_memory > 0 and spd_i[spd_t_ind] < spd_current_memory:
                        spd_current = spd_current_memory
                    else:
                        spd_current = spd_i[spd_t_ind]
                        spd_current_memory = 0
                    
                    # spd_current = spd_i[spd_t_ind]
                    I_spd_vec[ii] = spd_current
                    st_ind_last = st_ind
                    I_tot = spd_current+I_sy
                    I_drive = I_tot-Ic # all data so far is with 40uA JJs
                    if I_drive < np.min(I_drive_list):
                        gf = 0
                    else:                    
                        I_drive_ind = (np.abs(I_drive_list[:] - I_drive)).argmin()
                        I_si_ind = (np.abs(np.asarray(I_si_array[I_drive_ind][:]) - I_si_vec[ii])).argmin()                        
                        
                        #no interpolation
                        gf = dt*I_fq*rate_array[I_drive_ind][I_si_ind] # growth factor
                    
                        # linear interpolation
                        # gf = dt*I_fq*np.interp(spd_current,I_drive_list,rate_array[:][I_si_ind])
                                
                I_si_vec[ii+1] = gf + (1-dt/tau_si)*I_si_vec[ii]
            
            #below is the approach where you update the current in each jj at each time step assuming they're all below Ic (inductive division). this approach gives poor results
            # # make I_sf list
            # I_sf_list__spd = np.zeros([len(I_sy_list__spd)])
            # for qq in range(len(I_sy_list__spd)):
            #     Isf, Ijtl, Isi1, Isi2, L_jsf, L_jsi = synapse_current_distribution__2jj(Ic,L_jtl,L_si,[I_bias_sy,I_bias_si],I_bias_sy,0,I_bias_si,0)
            #     for pp in range(5):
            #         Isf, Ijtl, Isi1, Isi2, L_jsf, L_jsi = synapse_current_distribution__2jj(Ic,L_jtl,L_si,[I_bias_sy,I_bias_si],Isf,Ijtl,Isi1,Isi2)
            #     I_sf_list__spd[qq] = Isf

            # Isf, Ijtl, Isi1, Isi2, L_jsf, L_jsi = synapse_current_distribution__2jj(Ic,L_jtl,L_si,[I_bias_sy,I_bias_si],Isf,Ijtl,Isi1,Isi2)
            # I_bias_list_1_perm = I_bias_list[1]
            # for ii in range(nt-1):  
                
                # # find most recent spike time
                # _pt = time_vec[ii] # present time  
                # st_ind = (np.abs(spike_times[:] - _pt)).argmin()
                # gf = 0
                # if st_ind == 0 and spike_times[st_ind] > _pt:
                #     gf = 0 # growth factor
                #     # print('code 1: st_ind == 0 and spike_times[st_ind] > _pt')
                # if st_ind > 0 and spike_times[st_ind] > _pt:
                #     st_ind -= 1
                #     # print('code 2: st_ind > 0 and spike_times[st_ind] > _pt')
                # if _pt - spike_times[st_ind] > spd_duration:
                #     gf = 0 # growth factor
                #     # print('code 3: _pt - spike_times[st_ind] > spd_duration')
                # if spike_times[st_ind] <= _pt and _pt - spike_times[st_ind] < spd_duration:
                #     # print('code 4')
                    
                #     dt_spk = _pt - spike_times[st_ind]
                #     spd_t_ind  = (np.abs(spd_t[:] - dt_spk)).argmin()
                                    
                #     # update current distribution throughout circuit and JJ inductances
                #     I_bias_list[1] = I_bias_list_1_perm - I_si_vec[ii]                                
                #     Isf, Ijtl, Isi1, Isi2, L_jsf, L_jsi = synapse_current_distribution__2jj(Ic,L_jtl,L_si,[I_bias_sy,I_bias_si],Isf,Ijtl,Isi1,Isi2)
                #     # print('Isf = {}uA, Ijtl = {}uA, Isi1 = {}uA, Isi2 = {}uA'.format(Isf, Ijtl, Isi1, Isi2))
                #     I_sf_ind_spd = (np.abs( I_sf_list__spd[:] - Isf )).argmin()
                #     spd_i = spd_response_array[I_sf_ind_spd]
                    
                #     # I_sy_ind_spd = (np.abs( I_sy_list__spd[:] - I_bias_sy )).argmin()
                #     # spd_i = spd_response_array[I_sy_ind_spd]
                    
                #     # this block to avoid spd drive going too low at the onset of each spike 
                #     if st_ind - st_ind_last == 1:
                #         spd_current = np.max([I_spd_vec[ii-1],spd_i[spd_t_ind]])
                #         spd_current_memory = spd_current
                #     if spd_current_memory > 0 and spd_i[spd_t_ind] < spd_current_memory:
                #         spd_current = spd_current_memory
                #     else:
                #         spd_current = spd_i[spd_t_ind]
                #         spd_current_memory = 0
                        
                #     I_spd_vec[ii] = spd_current
                    
                #     st_ind_last = st_ind
                    
                #     I_drive = I_bias_sy+spd_current-Ic
                #     if I_drive < np.min(I_drive_list):
                #         gf = 0
                #     else:
                #         I_drive_ind = (np.abs(I_drive_list[:] - I_drive)).argmin()
                #         I_si_ind = (np.abs(I_si_array[I_drive_ind] - I_si_vec[ii])).argmin()
                #         gf = dt*I_fq*rate_array[I_drive_ind][I_si_ind] # growth factor
                #         # print('rate_array[I_drive_ind][I_si_ind] = {}'.format(rate_array[I_drive_ind][I_si_ind]) )
                #         # print('gf = {}uA/us'.format(gf))
                #         # gf = dt*I_fq*master_rate_matrix__imported[I_drive_ind,I_si_ind] 
                       
                #         # linear interpolation
                #         # rate = np.interp(spd_current,I_drive_vec__imported,master_rate_matrix__imported[:,I_si_ind])
                #         # gf = dt*I_fq*rate                                
               
                # I_si_vec[ii+1] = gf + (1-dt/tau_si)*I_si_vec[ii]
                
        if num_jjs == 3:            
            
            I_sy = I_bias_sy
            I_sy_ind_spd = (np.abs(I_sy_list__spd[:] - I_sy)).argmin()
            spd_i = spd_response_array[I_sy_ind_spd]
            for ii in range(nt-1):                
               
                # find most recent spike time
                _pt = time_vec[ii] # present time  
                st_ind = (np.abs(spike_times[:] - _pt)).argmin()
                gf = 0
                if st_ind == 0 and spike_times[st_ind] > _pt:
                    gf = 0 # growth factor
                    # print('code 1: st_ind == 0 and spike_times[st_ind] > _pt')
                if st_ind > 0 and spike_times[st_ind] > _pt:
                    st_ind -= 1
                    # print('code 2: st_ind > 0 and spike_times[st_ind] > _pt')
                if _pt - spike_times[st_ind] > spd_duration:
                    gf = 0 # growth factor
                    # print('code 3: _pt - spike_times[st_ind] > spd_duration')
                if spike_times[st_ind] <= _pt and _pt - spike_times[st_ind] < spd_duration:
                    # print('code 4')
                    
                    dt_spk = _pt - spike_times[st_ind]
                    spd_t_ind  = (np.abs(spd_t[:] - dt_spk)).argmin()
                    
                    # this block to avoid spd drive going too low at the onset of each spike 
                    if st_ind - st_ind_last == 1:
                        spd_current = np.max([I_spd_vec[ii-1],spd_i[spd_t_ind]])
                        spd_current_memory = spd_current
                    if spd_current_memory > 0 and spd_i[spd_t_ind] < spd_current_memory:
                        spd_current = spd_current_memory
                    else:
                        spd_current = spd_i[spd_t_ind]
                        spd_current_memory = 0
                    
                    # spd_current = spd_i[spd_t_ind]
                    I_spd_vec[ii] = spd_current
                    st_ind_last = st_ind
                    I_tot = spd_current+I_sy
                    I_drive = I_tot-Ic # all data so far is with 40uA JJs
                    if I_drive < np.min(I_drive_list):
                        gf = 0
                    else:                    
                        I_drive_ind = (np.abs(I_drive_list[:] - I_drive)).argmin()
                        I_si_ind = (np.abs(np.asarray(I_si_array[I_drive_ind][:]) - I_si_vec[ii])).argmin()                        
                        
                        #no interpolation
                        gf = dt*I_fq*rate_array[I_drive_ind][I_si_ind] # growth factor
                    
                        # linear interpolation
                        # gf = dt*I_fq*np.interp(spd_current,I_drive_list,rate_array[:][I_si_ind])
                                
                I_si_vec[ii+1] = gf + (1-dt/tau_si)*I_si_vec[ii]
            
                #below is the approach where you update the current in each jj at each time step assuming they're all below Ic (inductive division). this approach gives poor results
                # # make I_sf list
                # I_sf_list__spd = np.zeros([len(I_sy_list__spd)])
                # for qq in range(len(I_sy_list__spd)):
                #     I1, I2, I3, Ijsf, Ijtl, Ijsi, Lj1, Lj2, Lj3  = synapse_current_distribution__3jj(Ic,L_jtl1,L_jtl2,L_si,I_bias_list,I_bias_sy,I_bias_jtl,I_bias_si)
                #     for pp in range(5):
                #         I1, I2, I3, Ijsf, Ijtl, Ijsi, Ljsf, Ljtl, Ljsi = synapse_current_distribution__3jj(Ic,L_jtl1,L_jtl2,L_si,I_bias_list,Ijsf,Ijtl,Ijsi)
                #     I_sf_list__spd[qq] = Ijsf
                
                # # spd_duration = spd_t[-1]
                # # print('L_jtl1 = {}; L_jtl2 = {}; L_si = {}'.format(L_jtl1,L_jtl2,L_si))
                # I1, I2, I3, Ijsf, Ijtl, Ijsi, Lj1, Lj2, Lj3  = synapse_current_distribution__3jj(Ic,L_jtl1,L_jtl2,L_si,I_bias_list,I_bias_sy,I_bias_jtl,I_bias_si)
                # I_bias_list_2_perm = I_bias_list[2]
            
                #     dt_spk = _pt - spike_times[st_ind]
                #     spd_t_ind  = (np.abs(spd_t[:] - dt_spk)).argmin()
                                    
                #     # update current distribution throughout circuit and JJ inductances
                #     I_bias_list[2] = I_bias_list_2_perm - I_si_vec[ii]                                
                #     I1, I2, I3, Ijsf, Ijtl, Ijsi, Ljsf, Ljtl, Ljsi = synapse_current_distribution__3jj(Ic,L_jtl1,L_jtl2,L_si,I_bias_list,Ijsf,Ijtl,Ijsi)
                #     I_sf_ind_spd = (np.abs( I_sf_list__spd[:] - Ijsf )).argmin()
                #     spd_i = spd_response_array[I_sf_ind_spd]
                    
                #     # I_sy_ind_spd = (np.abs( I_sy_list__spd[:] - I_bias_sy )).argmin()
                #     # spd_i = spd_response_array[I_sy_ind_spd]
                    
                #     # this block to avoid spd drive going too low at the onset of each spike 
                #     if st_ind - st_ind_last == 1:
                #         spd_current = np.max([I_spd_vec[ii-1],spd_i[spd_t_ind]])
                #         spd_current_memory = spd_current
                #     if spd_current_memory > 0 and spd_i[spd_t_ind] < spd_current_memory:
                #         spd_current = spd_current_memory
                #     else:
                #         spd_current = spd_i[spd_t_ind]
                #         spd_current_memory = 0
                        
                #     I_spd_vec[ii] = spd_current
                    
                #     st_ind_last = st_ind
                    
                #     I_drive = I_bias_sy+spd_current-Ic
                #     if I_drive < np.min(I_drive_list):
                #         gf = 0
                #     else:
                #         I_drive_ind = (np.abs(I_drive_list[:] - I_drive)).argmin()
                #         I_si_ind = (np.abs(I_si_array[I_drive_ind] - I_si_vec[ii])).argmin()
                #         gf = dt*I_fq*rate_array[I_drive_ind][I_si_ind] # growth factor
                #         # gf = dt*I_fq*master_rate_matrix__imported[I_drive_ind,I_si_ind] 
                       
                #         # linear interpolation
                #         # rate = np.interp(spd_current,I_drive_vec__imported,master_rate_matrix__imported[:,I_si_ind])
                #         # gf = dt*I_fq*rate                                
               
                # I_si_vec[ii+1] = gf + (1-dt/tau_si)*I_si_vec[ii] 
                
    print('done time stepping')
    return I_spd_vec, I_si_vec, I_sf_vec

    
def synapse_current_distribution__2jj(Ic,Ljtl,Lsi,Ib,Isf,Ijtl,Isi1,Isi2):
    
    Isy = Ib[0]
    Isc = Ib[1]
    
    Ljsf = Ljj_pH(Ic,Isf)
    Ljsi = Ljj_pH(Ic,Isi1)    
    
    denom = ( Ljsi*(Ljsf+Ljtl)+(Ljsf+Ljsi+Ljtl)*Lsi )
    
    # print('Lj1 = {}; Lj2 = {}; Lj3 = {}'.format(Lj1,Lj2,Lj3))
        
    Isf = ( Isy*Ljsi*Ljtl+Isc*Ljsi*Lsi+Isy*(Ljsi+Ljtl)*Lsi ) / denom
    
    Ijtl = ( -Isc*Ljsi*Lsi+Isy*Ljsf*(Ljsi+Lsi) ) / denom
    
    Isi1 = ( (Isy*Ljsf+Isc*(Ljsf+Ljtl))*Lsi ) / denom
    
    Isi2 = ( Isy*Ljsf*Ljsi+Isc*Ljsi*(Ljsf+Ljtl) ) / denom
      
    return Isf, Ijtl, Isi1, Isi2, Ljsf, Ljsi 


def synapse_current_distribution__3jj(Ic,L1,L2,L3,Ib,Ij1,Ij2,Ij3):
    
    Ib1 = Ib[0]
    Ib2 = Ib[1]
    Ib3 = Ib[2]
    
    Lj1 = Ljj_pH(Ic,Ij1)
    Lj2 = Ljj_pH(Ic,Ij2)
    Lj3 = Ljj_pH(Ic,Ij3)
    
    # print('Lj1 = {}; Lj2 = {}; Lj3 = {}'.format(Lj1,Lj2,Lj3))
        
    I1 = ( (-Lj2*(Ib2*L2*L3+Ib3*L3*Lj3+Ib2*(L2+L3)*Lj3)+Ib1*Lj1*(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3))/
                (L2*L3*(L1+Lj1)+L3*(L1+L2+Lj1)*Lj2+((L2+L3)*(L1+Lj1)+(L1+L2+L3+Lj1)*Lj2)*Lj3) )

    I2 = ( (-Ib3*L3*(L1+Lj1+Lj2)*Lj3+Ib1*Lj1*Lj2*(L3+Lj3)+Ib2*(L1+Lj1)*Lj2*(L3+Lj3))/
                (L2*L3*(L1+Lj1)+L3*(L1+L2+Lj1)*Lj2+((L2+L3)*(L1+Lj1)+(L1+L2+L3+Lj1)*Lj2)*Lj3) )

    I3 = ( ((Ib3*L2*(L1+Lj1)+(Ib1*Lj1+Ib2*(L1+Lj1)+Ib3*(L1+L2+Lj1))*Lj2)*Lj3)/
                (L2*L3*(L1+Lj1)+L3*(L1+L2+Lj1)*Lj2+((L2+L3)*(L1+Lj1)+(L1+L2+L3+Lj1)*Lj2)*Lj3) )
    
    Ij1 = ( (L3*(Ib1*L1*L2+Ib2*L2*Lj2+Ib1*(L1+L2)*Lj2)+(Ib1*L1*(L2+L3)+(Ib3*L3+Ib2*(L2+L3)+Ib1*(L1+L2+L3))*Lj2)*Lj3)/
                 (L2*L3*(L1+Lj1)+L3*(L1+L2+Lj1)*Lj2+((L2+L3)*(L1+Lj1)+(L1+L2+L3+Lj1)*Lj2)*Lj3) )
    
    Ij2 = ( (Ib3*L3*(L1+Lj1)*Lj3+Ib1*Lj1*(L2*L3+(L2+L3)*Lj3)+Ib2*(L1+Lj1)*(L3*Lj3+L2*(L3+Lj3)))/
                 (L2*L3*(L1+Lj1)+L3*(L1+L2+Lj1)*Lj2+((L2+L3)*(L1+Lj1)+(L1+L2+L3+Lj1)*Lj2)*Lj3) )
    
    Ij3 = ( (Ib3*L2*L3*(L1+Lj1)+L3*(Ib1*Lj1+Ib2*(L1+Lj1)+Ib3*(L1+L2+Lj1))*Lj2)/
                 (L2*L3*(L1+Lj1)+L3*(L1+L2+Lj1)*Lj2+((L2+L3)*(L1+Lj1)+(L1+L2+L3+Lj1)*Lj2)*Lj3) )
      
    return I1, I2, I3, Ij1, Ij2, Ij3, Lj1, Lj2, Lj3 


def dendrite_time_stepper(time_vec,I_drive,L3,tau_di):
    
    with open('../_circuit_data/master__dnd__rate_matrix.soen', 'rb') as data_file:         
        data_array_imported = pickle.load(data_file)
    
    I_di_list__imported = data_array_imported['I_di_list']
    I_drive_vec__imported = data_array_imported['I_drive_vec']
    master_rate_matrix__imported = data_array_imported['master_rate_matrix']
        
    p = physical_constants()
    Phi0 = p['Phi0']
    I_fq = Phi0/L3
    
    I_di_vec = np.zeros([len(time_vec),1])
    for ii in range(len(time_vec)-1):
        dt = time_vec[ii+1]-time_vec[ii]
                               
        if I_drive[ii] > 18.6e-6:
            ind1 = (np.abs(np.asarray(I_drive_vec__imported)-I_drive[ii])).argmin()
            ind2 = (np.abs(np.asarray(I_di_list__imported[ind1])-I_di_vec[ii])).argmin()
            rate = master_rate_matrix__imported[ind1,ind2]
            # linear interpolation
            # rate = np.interp(I_drive[ii],I_drive_vec__imported,master_rate_matrix__imported[:,ind2])            
        else:
            rate = 0

        I_di_vec[ii+1] = rate*I_fq*dt + (1-dt/tau_di)*I_di_vec[ii]        
    
    return I_di_vec

def dendritic_time_stepper(time_vec,R,I_drive,I_b,Ic,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,tau_di,mu_1,mu_2):
    
    p = physical_constants()
    Phi0 = p['Phi0']
    prefactor = Ic*R/Phi0
    I_fq = Phi0/L3
        
    #initial approximations
    Lj0 = Ljj(Ic,0)
    Iflux = 0
    Idr2_prev = ((Lm2+Ldr1+Lj0)*I_b[0]+M_direct*Iflux)/( Lm2+Ldr1+Ldr2+2*Lj0 + (Lm2+Ldr1+Lj0)*(Ldr2+Lj0)/L1 )
    Idr1_prev = I_b[0]-( 1 + (Ldr2+Lj0)/L1 )*Idr2_prev
    Ij2_prev = I_b[1]
    Ij3_prev = I_b[2]
    
    I_di_vec = np.zeros([len(time_vec),1])
    # Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,0,I_b[0],I_b[1],I_b[2],M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
    # print('Idr1 = {}uA, Idr2 = {}uA, Ij2 = {}uA, Ij3 = {}uA'.format(Idr1_next*1e6,Idr2_next*1e6,Ij2_next*1e6,Ij3_next*1e6))
    for ii in range(len(time_vec)-1):
        dt = time_vec[ii+1]-time_vec[ii]
                               
                                                              #dendrite_current_splitting(Ic,  Iflux,        Ib1,   Ib2,   Ib3,   M,       Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
        Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,I_drive[ii+1],I_b,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
        # I_di_sat = I_di_sat_of_I_dr_2(Idr2_next)
        
        Ljdr2 = Ljj(Ic,Idr2_next)
        Lj2 = Ljj(Ic,Ij2_next)
        Lj3 = Ljj(Ic,Ij3_next)        
        # Ljdr2 = Ljj(Ic,0)
        # Lj2 = Ljj(Ic,0)
        # Lj3 = Ljj(Ic,0)
        
        Idr1_prev = Idr1_next
        Idr2_prev = Idr2_next
        Ij2_prev = Ij2_next#(Lj3/(L2+Lj2))*I_di_vec[ii]
        Ij3_prev = Ij3_next
        
        # I_j_df_fluxon_soen = Phi0/(L1+Ldr2+Ljdr2+Lj2)
        # I_j_2_fluxon_soen = Phi0/(Lj2+L_pp)
        # I_j_3_fluxon_soen = Phi0/(L3+Lj3)
        
        I_loop2_from_di = (Lj3/(L2+Lj2))*I_di_vec[ii]
        I_loop1_from_loop2 = (Lj2/(L1+Ljdr2+Ldr2))*I_loop2_from_di
        # print('I_loop2_from_di = {}uA, I_loop1_from_loop2 = {}uA'.format(I_loop2_from_di*1e6,I_loop1_from_loop2*1e6))
        
        Idr2_next -= I_loop1_from_loop2
        Ij2_next -= I_loop2_from_di
        Ij3_next -= I_di_vec[ii] - I_loop2_from_di        
                
        L_ppp = Lj3*L3/(Lj3+L3)
        L_pp = L2+L_ppp
        L_p = Lj2*L_pp/(Lj2+L_pp)
        # print('L_p = {}pH, L_pp = {}pH, L_ppp = {}pH'.format(L_p*1e12,L_pp*1e12,L_ppp*1e12))
        
        large_number = 1e9
        
        I_flux_1 = 6e-6
        I_flux_2 = 20e-6
        
        Ij2_next += I_flux_1 # (Phi0/(L1+L_p))*(L_pp)/(Lj2+L_pp)#(Lj3/(L2+Lj2))*I_di_vec[ii]
        # print('Ij2_next += {}uA'.format(1e6*(Phi0/(L1+L_p))*(L_pp)/(Lj2+L_pp)))
        # Ij3_next += (Phi0/L_pp)*(L3/(L3+Lj3)) + (Phi0/(L1+L_p))*L3/(Lj3+L3) - I_di_vec[ii]
        Ij3_next += I_flux_2 # (Phi0/L_pp)*(L3/(L3+Lj3))
        # print('Ij3_next += {}uA'.format(1e6*(Phi0/L_pp)*(L3/(L3+Lj3))))
        # print('term_1 = {}; term_2 = {}'.format( (Phi0/L_pp)*(L3/(L3+Lj3)) , (Phi0/(L1+L_p))*L3/(Lj3+L3) ) )
        if Idr2_next > Ic:
            factor_1 = inter_fluxon_interval(Idr2_next) # ( (Idr2_next/Ic)**mu_1 - 1 )**(-mu_2)            
        else:
            factor_1 = large_number
        if Ij2_next > Ic:
            factor_2 = inter_fluxon_interval(Ij2_next) # ( (Ij2_next/Ic)**mu_1 - 1 )**(-mu_2)  
        else:
            factor_2 = large_number
        if Ij3_next > Ic:
            factor_3 = inter_fluxon_interval(Ij3_next) # ( (Ij3_next/Ic)**mu_1 - 1 )**(-mu_2)  
        else:
            factor_3 = large_number

        # print('factor_1 = {}, factor_2 = {}, factor_3 = {}'.format(factor_1,factor_2,factor_3))
        r_tot = (factor_1+factor_2+factor_3)**(-1)
        I_di_vec[ii+1] = r_tot*I_fq*dt + (1-dt/tau_di)*I_di_vec[ii]        
    
    return I_di_vec

def dendritic_time_stepper_old2(time_vec,A_prefactor,I_drive,I_b,I_th,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,tau_di,mu_1,mu_2,mu_3,mu_4):
        
    #initial approximations
    Lj0 = Ljj(I_th,0)
    Iflux = 0
    Idr2_prev = ((Lm2+Ldr1+Lj0)*I_b[0]+M_direct*Iflux)/( Lm2+Ldr1+Ldr2+2*Lj0 + (Lm2+Ldr1+Lj0)*(Ldr2+Lj0)/L1 )
    Idr1_prev = I_b[0]-( 1 + (Ldr2+Lj0)/L1 )*Idr2_prev
    Ij2_prev = I_b[1]
    Ij3_prev = I_b[2]
    
    I_di_vec = np.zeros([len(time_vec),1])
    for ii in range(len(time_vec)-1):
        dt = time_vec[ii+1]-time_vec[ii]
                                                                                            
        Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(I_th,I_drive[ii+1],I_b[0],I_b[1],I_b[2],M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
        I_di_sat = I_di_sat_of_I_dr_2(Idr2_next)
        if Idr2_next > I_th:
            factor_1 = ( (Idr2_next/I_th)**mu_1 - 1 )**mu_2            
        else:
            factor_1 = 0
        if I_di_vec[ii] <= I_di_sat:
            factor_2 = ( 1-(I_di_vec[ii]/I_di_sat)**mu_3 )**mu_4
        else:
            factor_2 = 0
        I_di_vec[ii+1] = dt * A_prefactor * factor_1 * factor_2 + (1-dt/tau_di)*I_di_vec[ii]
        Idr1_prev = Idr1_next
        Idr2_prev = Idr2_next
        Ij2_prev = Ij2_next
        Ij3_prev = Ij3_next
    
    return I_di_vec

def dendritic_drive__piecewise_linear(time_vec,pwl):
    
    input_signal__dd = np.zeros([len(time_vec),1])
    for ii in range(len(pwl)-1):
        t1_ind = (np.abs(np.asarray(time_vec)-pwl[ii][0])).argmin()
        t2_ind = (np.abs(np.asarray(time_vec)-pwl[ii+1][0])).argmin()
        slope = (pwl[ii+1][1]-pwl[ii][1])/(pwl[ii+1][0]-pwl[ii][0])
        # print('t1_ind = {}'.format(t1_ind))
        # print('t2_ind = {}'.format(t2_ind))
        # print('slope = {}'.format(slope))
        partial_time_vec = time_vec[t1_ind:t2_ind+1]
        input_signal__dd[t1_ind] = pwl[ii][1]
        for jj in range(len(partial_time_vec)-1):
            input_signal__dd[t1_ind+jj+1] = input_signal__dd[t1_ind+jj]+(partial_time_vec[jj+1]-partial_time_vec[jj])*slope
    input_signal__dd[t2_ind:] = input_signal__dd[t2_ind]*np.ones([len(time_vec)-t2_ind,1])
    
    return input_signal__dd

def dendritic_drive__exp_pls_train__LR(time_vec,exp_pls_trn_params):
        
    t_r1_start = exp_pls_trn_params['t_r1_start']
    t_r1_rise = exp_pls_trn_params['t_r1_rise']
    t_r1_pulse = exp_pls_trn_params['t_r1_pulse']
    t_r1_fall = exp_pls_trn_params['t_r1_fall']
    t_r1_period = exp_pls_trn_params['t_r1_period']
    value_r1_off = exp_pls_trn_params['value_r1_off']
    value_r1_on = exp_pls_trn_params['value_r1_on']
    r2 = exp_pls_trn_params['r2']
    L1 = exp_pls_trn_params['L1']
    L2 = exp_pls_trn_params['L2']
    Ib = exp_pls_trn_params['Ib']
    
    # make vector of r1(t)
    sq_pls_trn_params = dict()
    sq_pls_trn_params['t_start'] = t_r1_start
    sq_pls_trn_params['t_rise'] = t_r1_rise
    sq_pls_trn_params['t_pulse'] = t_r1_pulse
    sq_pls_trn_params['t_fall'] = t_r1_fall
    sq_pls_trn_params['t_period'] = t_r1_period
    sq_pls_trn_params['value_off'] = value_r1_off
    sq_pls_trn_params['value_on'] = value_r1_on
    # print('making resistance vec ...')
    r1_vec = dendritic_drive__square_pulse_train(time_vec,sq_pls_trn_params)
    
    input_signal__dd = np.zeros([len(time_vec),1])
    # print('time stepping ...')
    for ii in range(len(time_vec)-1):
        # print('ii = {} of {}'.format(ii+1,len(time_vec)-1))
        dt = time_vec[ii+1]-time_vec[ii]
        input_signal__dd[ii+1] = input_signal__dd[ii]*( 1 - dt*(r1_vec[ii]+r2)/(L1+L2) ) + dt*Ib*r1_vec[ii]/(L1+L2)
    
    return input_signal__dd

def dendritic_drive__exponential(time_vec,exp_params):
        
    t_rise = exp_params['t_rise']
    t_fall = exp_params['t_fall']
    tau_rise = exp_params['tau_rise']
    tau_fall = exp_params['tau_fall']
    value_on = exp_params['value_on']
    value_off = exp_params['value_off']
    
    input_signal__dd = np.zeros([len(time_vec),1])
    for ii in range(len(time_vec)):
        time = time_vec[ii]
        if time < t_rise:
            input_signal__dd[ii] = value_off
        if time >= t_rise and time < t_fall:
            input_signal__dd[ii] = value_off+(value_on-value_off)*(1-np.exp(-(time-t_rise)/tau_rise))
        if time >= t_fall:
            input_signal__dd[ii] = value_off+(value_on-value_off)*(1-np.exp(-(time-t_rise)/tau_rise))*np.exp(-(time-t_fall)/tau_fall)
    
    return input_signal__dd

def dendritic_drive__square_pulse_train(time_vec,sq_pls_trn_params):
    
    input_signal__dd = np.zeros([len(time_vec),1])
    dt = time_vec[1]-time_vec[0]
    t_start = sq_pls_trn_params['t_start']
    t_rise = sq_pls_trn_params['t_rise']
    t_pulse = sq_pls_trn_params['t_pulse']
    t_fall = sq_pls_trn_params['t_fall']
    t_period = sq_pls_trn_params['t_period']
    value_off = sq_pls_trn_params['value_off']
    value_on = sq_pls_trn_params['value_on']
    
    tf_sub = t_rise+t_pulse+t_fall
    time_vec_sub = np.arange(0,tf_sub+dt,dt)
    pwl = [[0,value_off],[t_rise,value_on],[t_rise+t_pulse,value_on],[t_rise+t_pulse+t_fall,value_off]]
    
    pulse = dendritic_drive__piecewise_linear(time_vec_sub,pwl)    
    num_pulses = np.floor((time_vec[-1]-t_start)/t_period).astype(int)        
    ind_start = (np.abs(np.asarray(time_vec)-t_start)).argmin()
    ind_pulse_end = (np.abs(np.asarray(time_vec)-t_start-t_rise-t_pulse-t_fall)).argmin()
    ind_per_end = (np.abs(np.asarray(time_vec)-t_start-t_period)).argmin()
    num_ind_pulse = len(pulse) # ind_pulse_end-ind_start
    num_ind_per = ind_per_end-ind_start
    for ii in range(num_pulses):
        input_signal__dd[ind_start+ii*num_ind_per:ind_start+ii*num_ind_per+num_ind_pulse] = pulse[:]
        
    if t_start+num_pulses*t_period <= time_vec[-1] and t_start+(num_pulses+1)*t_period >= time_vec[-1]:
        ind_final = (np.abs(np.asarray(time_vec)-t_start-num_pulses*t_period)).argmin()
        ind_end = (np.abs(np.asarray(time_vec)-t_start-num_pulses*t_period-t_rise-t_pulse-t_fall)).argmin()
        num_ind = ind_end-ind_final
        input_signal__dd[ind_final:ind_end] = pulse[0:num_ind]
        
    return input_signal__dd

# def dendritic_drive__linear_ramp(time_vec, time_on = 5e-9, slope = 1e-6/1e-9):
    
#     t_on_ind = (np.abs(np.asarray(time_vec)-time_on)).argmin()
#     input_signal__dd = np.zeros([len(time_vec),1])
#     partial_time_vec = time_vec[t_on_ind:]
#     for ii in range(len(partial_time_vec)):
#         time = partial_time_vec[ii]
#         input_signal__dd[t_on_ind+ii] = (time-time_on)*slope
    
#     return input_signal__dd

def dendrite_current_splitting(Ic,Iflux,Ib,M,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev):
    # print('Ic = {}'.format(Ic))
    # print('Iflux = {}'.format(Iflux))
    # print('Ib1 = {}'.format(Ib1))
    # print('Ib2 = {}'.format(Ib2))
    # print('Ib3 = {}'.format(Ib3))
    # print('M = {}'.format(M))
    # print('Lm2 = {}'.format(Lm2))
    # print('Ldr1 = {}'.format(Ldr1))
    # print('Ldr2 = {}'.format(Ldr2))
    # print('L1 = {}'.format(L1))
    # print('L2 = {}'.format(L2))
    # print('L3 = {}'.format(L3))
    # pause(10)
    #see pgs 74, 75 in green lab notebook from 2020_04_01
    
    Ib1 = Ib[0]
    Ib2 = Ib[1]
    Ib3 = Ib[2]
    
    # Lj0 = Ljj(Ic,0)
    Lj2 = Ljj(Ic,Ij2_prev)
    Lj3 = Ljj(Ic,Ij3_prev)
    Ljdr1 = Ljj(Ic,Idr1_prev)
    Ljdr2 = Ljj(Ic,Idr2_prev)
    
    Idr1_next = ( -((-Lj2*(-Ib3*L3*Lj3-Ib2*(L3*Lj3+L2*(L3+Lj3)))
                    +Ib1*(-Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    +L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))))
                    *(-Ldr2-Ljdr2)-Iflux*(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))*M)
                    /((Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3)))
                    *(-Ldr2-Ljdr2)-(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))
                    *(Ldr1+Ljdr1+Lm2)) )
                 
    Idr2_next = ( (Iflux*M)/(Ldr2+Ljdr2)+((Ldr1+Ljdr1+Lm2)
                    *((-Lj2*(-Ib3*L3*Lj3-Ib2*(L3*Lj3+L2*(L3+Lj3)))
                    +Ib1*(-Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    +L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))))
                    *(-Ldr2-Ljdr2)-Iflux*(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    *(Ldr2+Ljdr2))*M))/((-Ldr2-Ljdr2)
                    *((Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3)))
                    *(-Ldr2-Ljdr2)-(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))
                    *(Ldr1+Ljdr1+Lm2))) )
                                        
    Ij2_next = ( (1/Lj2)*(-Ib1*L1+(Iflux*(L1+Ldr2+Ljdr2)*M)
                     /(Ldr2+Ljdr2)-(((Ldr2+Ljdr2)*(Ldr1+Ljdr1+Lm2)
                    +L1*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2))
                    *(-(Lj2*(Ib2*L2*L3+Ib3*L3*Lj3+Ib2*(L2+L3)*Lj3)
                    +Ib1*(L1*L3*(L2+Lj2)+L3*Lj2*Lj3
                    +L1*(L2+L3+Lj2)*Lj3+L2*Lj2*(L3+Lj3)))
                    *(Ldr2+Ljdr2)-Iflux*(-L1*(L3*(L2+Lj2)
                    +(L2+L3+Lj2)*Lj3)-Lj2*(L3*Lj3+L2*(L3+Lj3))
                    -(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr2+Ljdr2))*M))
                    /((Ldr2+Ljdr2)*((L1*L3*(L2+Lj2)+L3*Lj2*Lj3
                    +L1*(L2+L3+Lj2)*Lj3+L2*Lj2*(L3+Lj3))
                    *(Ldr2+Ljdr2)-(-L1*(L3*(L2+Lj2)
                    +(L2+L3+Lj2)*Lj3)-Lj2*(L3*Lj3+L2*(L3+Lj3))
                    -(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    *(Ldr2+Ljdr2))*(Ldr1+Ljdr1+Lm2)))) )                                       
                                                                                                 
    Ij3_next = ( (L3*(Ib3*(Lj2*(Ldr2+Ljdr2)*(Ldr1+Ljdr1+Lm2)
                    +L1*(L2+Lj2)*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    +L2*(Lj2*Ljdr1+Lj2*Ljdr2+Ljdr1*Ljdr2
                    +Ldr1*(Ldr2+Lj2+Ljdr2)+(Lj2+Ljdr2)*Lm2
                    +Ldr2*(Lj2+Ljdr1+Lm2)))+Lj2*(Ib2*((Ldr2+Ljdr2)
                    *(Ldr1+Ljdr1+Lm2)+L1*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2))
                    +(Ldr2+Ljdr2)*(Ib1*(Ldr1+Ljdr1+Lm2)+Iflux*M))))
                    /(L3*Ldr1*Ldr2*Lj2+L3*Ldr1*Ldr2*Lj3
                    +L3*Ldr1*Lj2*Lj3+L3*Ldr2*Lj2*Lj3+Ldr1*Ldr2*Lj2*Lj3
                    +L3*Ldr2*Lj2*Ljdr1+L3*Ldr2*Lj3*Ljdr1+L3*Lj2*Lj3*Ljdr1
                    +Ldr2*Lj2*Lj3*Ljdr1+L3*Ldr1*Lj2*Ljdr2+L3*Ldr1*Lj3*Ljdr2
                    +L3*Lj2*Lj3*Ljdr2+Ldr1*Lj2*Lj3*Ljdr2+L3*Lj2*Ljdr1*Ljdr2
                    +L3*Lj3*Ljdr1*Ljdr2+Lj2*Lj3*Ljdr1*Ljdr2+(Lj2*Lj3*(Ldr2+Ljdr2)
                    +L3*(Lj2*Lj3+Ldr2*(Lj2+Lj3)+(Lj2+Lj3)*Ljdr2))*Lm2
                    +L1*(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    +L2*(L3+Lj3)*(Lj2*Ljdr1+Lj2*Ljdr2+Ljdr1*Ljdr2
                    +Ldr1*(Ldr2+Lj2+Ljdr2)+(Lj2+Ljdr2)*Lm2+Ldr2*(Lj2+Ljdr1+Lm2))) )
    
    I1_next = ( Ib1-(Iflux*M)/(Ldr2+Ljdr2)+((Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    *(-(Lj2*(Ib2*L2*L3+Ib3*L3*Lj3+Ib2*(L2+L3)*Lj3)
                    +Ib1*(L1*L3*(L2+Lj2)+L3*Lj2*Lj3+L1*(L2+L3+Lj2)
                    *Lj3+L2*Lj2*(L3+Lj3)))*(Ldr2+Ljdr2)
                    -Iflux*(-L1*(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    -Lj2*(L3*Lj3+L2*(L3+Lj3))-(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    *(Ldr2+Ljdr2))*M))/((Ldr2+Ljdr2)*((L1*L3*(L2+Lj2)
                    +L3*Lj2*Lj3+L1*(L2+L3+Lj2)*Lj3+L2*Lj2*(L3+Lj3))
                    *(Ldr2+Ljdr2)-(-L1*(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    -Lj2*(L3*Lj3+L2*(L3+Lj3))-(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    *(Ldr2+Ljdr2))*(Ldr1+Ljdr1+Lm2))) )
                                                       
    I2_next = ( Ib1+Ib2+(Ib1*L1)/Lj2-(Iflux*(L1+Ldr2+Lj2+Ljdr2)*M)
                    /(Lj2*(Ldr2+Ljdr2))+((L1+Lj2+((L1+Ldr2+Lj2+Ljdr2)*
                    (Ldr1 + Ljdr1 + Lm2))/(Ldr2+Ljdr2))
                    *(-(Lj2*(Ib2*L2*L3+Ib3*L3*Lj3+Ib2*(L2+L3)*Lj3)
                    +Ib1*(L1*L3*(L2+Lj2)+L3*Lj2*Lj3
                    +L1*(L2+L3+Lj2)*Lj3+L2*Lj2*(L3+Lj3)))
                    *(Ldr2+Ljdr2)-Iflux*(-L1*(L3*(L2+Lj2)
                    +(L2+L3+Lj2)*Lj3)-Lj2*(L3*Lj3+L2*(L3+Lj3))
                    -(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr2+Ljdr2))*M))
                    /(Lj2*((L1*L3*(L2+Lj2)+L3*Lj2*Lj3+L1*(L2+L3+Lj2)*Lj3
                    +L2*Lj2*(L3+Lj3))*(Ldr2+Ljdr2)-(-L1*(L3*(L2+Lj2)
                    +(L2+L3+Lj2)*Lj3)-Lj2*(L3*Lj3+L2*(L3+Lj3))
                    -(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr2+Ljdr2))*(Ldr1+Ljdr1+Lm2))) )
                                                         
    I3_next = ( (Lj3*(Ib3*(Lj2*(Ldr2+Ljdr2)*(Ldr1+Ljdr1+Lm2)
                    +L1*(L2+Lj2)*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    +L2*(Lj2*Ljdr1+Lj2*Ljdr2+Ljdr1*Ljdr2
                    +Ldr1*(Ldr2+Lj2+Ljdr2)+(Lj2+Ljdr2)*Lm2
                    +Ldr2*(Lj2+Ljdr1+Lm2)))+Lj2*(Ib2*((Ldr2+Ljdr2)*(Ldr1+Ljdr1+Lm2)
                    +L1*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2))+(Ldr2+Ljdr2)*(Ib1*(Ldr1+Ljdr1+Lm2)
                    +Iflux*M))))/(L3*Ldr1*Ldr2*Lj2+L3*Ldr1*Ldr2*Lj3+L3*Ldr1*Lj2*Lj3
                    +L3*Ldr2*Lj2*Lj3+Ldr1*Ldr2*Lj2*Lj3
                    +L3*Ldr2*Lj2*Ljdr1+L3*Ldr2*Lj3*Ljdr1
                    +L3*Lj2*Lj3*Ljdr1+Ldr2*Lj2*Lj3*Ljdr1
                    +L3*Ldr1*Lj2*Ljdr2+L3*Ldr1*Lj3*Ljdr2
                    +L3*Lj2*Lj3*Ljdr2+Ldr1*Lj2*Lj3*Ljdr2
                    +L3*Lj2*Ljdr1*Ljdr2+L3*Lj3*Ljdr1*Ljdr2
                    +Lj2*Lj3*Ljdr1*Ljdr2+(Lj2*Lj3*(Ldr2+Ljdr2)
                    +L3*(Lj2*Lj3+Ldr2*(Lj2+Lj3)+(Lj2+Lj3)*Ljdr2))*Lm2
                    +L1*(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    +L2*(L3+Lj3)*(Lj2*Ljdr1+Lj2*Ljdr2+Ljdr1*Ljdr2
                    +Ldr1*(Ldr2+Lj2+Ljdr2)+(Lj2+Ljdr2)*Lm2
                    +Ldr2*(Lj2+Ljdr1+Lm2))) )
                                                
    return Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1_next, I2_next, I3_next

def dendrite_current_splitting__old(Ic,Iflux,Ib1,Ib2,Ib3,M,Lm2,Ldr1,Ldr2,L1,L2,L3):
    # print('Ic = {}'.format(Ic))
    # print('Iflux = {}'.format(Iflux))
    # print('Ib1 = {}'.format(Ib1))
    # print('Ib2 = {}'.format(Ib2))
    # print('Ib3 = {}'.format(Ib3))
    # print('M = {}'.format(M))
    # print('Lm2 = {}'.format(Lm2))
    # print('Ldr1 = {}'.format(Ldr1))
    # print('Ldr2 = {}'.format(Ldr2))
    # print('L1 = {}'.format(L1))
    # print('L2 = {}'.format(L2))
    # print('L3 = {}'.format(L3))
    # pause(10)
    #see pgs 74, 75 in green lab notebook from 2020_04_01
    
    Lj0 = Ljj(Ic,0)
    Lj2 = Ljj(Ic,Ib2)#approximation; current passing through jj2 is not exactly Ib2
    Lj3 = Ljj(Ic,Ib3)#approximation; current passing through jj3 is not exactly Ib3
    
    #initial approximations
    Idr2_prev = ((Lm2+Ldr1+Lj0)*Ib1+M*Iflux)/( Lm2+Ldr1+Ldr2+2*Lj0 + (Lm2+Ldr1+Lj0)*(Ldr2+Lj0)/L1 )
    Idr1_prev = Ib1-( 1 + (Ldr2+Lj0)/L1 )*Idr2_prev
    # I1_prev = Ib1-Idr1_prev-Idr2_prev
    
    Idr1_next = Ib1/2
    Idr2_next = Ib1/2
    num_it = 1
    while abs((Idr2_next-Idr2_prev)/Idr2_next) > 1e-4:
        
        # print('num_it = {:d}'.format(num_it))
        num_it += 1
        
        Idr1_prev = Idr1_next
        Idr2_prev = Idr2_next
        
        Ljdr1 = Ljj(Ic,Idr1_prev)
        Ljdr2 = Ljj(Ic,Idr2_prev)
        
        Idr1_next = ( -((-Lj2*(-Ib3*L3*Lj3-Ib2*(L3*Lj3+L2*(L3+Lj3)))
                        +Ib1*(-Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        +L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))))
                        *(-Ldr2-Ljdr2)-Iflux*(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))*M)
                        /((Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3)))
                        *(-Ldr2-Ljdr2)-(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))
                        *(Ldr1+Ljdr1+Lm2)) )
                     
        Idr2_next = ( (Iflux*M)/(Ldr2+Ljdr2)+((Ldr1+Ljdr1+Lm2)
                        *((-Lj2*(-Ib3*L3*Lj3-Ib2*(L3*Lj3+L2*(L3+Lj3)))
                        +Ib1*(-Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        +L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))))
                        *(-Ldr2-Ljdr2)-Iflux*(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        *(Ldr2+Ljdr2))*M))/((-Ldr2-Ljdr2)
                        *((Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3)))
                        *(-Ldr2-Ljdr2)-(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))
                        *(Ldr1+Ljdr1+Lm2))) )
                                            
        if num_it > 10:
            # print('dendrite_current_splitting _ num_it > 10 _ convergence unlikely\nIdr2_prev = {}, Idr2_next = {}\n\n'.format(Idr2_prev,Idr2_next))
            break
                                            
    Idr = Idr2_next
    
    return Idr

def Ljj(critical_current,current):
    
    norm_current = np.max([np.min([current/critical_current,1]),1e-9])
    L = (3.2910596281416393e-16/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

def Ljj_pH(critical_current,current):
    
    norm_current = np.max([np.min([current/critical_current,1]),1e-9])
    L = (3.2910596281416393e2/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

def I_di_sat_of_I_dr_2(Idr2):
    
    if Idr2 < 40e-6:
        I_di_sat = 10e-6
    if Idr2 >= 40e-6 and Idr2 <= 44.7416e-6:
        I_di_sat = 1.77837*Idr2-61.34579e-6
    if Idr2 > 44.7416e-6:
        I_di_sat = 18.0660e-6
    
    return I_di_sat

def amp_fitter(time_vec,I_di):
    
    time_pts = [10,15,20,25,30,35,40,45,50]
    target_vals = [22.7608,37.0452,51.0996,65.3609,79.6264,93.8984,107.963,122.233,136.493]
    actual_vals = np.zeros([len(time_pts),1])
    
    for ii in range(len(time_pts)):
        ind = (np.abs(time_vec*1e9-time_pts[ii])).argmin()
        actual_vals[ii] = I_di[ind]
    
    error = 0
    for ii in range(len(time_pts)):
        error += abs( target_vals[ii]-actual_vals[ii]*1e9 )**2
    
    return error

def mu_fitter(data_dict,time_vec,I_di,mu1,mu2,amp):
    
    time_vec_spice = data_dict['time']
    target_vec = data_dict['L9#branch']

    # fig, ax = plt.subplots(nrows = 1, ncols = 1)   
    # fig.suptitle('Comparing WR and soen_sim')
    # plt.title('amp = {}; mu1 = {}; mu2 = {}'.format(amp,mu1,mu2))
    
    # ax.plot(time_vec_spice*1e9,target_vec*1e9,'o-', label = 'WR')        
    # ax.plot(time_vec*1e9,I_di*1e9,'o-', label = 'soen_sim')    
    # ax.legend()
    # ax.set_xlabel(r'Time [ns]')
    # ax.set_ylabel(r'$I_{di} [nA]$') 
    
    error = 0
    norm = 0
    for ii in range(len(time_vec)):
        ind = (np.abs(time_vec_spice-time_vec[ii])).argmin()
        error += abs( target_vec[ind]-I_di[ii] )**2
        norm += abs( target_vec[ind] )**2
    
    error = error/norm
    
    return error

def mu_fitter_3_4(data_dict,time_vec,I_di,mu3,mu4):
    
    time_vec_spice = data_dict['time']
    target_vec = data_dict['L9#branch']

    # fig, ax = plt.subplots(nrows = 1, ncols = 1)   
    # fig.suptitle('Comparing WR and soen_sim')
    # plt.title('amp = {}; mu1 = {}; mu2 = {}'.format(amp,mu1,mu2))
    
    # ax.plot(time_vec_spice*1e9,target_vec*1e9,'o-', label = 'WR')        
    # ax.plot(time_vec*1e9,I_di*1e9,'o-', label = 'soen_sim')    
    # ax.legend()
    # ax.set_xlabel(r'Time [ns]')
    # ax.set_ylabel(r'$I_{di} [nA]$') 
    
    error = 0
    norm = 0
    for ii in range(len(time_vec)):
        ind = (np.abs(time_vec_spice-time_vec[ii])).argmin()
        error += abs( target_vec[ind]-I_di[ii] )**2
        norm += abs( target_vec[ind] )**2
    
    error = error/norm
    
    return error


def chi_squared_error(target_data,actual_data):
    
    print('calculating chi^2 ...')
    dt1 = actual_data[0,1]-actual_data[0,0]
    error = 0
    for ii in range(len(actual_data[0,:])):
        ind = (np.abs(target_data[0,:]-actual_data[0,ii])).argmin()        
        error += np.abs( target_data[1,ind]-actual_data[1,ii] )**2
        
    dt2 = target_data[0,1]-target_data[0,0]
    norm = 0
    for ii in range(len(target_data[0,:])):
        norm += np.abs( target_data[1,ii] )**2    
    error = dt1*error/(dt2*norm)     
    # for ii in range(len(actual_data[0,0:-1])):
    #     dt = actual_data[0,ii+1]-actual_data[0,ii]
    #     ind = (np.abs(target_data[0,:]-actual_data[0,ii])).argmin()        
    #     error += dt*np.abs( target_data[1,ind]-actual_data[1,ii] )**2
    #     norm += dt*np.abs( target_data[1,ind] )**2    
    # error = error/norm    
    print('done calculating chi^2.')
    
    return error


def read_wr_data(file_path):
    
    print('reading wr data file ...')
    f = open(file_path, 'rt')
    
    file_lines = f.readlines()
    
    counter = 0
    for line in file_lines:
        counter += 1
        if line.find('No. Variables:') != -1:
            ind_start = line.find('No. Variables:')
            num_vars = int(line[ind_start+15:])
        if line.find('No. Points:') != -1:
            ind_start = line.find('No. Points:')
            num_pts = int(line[ind_start+11:])
        if str(line) == 'Variables:\n':            
            break    

    var_list = []
    for jj in range(num_vars):
        if jj <= 9:
            var_list.append(file_lines[counter+jj][3:-3]) 
        if jj > 9:
            var_list.append(file_lines[counter+jj][4:-3]) 

    data_mat = np.zeros([num_pts,num_vars])
    tn = counter+num_vars+1
    for ii in range(num_pts):
        # print('\n\nii = {}\n'.format(ii))
        for jj in range(num_vars):
            ind_start = file_lines[tn+jj].find('\t')
            # print('tn+jj = {}'.format(tn+jj))
            data_mat[ii,jj] = float(file_lines[tn+jj][ind_start+1:])
            # print('data_mat[ii,jj] = {}'.format(data_mat[ii,jj]))
        tn += num_vars
    
    f.close
    
    data_dict = dict()
    for ii in range(num_vars):
        data_dict[var_list[ii]] = data_mat[:,ii]
        
    print('done reading wr data file.')
    
    return data_dict

def omega_LRC(L,R,C):
    
    omega_r = np.sqrt( (L*C)**(-1) - 0.25*(R/L)**(2) )
    omega_i = R/(2*L)
    
    return omega_r, omega_i  

def load_neuron_data(load_string):
        
    with open('data/'+load_string, 'rb') as data_file:         
        neuron_imported = pickle.load(data_file)
    
    return neuron_imported
    
def save_session_data(data_array = [],save_string = 'soen_sim',include_time = True):
    
    if include_time == True:
        tt = time.time()     
        s_str = save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.dat'
    if include_time == False:
        s_str = save_string
    with open('soen_sim_data/'+s_str, 'wb') as data_file:
            pickle.dump(data_array, data_file)
            
    return

def load_session_data(load_string):
        
    with open('soen_sim_data/'+load_string, 'rb') as data_file:         
        data_array_imported = pickle.load(data_file)

    return data_array_imported

def t_fq(I,Ic,R,mu1,mu2):
    
    p = physical_constants()
    t_fq_vec = (p['Phi0']/(Ic*R))*((I/Ic)**mu1-1)**(-mu2)
    
    return t_fq_vec

def V_fq(I,Ic,R,mu1,mu2):
    
    V_fq_vec = (Ic*R)*((I/Ic)**mu1-1)**(mu2)
    
    return V_fq_vec

def V_fq__fit(I,mu1,mu2,V0):
    
    Ic = 40e-6
    R = 4.125
    
    V_fq_vec = (Ic*R)*((I/Ic)**mu1-1)**(mu2)+V0
    
    return V_fq_vec

def inter_fluxon_interval__fit(I,mu1,mu2,V0):
    
    Ic = 40e-6
    R = 4.125
    
    V_fq_vec = (Ic*R)*((I/Ic)**mu1-1)**(mu2)+V0
    p = physical_constants()
    ifi_vec = p['Phi0']/V_fq_vec
    
    return ifi_vec

def inter_fluxon_interval__fit_2(I_di,t0,I_fluxon,mu1,mu2,V0):
    
    Ic = 40e-6
    R = 4.125
    Lj2 = Ljj(Ic,Ic)
    Lj3 = Lj2
    L2 = 77.5e-12
    I0 = 35.2699e-6
    Phi0 = 2.06783375e-15

    t_fq = np.zeros([len(I_di)])
    for ii in range(len(I_di)):
        I_loop2_from_di = (Lj3/(L2+Lj2))*I_di[ii]
        if I0+I_fluxon+I_loop2_from_di-I_di[ii] > Ic:
            t_fq[ii] = t0 + Phi0 * ( (Ic*R)*(((I0+I_fluxon+I_loop2_from_di-I_di[ii])/Ic)**mu1-1)**(mu2)+V0 )**(-1)
        else:
            t_fq[ii] = 1e-6
    
    return t_fq

def inter_fluxon_interval__fit_3(I_di,I_bar_1,I_bar_2):
    
    Ic = 40e-6
    R = 4.125
    Phi0 = 2.06783375e-15
    V0 = 105e-6
    mu1 = 2.8
    mu2 = 0.5

    t_1 = Phi0/((Ic*R)*((I_bar_1/Ic)**mu1-1)**mu2+V0)
    t_2 = np.zeros([len(I_di)])
    for ii in range(len(I_di)):
        if I_bar_2-I_di[ii] > Ic:
            t_2[ii] = Phi0/((Ic*R)*(((I_bar_2-I_di[ii])/Ic)**mu1-1)**mu2+V0)
        else:
            t_2[ii] = 1
    t_fq = t_1+t_2
    
    return t_fq

def inter_fluxon_interval(I):
    
    V_fq_vec = (40e-6*4.125)*((I/40e-6)**2.839-1)**(0.501)+103.047e-6    
    ifi_vec = 2.06783375e-15/V_fq_vec
    
    return ifi_vec


def syn_1jj_rate_fit(I_sf,mu1,mu2,V0):
    
    Ic = 40
    rn = 4.125
    Phi0 = 1e6*1e6*2.06783375e-15
    print('I_sf = {}'.format(I_sf))
    rate = ( Ic*rn*( (I_sf/Ic)**mu1 - 1 )**mu2 + V0 )/Phi0
    print('rate = {}'.format(rate))
    # rate = np.real(rate)
    
    return rate


def syn_1jj_Vsf_vs_Isf_fit(I_sf,mu1,mu2,V0):
    
    Ic = 40
    # Rn = 4.125
    Ir = 1.1768
    
    V_sf = V0*( (I_sf/(Ic-Ir))**mu1 - 1 )**mu2
    # V_sf = V0*( I_sf**mu1 - (Ic-Ir)**mu1 )**(1/mu1)
    
    return V_sf

def syn_1jj_rate_vs_Isf(I_sf):
    
    r_sf = 1e3*( 233.966*( (I_sf/(38.8232))**3.464271 - 1 )**0.306768 )/2.06783375
    
    return r_sf


def syn_isolatedjj_voltage_fit(I_bias,V0,mu1,mu2,Ir):
    
    Ic = 40e-6
    V = V0*( ( I_bias/(Ic-Ir) )**mu1 - 1 )**mu2
    
    return V

def cv(start1,stop1,d1,start2,stop2,d2):
    
    vec = np.concatenate((np.arange(start1,stop1+d1,d1),np.arange(start2,stop2+d2,d2)))
    # vec = np.arange(start1,stop1+d1,d1)
    # vec = np.arange(start2,stop2+d2,d2)
    
    return vec


# def syn_isolatedjj_voltage_fit(I_bias,V0,mu,Ir):
    
#     Ic = 40
#     V = V0*( ( I_bias/(Ic-Ir) )**mu - 1 )**(1/mu)
    
#     return V
