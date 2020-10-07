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

def neuron_time_stepper(neuron_object):

    n = neuron_object
    print('\nsimulating neuron with {:d} synapse(s) and {:d} dendrite(s)\n'.format(len(n.synapses),len(n.dendrites)))
    
    t_init = time.time()

    # current_conversion = 1e6
    # inductance_conversion = 1e12
    # time_conversion = 1e6
    # resistance_conversion = inductance_conversion/time_conversion
    # voltage_conversion = current_conversion*resistance_conversion
    
    time_vec = n.time_vec
    nt = len(time_vec)
    dt = time_vec[1]-time_vec[0]
    n.dt = dt
    # print('dt = {}'.format(dt))
    
    p = physical_constants()
    Phi0 = p['Phi0'] * 1e18 # uA pH
               
    
    #parameters used to generate dendrite rate arrays
    dt_data = 1 # ps
    L_di_data = 77.5 # nH
    dsf_data = 200 # downsample factor
    
    #-------------------
    # load dendrite data
    #-------------------
    
    print('loading dendrite rate array ...')
    # DR loop inductances (in all cases, left side has additional 10pH through which flux is coupled (M = k*sqrt(L1*L2); in this case k = 1, L1 = 200pH, L2 = 10pH))
    dL = 3 # pH
    # L_left_list = np.arange(17,23+dL,dL) # pH
    # L_right_list = np.flip(np.arange(17,23+dL,dL)) # pH
    L_left_list = np.arange(20,20+dL,dL) # pH
    L_right_list = np.flip(np.arange(20,20+dL,dL)) # pH
    num_L = len(L_right_list)
    
    # dendritic firing junction bias current
    dI = 1 # uA
    I_de_list_2jj = np.arange(52,80+dI,dI) # uA
    I_de_list_4jj = np.arange(56,85+dI,dI) # uA
    
    # 'master__dnd_2jj__rate_array__Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq])
    # master__dnd_2jj__rate_array__Llft09.00_Lrgt21.00_Ide72.00.soen
    I_di_array__dict = dict()
    influx_list__dend__dict = dict()
    rate_array__dend__dict = dict()
    for pp in range(num_L):  
        for numjj in [4]: #[2,4]: 
            if numjj == 2:
                I_de_list = I_de_list_2jj
            elif numjj == 4:
                I_de_list = I_de_list_4jj
            num_I_de = len(I_de_list)
            for qq in range(num_I_de):
                
                _temp_str_1 = '../_circuit_data/master_dnd_rate_array_'
                _temp_str_2 = '{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}_Ldi{:07.2f}nH_dt{:04.1f}ps_dsf{:d}'.format(numjj,L_left_list[pp],L_right_list[pp],I_de_list[qq],L_di_data,dt_data,dsf_data)
            
                with open('{}{}.soen'.format(_temp_str_1,_temp_str_2), 'rb') as data_file:         
                    data_array_imported = pickle.load(data_file)
                
                I_di_array__dict[_temp_str_2] = data_array_imported['I_di_array']
                influx_list__dend__dict[_temp_str_2] = data_array_imported['influx_list']
                rate_array__dend__dict[_temp_str_2] = data_array_imported['rate_array']   
                
    #-----------------------
    # end load dendrite data
    #-----------------------
    
    #-----------------
    # load neuron data
    #-----------------
    
    print('loading neuron threshold array ...')
    # DR loop inductances (in all cases, left side has additional 10pH through which flux is coupled (M = k*sqrt(L1*L2); in this case k = 1, L1 = 200pH, L2 = 10pH))
    dL = 3 # pH

    L_left_list = np.arange(20,20+dL,dL) # pH
    L_right_list = np.flip(np.arange(20,20+dL,dL)) # pH
    num_L = len(L_right_list)
    
    # dendritic firing junction bias current
    dI = 1 # uA
    I_de_list_2jj = np.arange(52,80+dI,dI) # uA
    I_de_list_4jj = np.arange(56,85+dI,dI) # uA
    
    # 'master__dnd_2jj__rate_array__Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}'.format(L_left_list[pp],L_right_list[pp],I_de_list[qq])
    # master__dnd_2jj__rate_array__Llft09.00_Lrgt21.00_Ide72.00.soen
    I_di_th_array__neu__dict = dict()
    influx_array__neu__dict = dict()
    I_de_vec__neu__dict = dict()
    I_fq1_array__neu__dict = dict()
    I_fq2_array__neu__dict = dict()
    for pp in range(num_L):  
        for numjj in [4]: # [2,4]: 
            if numjj == 2:
                I_de_list = I_de_list_2jj
            elif numjj == 4:
                I_de_list = I_de_list_4jj
            num_I_de = len(I_de_list)
            for qq in range(num_I_de):
                
                _temp_str_1 = '../_circuit_data/master_neu_Idr_thr_'
                _temp_str_2 = '{:1d}jj_Llft{:04.1f}_Lrgt{:04.1f}_Lnf65.0'.format(numjj,L_left_list[pp],L_right_list[pp])
            
                with open('{}{}.soen'.format(_temp_str_1,_temp_str_2), 'rb') as data_file:         
                    data_array_imported = pickle.load(data_file)
                
                I_di_th_array__neu__dict[_temp_str_2] = data_array_imported['I_di_th_array']
                influx_array__neu__dict[_temp_str_2] = data_array_imported['Phi_dr_applied_array']
                I_de_vec__neu__dict[_temp_str_2] = data_array_imported['I_de_vec'] 
                I_fq1_array__neu__dict[_temp_str_2] = data_array_imported['I_fq1_array'] 
                I_fq2_array__neu__dict[_temp_str_2] = data_array_imported['I_fq2_array'] 
                
    #---------------------
    # end load neuron data
    #---------------------    
    
    
    #initialize all direct synapses
    print('configuring synapses ...')
    for sy_name in n.input_synaptic_connections:
        # print('sy_name = {}'.format(sy_name))
        
        #load all rate data to synapses 
        _temp_str_2 = '{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}_Ldi{:07.2f}nH_dt{:04.1f}ps_dsf{:d}'.format(n.synapses[sy_name].num_jjs,n.dendrites['{}__d'.format(sy_name)].L_left,n.dendrites['{}__d'.format(sy_name)].L_right,n.synapses[sy_name].bias_currents[0],L_di_data,dt_data,dsf_data)
        n.synapses[sy_name].I_di_array = I_di_array__dict[_temp_str_2]
        n.synapses[sy_name].influx_list__dend = influx_list__dend__dict[_temp_str_2]
        n.synapses[sy_name].rate_array__dend = rate_array__dend__dict[_temp_str_2]
                
        #change units to microamps and picohenries
        n.synapses[sy_name].I_fq = Phi0/n.synapses[sy_name].integration_loop_total_inductance
        n.synapses[sy_name].I_di_vec = np.zeros([nt])
        n.synapses[sy_name].I_spd2_vec = np.zeros([nt])
        n.synapses[sy_name].influx_vec = np.zeros([nt])
        n.synapses[sy_name].tau_di = n.synapses[sy_name].integration_loop_time_constant
        n.synapses[sy_name].M = n.dendrites['{}__d'.format(n.name)].input_synaptic_inductances[sy_name][1]*np.sqrt(n.synapses[sy_name].integration_loop_output_inductance*n.dendrites['{}__d'.format(n.name)].input_synaptic_inductances[sy_name][0])        
        n.synapses[sy_name].M_self = n.synapses[sy_name].M_self
        # self.synaptic_dendrite_input_synaptic_inductance[1]*np.sqrt(self.synaptic_dendrite_input_synaptic_inductance[0]*self.synaptic_circuit_inductors[2])
        
        n.synapses[sy_name].st_ind_last = 0
        n.synapses[sy_name].spd_current_memory = 0
                              
    #initialize all dendrites, the synapses on the dendrites, and the direct connections to the dendrites
    print('configuring dendrites ...')
    for de_name in n.input_dendritic_connections:
        # print('de_name = {}'.format(de_name))
        
        #load all rate data to dendrites
        # print('n.dendrites[de_name].num_jjs = {}'.format(n.dendrites[de_name].num_jjs))
        # print('inductance_conversion*n.dendrites[de_name].L_left = {}'.format(inductance_conversion*n.dendrites[de_name].L_left))
        # print('inductance_conversion*n.dendrites[de_name].L_right = {}'.format(inductance_conversion*n.dendrites[de_name].L_right))
        # print('current_conversion*n.dendrites[de_name].bias_currents[0] = {}'.format(current_conversion*n.dendrites[de_name].bias_currents[0]))
        _temp_str_2 = '{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}_Ldi{:07.2f}nH_dt{:04.1f}ps_dsf{:d}'.format(n.dendrites[de_name].num_jjs,n.dendrites[de_name].L_left,n.dendrites[de_name].L_right,n.dendrites[de_name].bias_currents[0],L_di_data,dt_data,dsf_data)
        n.dendrites[de_name].I_di_array = I_di_array__dict[_temp_str_2]
        n.dendrites[de_name].influx_list__dend = influx_list__dend__dict[_temp_str_2]
        n.dendrites[de_name].rate_array__dend = rate_array__dend__dict[_temp_str_2]
        
        #change units to microamps and picohenries
        n.dendrites[de_name].I_fq = Phi0/n.dendrites[de_name].integration_loop_total_inductance
        n.dendrites[de_name].I_di_vec = np.zeros([nt])
        n.dendrites[de_name].I_drive_vec = np.zeros([nt])
        n.dendrites[de_name].influx_vec = np.zeros([nt])
        n.dendrites[de_name].tau_di = n.dendrites[de_name].integration_loop_time_constant
        n.dendrites[de_name].M = n.dendrites['{}__d'.format(n.name)].input_dendritic_inductances[de_name][1]*np.sqrt(n.dendrites[de_name].integration_loop_output_inductance*n.dendrites['{}__d'.format(n.name)].input_dendritic_inductances[de_name][0])        
        
        # t1 = n.dendrites[de_name].input_dendritic_inductances['{}__d'.format(n.name)][1]
        # print('t1 = {}'.format(t1))
        # t2 = n.dendrites['{}__d'.format(n.name)].integration_loop_output_inductance
        # print('t2 = {}'.format(t2))
        # print('n.dendrites[de_name].M = {}'.format(n.dendrites[de_name].M))
        
        for de__dir_name in n.dendrites[de_name].input_direct_connections:
            # print('de__dir_name = {}'.format(de__dir_name))
                        
            #change units to microamps and picohenries
            n.dendrites[de_name].direct_connections[de__dir_name].M = n.dendrites[de_name].input_direct_inductances[de__dir_name][1]*np.sqrt(n.dendrites[de_name].input_direct_inductances[de__dir_name][0]*n.dendrites[de_name].direct_connections[de__dir_name].output_inductance)
            
        for de__sy_name in n.dendrites[de_name].input_synaptic_connections:
            # print('de__sy_name = {}'.format(de__sy_name))
            
            #load all rate data to synapses 
            _temp_str_2 = '{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}_Ldi{:07.2f}nH_dt{:04.1f}ps_dsf{:d}'.format(n.synapses[de__sy_name].num_jjs,n.synapses[de__sy_name].L_left,n.synapses[de__sy_name].L_right,n.synapses[de__sy_name].bias_currents[0],L_di_data,dt_data,dsf_data)
            n.synapses[de__sy_name].I_di_array = I_di_array__dict[_temp_str_2]
            n.synapses[de__sy_name].influx_list__dend = influx_list__dend__dict[_temp_str_2]
            n.synapses[de__sy_name].rate_array__dend = rate_array__dend__dict[_temp_str_2]
                    
            #change units to microamps and picohenries
            n.synapses[de__sy_name].I_fq = Phi0/n.synapses[de__sy_name].integration_loop_total_inductance
            n.synapses[de__sy_name].I_di_vec = np.zeros([nt])
            n.synapses[de__sy_name].I_spd2_vec = np.zeros([nt])
            n.synapses[de__sy_name].influx_vec = np.zeros([nt])
            n.synapses[de__sy_name].tau_di = n.synapses[de__sy_name].integration_loop_time_constant
            n.synapses[de__sy_name].M = n.dendrites['{}__d'.format(n.name)].input_synaptic_inductances[de__sy_name][1]*np.sqrt(n.synapses[de__sy_name].integration_loop_output_inductance*n.dendrites['{}__d'.format(n.name)].input_synaptic_inductances[de__sy_name][0])        
            
            n.synapses[de__sy_name].st_ind_last = 0
            n.synapses[de__sy_name].spd_current_memory = 0
            
        for de__de_name in n.dendrites[de_name].input_dendritic_connections:
            # print('de__de_name = {}'.format(de__de_name))
            
            #load all rate data to dendrites
            _temp_str_2 = '{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}'.format(n.dendrites[de__de_name].num_jjs,n.dendrites[de__de_name].L_left,n.dendrites[de__de_name].L_right,n.dendrites[de__de_name].bias_currents[0])
            n.dendrites[de__de_name].I_di_array = I_di_array__dict[_temp_str_2]
            n.dendrites[de__de_name].influx_list__dend = influx_list__dend__dict[_temp_str_2]
            n.dendrites[de__de_name].rate_array__dend = rate_array__dend__dict[_temp_str_2]
            
            n.dendrites[de__de_name].I_fq = Phi0/n.dendrites[de_name].integration_loop_total_inductance
            n.dendrites[de__de_name].I_di_vec = np.zeros([nt])
            n.dendrites[de__de_name].I_drive_vec = np.zeros([nt])
            n.dendrites[de__de_name].influx_vec = np.zeros([nt])
            n.dendrites[de__de_name].tau_di = n.dendrites[de_name].integration_loop_time_constant
            n.dendrites[de__de_name].M = n.dendrites[de_name].input_dendritic_inductances[de__de_name][1]*np.sqrt(n.dendrites[de__de_name].integration_loop_output_inductance*n.dendrites[de_name].input_dendritic_inductances[de__de_name][0])
            
    #initialize neuron
    print('configuring neuron ...')
    _temp_str_2 = '{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}_Ldi{:07.2f}nH_dt{:04.1f}ps_dsf{:d}'.format(n.dendrites['{}__d'.format(n.name)].num_jjs,n.dendrites['{}__d'.format(n.name)].L_left,n.dendrites['{}__d'.format(n.name)].L_right,n.dendrites['{}__d'.format(n.name)].bias_currents[0],L_di_data,dt_data,dsf_data)
    n.dendrites['{}__d'.format(n.name)].I_di_array = I_di_array__dict[_temp_str_2]
    n.dendrites['{}__d'.format(n.name)].influx_list__dend = influx_list__dend__dict[_temp_str_2]
    n.dendrites['{}__d'.format(n.name)].rate_array__dend = rate_array__dend__dict[_temp_str_2]
    
    n.dendrites['{}__d'.format(n.name)].I_fq = Phi0/n.dendrites['{}__d'.format(n.name)].integration_loop_total_inductance
    n.dendrites['{}__d'.format(n.name)].I_di_vec = np.zeros([nt])
    n.dendrites['{}__d'.format(n.name)].I_drive_vec = np.zeros([nt])
    n.dendrites['{}__d'.format(n.name)].influx_vec = np.zeros([nt])
    n.dendrites['{}__d'.format(n.name)].tau_di = n.dendrites['{}__d'.format(n.name)].integration_loop_time_constant
    # n.dendrites['{}__d'.format(n.name)].M = inductance_conversion*n.dendrites[de_name].input_dendritic_inductances[de__de_name][1]*np.sqrt(n.dendrites[de__de_name].integration_loop_output_inductance*n.dendrites[de_name].input_dendritic_inductances[de__de_name][0])
        
    # n.I_fq = current_conversion*Phi0/n.integration_loop_total_inductance
    # n.I_fq = current_conversion*Phi0/(n.integration_loop_total_inductance+Ljj(n.I_c*1e-6,n.bias_currents[2]))
    n.I_ni_vec = np.zeros([nt])
    n.I_drive_vec = np.zeros([nt])
    n.influx_vec = np.zeros([nt])
    n.influx_vec__no_refraction = np.zeros([nt])
    n.influx_vec__just_refraction = np.zeros([nt])
    n.tau_ni = n.integration_loop_time_constant
    
    # n.threshold_circuit = 'hTron'
    if n.threshold_circuit == 'JJ':
        n.state = 'subthreshold'
        n.state_next = 'subthreshold'
        n.threshold_I1 = np.zeros([nt])
        n.threshold_I2 = np.zeros([nt]) 
        n.threshold_Lt = n.threshold_circuit_total_inductance # inductance_conversion
        n.threshold_r = n.threshold_circuit_resistance # *resistance_conversion
        n.threshold_Ibias = n.threshold_circuit_bias_current # current_conversion
        n.threshold_M = -n.integration_loop_output_inductance[1]*np.sqrt(n.integration_loop_output_inductance[0]*n.threshold_circuit_inductances[0]) # *inductance_conversion
        n.refraction_M = n.neuronal_receiving_input_refractory_inductance[1]*np.sqrt(n.neuronal_receiving_input_refractory_inductance[0]*n.threshold_circuit_inductances[2])
    elif n.threshold_circuit == 'hTron':
        n.state = 'off'
        n.threshold_I_1 = np.zeros([nt])
        n.threshold_I_3 = np.zeros([nt])#np.ones([nt])*n.threshold_Ib
        n.threshold_M = n.integration_loop_output_inductance[1]*np.sqrt(n.integration_loop_output_inductance[0]*n.threshold_circuit_inductances[0]) # *inductance_conversion
        n.refraction_M = n.neuronal_receiving_input_refractory_inductance[1]*np.sqrt(n.neuronal_receiving_input_refractory_inductance[0]*n.threshold_circuit_inductances[4])
    
    
    # n.L_nr = n.L_nr # *inductance_conversion
    # n.threshold_integration = 0
    n.delta_I__spike = 53.5 # (51.315+3.5)
    n.spike_latch_time = 0.43

    n.spike_times = []
    n.spike_time_indices = []
    
    #------------------
    # step through time
    #------------------
    print('starting time stepping ...')
    
    Inr1_prev = 0
    Inr2_prev = 0
    Idr1_prev = 0
    Idr2_prev = 0
    Ij2_prev = 0
    Ij3_prev = 0
    
    ii_vec = np.arange(1,nt-1,1)
    for ii in ii_vec:
        
        _pt = time_vec[ii] # present time 
        
        #----------------------
        # step through synapses
        #----------------------
        for sy_name in n.input_synaptic_connections:
            
            
            spike_times = n.synapses[sy_name].input_signal.spike_times
    
            if len(spike_times) > 0:
                               
                # find most recent spike time 
                n.synapses[sy_name].st_ind = (np.abs(spike_times[:] - _pt)).argmin()
                if n.synapses[sy_name].st_ind == 0 and spike_times[n.synapses[sy_name].st_ind] > _pt: # first spike has not arrived
                    _gf = 0 # growth factor
                    # print('code 1: st_ind == 0 and spike_times[st_ind] > _pt')
                if n.synapses[sy_name].st_ind > 0 and spike_times[n.synapses[sy_name].st_ind] > _pt: # go back to previous spike
                    n.synapses[sy_name].st_ind -= 1
                    # print('code 2: st_ind > 0 and spike_times[st_ind] > _pt')
                if _pt - spike_times[n.synapses[sy_name].st_ind] > n.synapses[sy_name].spd_duration: # outside SPD pulse range
                    _gf = 0 # growth factor
                    # print('code 3: _pt - spike_times[st_ind] > spd_duration')
                if spike_times[n.synapses[sy_name].st_ind] <= _pt and (_pt - spike_times[n.synapses[sy_name].st_ind]) < n.synapses[sy_name].spd_duration: # the case that counts    
                    # print('code 4')
                    
                    n.synapses[sy_name].dt_spk = _pt - spike_times[n.synapses[sy_name].st_ind]                    
                    
                    #this block to avoid spd drive going too low at the onset of each spike                    
                    tn = spd_response(n.synapses[sy_name].I_spd,n.synapses[sy_name].r1,n.synapses[sy_name].r2,n.synapses[sy_name].t0,n.synapses[sy_name].tau_plus,n.synapses[sy_name].tau_minus,n.synapses[sy_name].dt_spk)
                    if n.synapses[sy_name].st_ind - n.synapses[sy_name].st_ind_last == 1:                        
                        spd_current = np.max([n.synapses[sy_name].I_spd2_vec[ii-1],tn])
                        n.synapses[sy_name].spd_current_memory = spd_current
                    if n.synapses[sy_name].spd_current_memory > 0 and tn < n.synapses[sy_name].spd_current_memory:
                        spd_current = n.synapses[sy_name].spd_current_memory
                    else:
                        spd_current = tn
                        n.synapses[sy_name].spd_current_memory = 0
                    n.synapses[sy_name].I_spd2_vec[ii] = spd_current
                    
                    n.synapses[sy_name].st_ind_last = n.synapses[sy_name].st_ind
                        
                    n.synapses[sy_name].influx_vec[ii] = n.synapses[sy_name].I_spd2_vec[ii]*n.synapses[sy_name].M_self
                    ind1 = (np.abs(n.synapses[sy_name].influx_list__dend-n.synapses[sy_name].influx_vec[ii])).argmin()                             
                    
                    synapse_interpolation = True
                    if synapse_interpolation == False:
                        
                        # no interpolation 
                        ind2 = (np.abs(n.synapses[sy_name].I_di_array[ind1]-n.synapses[sy_name].I_di_vec[ii])).argmin()
                        rate = n.synapses[sy_name].rate_array__dend[ind1][ind2] 
                        _gf = rate*n.synapses[sy_name].I_fq*dt # growth factor 
                
                    elif synapse_interpolation == True:
                        
                        # interpolation
                        influx_actual = n.synapses[sy_name].influx_vec[ii]
                        influx_closest = n.synapses[sy_name].influx_list__dend[ind1]
                        if influx_actual > influx_closest:
                            if ind1 < len(n.synapses[sy_name].influx_list__dend)-1:
                                ind1_p = ind1+1                        
                            else:
                                ind1_p = ind1
                            ind2 = (np.abs(n.synapses[sy_name].I_di_array[ind1]-n.synapses[sy_name].I_di_vec[ii])).argmin()
                            ind2_p = (np.abs(n.synapses[sy_name].I_di_array[ind1_p]-n.synapses[sy_name].I_di_vec[ii])).argmin()
                            rate1 = n.synapses[sy_name].rate_array__dend[ind1][ind2]
                            rate2 = n.synapses[sy_name].rate_array__dend[ind1_p][ind2_p]
                            influx_next_closest = n.synapses[sy_name].influx_list__dend[ind1_p]
                            x = influx_actual - influx_closest
                            if influx_next_closest != influx_closest:
                                m = (rate2-rate1)/(influx_next_closest-influx_closest)
                            else:
                                m = 0
                            rate = m*x+rate1
                        elif influx_actual < influx_closest:
                            if ind1 > 0:
                                ind1_p = ind1-1
                            else:
                                ind1_p = ind1
                            ind2 = (np.abs(n.synapses[sy_name].I_di_array[ind1]-n.synapses[sy_name].I_di_vec[ii])).argmin()
                            ind2_p = (np.abs(n.synapses[sy_name].I_di_array[ind1_p]-n.synapses[sy_name].I_di_vec[ii])).argmin()
                            rate1 = n.synapses[sy_name].rate_array__dend[ind1][ind2]
                            rate2 = n.synapses[sy_name].rate_array__dend[ind1_p][ind2_p]
                            influx_next_closest = n.synapses[sy_name].influx_list__dend[ind1_p]
                            x = influx_closest - influx_actual
                            if influx_next_closest != influx_closest:
                                m = (rate1-rate2)/(influx_closest-influx_next_closest)
                            else:
                                m = 0
                            rate = -m*x+rate1
                        elif influx_actual == influx_closest:
                            ind2 = (np.abs(n.synapses[sy_name].I_di_array[ind1]-n.synapses[sy_name].I_di_vec[ii])).argmin()
                            rate = n.synapses[sy_name].rate_array__dend[ind1][ind2] 
                        _gf = rate*n.synapses[sy_name].I_fq*dt # + (1-dt/n.dendrites[de_name].tau_di)*n.dendrites[de_name].I_di_vec[ii]                        
                    
                n.synapses[sy_name].I_di_vec[ii+1] = _gf + (1-dt/n.synapses[sy_name].tau_di)*n.synapses[sy_name].I_di_vec[ii]
        
        #-----------------------       
        # step through dendrites
        #-----------------------       
        for de_name in n.input_dendritic_connections:
            
            # calculate flux drive from direct inputs
            dir_flux = 0
            for de__dir_name in n.dendrites[de_name].input_direct_connections:
                # print('here1')
                dir_flux += n.dendrites[de_name].direct_connections[de__dir_name].M*n.dendrites[de_name].direct_connections[de__dir_name].drive_signal[ii+1]
            
            # calculate flux drive from synapses
            syn_flux = 0
            for de__sy_name in n.dendrites[de_name].input_synaptic_connections:
                if n.synapses[de__sy_name].inhibitory_or_excitatory == 'inhibitory':
                    _prefactor = -1
                elif n.synapses[de__sy_name].inhibitory_or_excitatory == 'excitatory':
                    _prefactor = 1
                syn_flux += _prefactor*n.synapses[de__sy_name].M*n.synapses[de__sy_name].I_di_vec[ii+1]
                
            # calculate flux drive from other dendrites
            dend_flux = 0
            for de__de_name in n.dendrites[de_name].input_dendritic_connections:
                # if ii == 0:
                # print('de__de_name = {}'.format(de__de_name))
                if n.dendrites[de__de_name].inhibitory_or_excitatory == 'inhibitory':
                    _prefactor = -1
                elif n.dendrites[de__de_name].inhibitory_or_excitatory == 'excitatory':
                    # print('here2: de__de_name = {}'.format(de__de_name))
                    _prefactor = 1  
                # print('_prefactor = {}'.format(_prefactor))
                # print('n.dendrites[de__de_name].M = {}'.format(n.dendrites[de__de_name].M))
                # print('_prefactor = {}'.format(_prefactor))
                dend_flux += _prefactor*n.dendrites[de__de_name].M*n.dendrites[de__de_name].I_di_vec[ii] 
                
            
            # total drive to this dendrite
            # print('syn_flux = {}; dend_flux = {}; dir_flux = {}'.format(syn_flux,dend_flux,dir_flux))
            n.dendrites[de_name].influx_vec[ii] = syn_flux+dend_flux+dir_flux
            ind1 = (np.abs(n.dendrites[de_name].influx_list__dend-n.dendrites[de_name].influx_vec[ii])).argmin()                             
    
            dendrite_interpolation = True
            if dendrite_interpolation == False:
                
                # no interpolation 
                ind2 = (np.abs(n.dendrites[de_name].I_di_array[ind1]-n.dendrites[de_name].I_di_vec[ii])).argmin()
                rate = n.dendrites[de_name].rate_array__dend[ind1][ind2] 
                # if rate > 0:
                #     print('rate = {}'.format(rate))
                n.dendrites[de_name].I_di_vec[ii+1] = rate*n.dendrites[de_name].I_fq*dt + (1-dt/n.dendrites[de_name].tau_di)*n.dendrites[de_name].I_di_vec[ii]  
        
            elif dendrite_interpolation == True:
                
                # interpolation
                influx_actual = n.dendrites[de_name].influx_vec[ii]
                influx_closest = n.dendrites[de_name].influx_list__dend[ind1]
                if influx_actual > influx_closest:
                    if ind1 < len(n.dendrites[de_name].influx_list__dend)-1:
                        ind1_p = ind1+1                        
                    else:
                        ind1_p = ind1
                    ind2 = (np.abs(n.dendrites[de_name].I_di_array[ind1]-n.dendrites[de_name].I_di_vec[ii])).argmin()
                    ind2_p = (np.abs(n.dendrites[de_name].I_di_array[ind1_p]-n.dendrites[de_name].I_di_vec[ii])).argmin()
                    rate1 = n.dendrites[de_name].rate_array__dend[ind1][ind2]
                    rate2 = n.dendrites[de_name].rate_array__dend[ind1_p][ind2_p]
                    influx_next_closest = n.dendrites[de_name].influx_list__dend[ind1_p]
                    x = influx_actual - influx_closest
                    if influx_next_closest != influx_closest:
                        m = (rate2-rate1)/(influx_next_closest-influx_closest)
                    else:
                        m = 0
                    rate = m*x+rate1
                elif influx_actual < influx_closest:
                    if ind1 > 0:
                        ind1_p = ind1-1
                    else:
                        ind1_p = ind1
                    ind2 = (np.abs(n.dendrites[de_name].I_di_array[ind1]-n.dendrites[de_name].I_di_vec[ii])).argmin()
                    ind2_p = (np.abs(n.dendrites[de_name].I_di_array[ind1_p]-n.dendrites[de_name].I_di_vec[ii])).argmin()
                    rate1 = n.dendrites[de_name].rate_array__dend[ind1][ind2]
                    rate2 = n.dendrites[de_name].rate_array__dend[ind1_p][ind2_p]
                    influx_next_closest = n.dendrites[de_name].influx_list__dend[ind1_p]
                    x = influx_closest - influx_actual
                    if influx_next_closest != influx_closest:
                        m = (rate1-rate2)/(influx_closest-influx_next_closest)
                    else:
                        m = 0
                    rate = -m*x+rate1
                elif influx_actual == influx_closest:
                    ind2 = (np.abs(n.dendrites[de_name].I_di_array[ind1]-n.dendrites[de_name].I_di_vec[ii])).argmin()
                    rate = n.dendrites[de_name].rate_array__dend[ind1][ind2]

                _gf = rate*n.dendrites[de_name].I_fq*dt                
                            
                n.dendrites[de_name].I_di_vec[ii+1] = _gf + (1-dt/n.dendrites[de_name].tau_di)*n.dendrites[de_name].I_di_vec[ii]
                                            

        #------------------
        # the neuron itself
        #------------------        
                
        syn_flux = 0
        syn_flux_2 = 0
        for sy_name in n.input_synaptic_connections:
            if n.synapses[sy_name].inhibitory_or_excitatory == 'inhibitory':
                _prefactor = -1
            elif n.synapses[sy_name].inhibitory_or_excitatory == 'excitatory':
                _prefactor = 1
            syn_flux += _prefactor*n.synapses[sy_name].M*n.synapses[sy_name].I_di_vec[ii+1] 
            syn_flux_2 += _prefactor*( n.synapses[sy_name].M )*( (n.synapses[sy_name].I_di_vec[ii+1]-n.synapses[sy_name].I_di_vec[ii]) )
        dend_flux = 0
        for de_name in n.input_dendritic_connections:
            if n.dendrites[de_name].inhibitory_or_excitatory == 'inhibitory':
                _prefactor = -1
            elif n.dendrites[de_name].inhibitory_or_excitatory == 'excitatory':
                _prefactor = 1                    
            dend_flux += _prefactor*n.dendrites[de_name].M*n.dendrites[de_name].I_di_vec[ii+1]
            # if de_name == '{}__r'.format(n.name):
            #     dend_flux__ref = _prefactor*n.dendrites[de_name].M*n.dendrites[de_name].I_di_vec[ii+1]
                
        if n.threshold_circuit == 'JJ':
        
            n.threshold_Ltt = n.threshold_Lt+Ljj_pH(n.threshold_junction_critical_current,n.threshold_I1[ii])
            n.L_nr = n.L_nr_0+Ljj_pH(n.I_c,n.I_c)+Ljj_pH(n.I_c,n.I_c)

            tn1 = n.dendrites['{}__d'.format(n.name)].L_right+Ljj_pH(n.junction_critical_current,n.junction_critical_current)
            tn2 = n.circuit_inductances[2]
            alt_branch_factor = tn2/(tn1+tn2)
            
            # print(n.state)
            if n.state == 'subthreshold':
                # n.V_j = 0
                n.threshold_I2[ii+1] = ( (1-n.threshold_r*dt/n.threshold_Ltt)*n.threshold_I2[ii] 
                                     - alt_branch_factor*( n.synapses[sy_name].M*n.refraction_M/(n.L_nr*n.threshold_Ltt) )*( n.synapses[sy_name].I_di_vec[ii+1]-n.synapses[sy_name].I_di_vec[ii] )
                                    + ( n.threshold_M/n.threshold_Ltt )*( n.I_ni_vec[ii] - n.I_ni_vec[ii-1] ) )
                                    
                n.threshold_I1[ii+1] = n.threshold_Ibias-n.threshold_I2[ii+1]
                if n.threshold_I1[ii+1] >= n.threshold_junction_critical_current:
                    # print('into pre')
                    n.state_next = 'prespiking'
                    n.spike_onset_time = _pt
                    
            elif n.state == 'spiking':
                # print('spiking')
                # n.V_j = p['Phi0'] # Phi0 # 
                n.threshold_I2[ii+1] = n.delta_I__spike + n.threshold_I2[ii]
                # n.threshold_I1[ii+1] = n.threshold_Ibias-n.threshold_I2[ii+1]
                n.state_next = 'subthreshold'
                n.spike_times.append(_pt+dt)
                n.spike_time_indices.append(ii+1)  
                                    
            elif n.state == 'prespiking':
                # print('prespiking; _pt = {}; ot = {}'.format(_pt,(n.spike_onset_time+n.spike_latch_time)))
                n.threshold_I1[ii+1] = n.threshold_I1[ii] # n.threshold_junction_critical_current-0.01
                n.threshold_I2[ii+1] = n.threshold_I2[ii]
                if _pt >= (n.spike_onset_time+n.spike_latch_time):
                    # print('out of pre')
                    n.state_next = 'spiking'   

            n.state = n.state_next                                                 
               
            #refractory flux
            ref_flux = -n.threshold_I2[ii+1]*n.refraction_M 
            
        elif n.threshold_circuit == 'hTron':
            
            I_gate = n.threshold_I_1[ii]
            I_channel = n.threshold_Ib - n.threshold_I_3[ii]
            
            # print('ii = {}; I_gate = {}uA, I_channel = {}uA'.format(ii,I_gate,I_channel))
            # time.sleep(0.1)
            
            if n.state == 'off':
                n.threshold_I_1[ii+1] = (1-dt*(n.r1/n.La)) * n.threshold_I_1[ii] + (n.threshold_M/n.La)*( n.I_ni_vec[ii]-n.I_ni_vec[ii-1] )
                n.threshold_I_3[ii+1] = (1-dt*(n.r4/n.Lb))*n.threshold_I_3[ii]
            elif n.state == 'on':
                r2 = n.r_gate
                r3 = n.r_channel
                ra = n.r1+r2
                rb = r3+n.r4 
                n.threshold_I_1[ii+1] = np.max( [(1-dt*(ra/n.La)),0] ) * n.threshold_I_1[ii] + (n.threshold_M/n.La)*( n.I_ni_vec[ii]-n.I_ni_vec[ii-1] )
                n.threshold_I_3[ii+1] = n.threshold_Ib # (1-dt*(rb/n.Lb))*n.threshold_I_3[ii] + np.min( [dt*(r3/n.Lb)*n.threshold_Ib,n.threshold_Ib] ) # dt*(r3/n.Lb)*n.threshold_Ib #             
          
            #refractory flux
            ref_flux = -n.threshold_I_3[ii+1]*n.refraction_M
                
            if n.state == 'off':
                if I_gate >= n.threshold_I_gate_on and I_channel > n.threshold_I_channel_on:
                    n.state = 'on'
                    n.spike_times.append(_pt+dt)
                    n.spike_time_indices.append(ii+1)
                    # print('ii = {} (t = {:7.0}ns); state = on from off'.format(ii,_pt))            
            elif n.state == 'on':
                n.state = 'off'
                # if I_gate < n.threshold_I_gate_off and I_channel < n.threshold_I_channel_off:
                    # n.state = 'off'

            
            
        
        # total flux drive to neuron        
        n.influx_vec[ii] = syn_flux + ref_flux # + dend_flux 
        n.influx_vec__no_refraction[ii] = syn_flux # + dend_flux
        n.influx_vec__just_refraction[ii] = ref_flux
        
        ind1 = (np.abs(n.dendrites['{}__d'.format(n.name)].influx_list__dend-n.influx_vec[ii])).argmin() 
        
        neuron_interpolation = True
        if neuron_interpolation == False:
            
            # no interpolation 
            ind2 = (np.abs(n.dendrites['{}__d'.format(n.name)].I_di_array[ind1]-n.dendrites['{}__d'.format(n.name)].I_di_vec[ii])).argmin()
            rate = n.dendrites['{}__d'.format(n.name)].rate_array__dend[ind1][ind2] 
            # if rate > 0:
            #     print('rate = {}'.format(rate))
            n.dendrites['{}__d'.format(n.name)].I_di_vec[ii+1] = rate*n.dendrites['{}__d'.format(n.name)].I_fq*dt + (1-dt/n.dendrites['{}__d'.format(n.name)].tau_di)*n.dendrites['{}__d'.format(n.name)].I_di_vec[ii]  
    
        elif neuron_interpolation == True:
            
            # interpolation
            influx_actual = n.influx_vec[ii]
            influx_closest = n.dendrites['{}__d'.format(n.name)].influx_list__dend[ind1]
            if influx_actual > influx_closest:
                if ind1 < len(n.dendrites['{}__d'.format(n.name)].influx_list__dend)-1:
                    ind1_p = ind1+1                        
                else:
                    ind1_p = ind1
                ind2 = (np.abs(n.dendrites['{}__d'.format(n.name)].I_di_array[ind1]-n.dendrites['{}__d'.format(n.name)].I_di_vec[ii])).argmin()
                ind2_p = (np.abs(n.dendrites['{}__d'.format(n.name)].I_di_array[ind1_p]-n.dendrites['{}__d'.format(n.name)].I_di_vec[ii])).argmin()
                rate1 = n.dendrites['{}__d'.format(n.name)].rate_array__dend[ind1][ind2]
                rate2 = n.dendrites['{}__d'.format(n.name)].rate_array__dend[ind1_p][ind2_p]
                influx_next_closest = n.dendrites['{}__d'.format(n.name)].influx_list__dend[ind1_p]
                x = influx_actual - influx_closest
                if influx_next_closest != influx_closest:
                    m = (rate2-rate1)/(influx_next_closest-influx_closest)
                else:
                    m = 0
                rate = m*x+rate1
            elif influx_actual < influx_closest:
                if ind1 > 0:
                    ind1_p = ind1-1
                else:
                    ind1_p = ind1
                ind2 = (np.abs(n.dendrites['{}__d'.format(n.name)].I_di_array[ind1]-n.dendrites['{}__d'.format(n.name)].I_di_vec[ii])).argmin()
                ind2_p = (np.abs(n.dendrites['{}__d'.format(n.name)].I_di_array[ind1_p]-n.dendrites['{}__d'.format(n.name)].I_di_vec[ii])).argmin()
                rate1 = n.dendrites['{}__d'.format(n.name)].rate_array__dend[ind1][ind2]
                rate2 = n.dendrites['{}__d'.format(n.name)].rate_array__dend[ind1_p][ind2_p]
                influx_next_closest = n.dendrites['{}__d'.format(n.name)].influx_list__dend[ind1_p]
                x = influx_closest - influx_actual
                if influx_next_closest != influx_closest:
                    m = (rate1-rate2)/(influx_closest-influx_next_closest)
                else:
                    m = 0
                rate = -m*x+rate1
            elif influx_actual == influx_closest:
                ind2 = (np.abs(n.dendrites['{}__d'.format(n.name)].I_di_array[ind1]-n.dendrites['{}__d'.format(n.name)].I_di_vec[ii])).argmin()
                rate = n.dendrites['{}__d'.format(n.name)].rate_array__dend[ind1][ind2]

            _gf = rate*n.dendrites['{}__d'.format(n.name)].I_fq*dt                
                        
            n.dendrites['{}__d'.format(n.name)].I_di_vec[ii+1] = _gf + (1-dt/n.dendrites['{}__d'.format(n.name)].tau_di)*n.dendrites['{}__d'.format(n.name)].I_di_vec[ii]
        
        n.I_ni_vec[ii+1] = n.dendrites['{}__d'.format(n.name)].I_di_vec[ii+1]
                                 

    print('\ndone running neuron simulation. total time was {:.3}s\n'.format(time.time()-t_init))        
    
    return n #I_di_vec


def spd_response(I_spd,r1,r2,t0,tau_plus,tau_minus,t):
    
    if t <= t0:
        I = I_spd*(r1/(r1+r2))*(1-np.exp(-t/tau_plus))
    elif t > t0:
        I = I_spd*(r1/(r1+r2))*(1-np.exp(-t0/tau_plus))*np.exp(-(t-t0)/tau_minus)
    
    return I

def dendritic_drive__piecewise_linear(time_vec,pwl):
    
    input_signal__dd = np.zeros([len(time_vec)])
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
    input_signal__dd[t2_ind:] = pwl[-1][1]*np.ones([len(time_vec)-t2_ind])
    
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
    
    input_signal__dd = np.zeros([len(time_vec)])
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
    
    input_signal__dd = np.zeros([len(time_vec)])
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
    
    input_signal__dd = np.zeros([len(time_vec)])
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


def Vj_of_Ij(Ij):
    
    # print('Isf = {}'.format(Isf))
    # return 235.19243476368464*( (Isf/38.81773424470013e-6)**3.4193613971219454 - 1 )**0.3083945546392435 # unit of uV
    return 1e-6*236.878860808991*( (Ij/38.80656547520097e-6)**3.3589340685815574 - 1 )**0.310721713450461 # unit of V


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


def chi_squared_error__ISI(target_data,actual_data):
    
    print('calculating chi^2 ISI ...')
    
    tn = np.min([len(target_data),len(actual_data)])
    if tn > 1:
        
        Dt_ISI__wr = np.diff(target_data)
        Dt_ISI__soen = np.diff(actual_data)
        error = 0
        norm = 0
    
        for ii in range(tn-1):        
            error += np.abs( Dt_ISI__wr[ii]-Dt_ISI__soen[ii] )**2
            norm += np.abs( Dt_ISI__wr[ii] )**2
       
        error = error/norm
        
    elif tn == 1:
        
        error = np.abs( target_data[0]-actual_data[0] )**2/50**2 # denominator is refractory time constant in ns
    
    print('done calculating chi^2 ISI.')
    
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

def Vj(I,Ic,R):
    
    if I >= Ic:        
        V = Ic*R*np.sqrt( (I/Ic)**2 - 1 )
        print('V = {}uV'.format(V*1e6))
    else: 
        V = 0
    
    return V

def Ljj(critical_current,current):
    
    norm_current = np.max([np.min([current/critical_current,1]),1e-9])
    L = (3.2910596281416393e-16/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

def Ljj_pH(critical_current,current):
    
    norm_current = np.max([np.min([current/critical_current,1]),1e-9])
    L = (3.2910596281416393e2/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

def low_pass_filter(y,t,r,L1,L2,C,M,dIdrive_dt):
    
    I1, I2, I4 = y
    dydt = [ (L2/L1)*I4+(r/L1)*I2-M*dIdrive_dt, I4, -(r/L2)*I4-(1/(L2*C))*(I2+I1)]
    
    return dydt

def nTron(I_gate,I_on,I_off,r_gate,r_channel,state):
    
    if I_gate >= I_on:
        state = 'on'
    elif I_gate < I_on and I_gate >= I_off:
        if state == 'on':
            state = 'latched'
    elif I_gate < I_off:
        if state == 'on' or state == 'latched':
            state = 'off'
    
    if state == 'on' or state == 'latched':
        r1 = r_gate
        r2 = r_channel
    elif state == 'off':
        r1 = 0
        r2 = 0
            
    return r1, r2

def ind_in_par(L1,L2):
    
    return (L1*L2/(L1+L2))