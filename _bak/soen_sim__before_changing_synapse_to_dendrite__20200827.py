import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import time
import pickle
import copy
from scipy.signal import find_peaks

from _functions import neuron_time_stepper, synapse_time_stepper__Isf_ode__spd_delta, synapse_time_stepper__2jj__ode, synapse_time_stepper, synapse_time_stepper__1jj_ode, synapse_time_stepper__Isf_ode__spd_jj_test, dendritic_drive__piecewise_linear, Ljj, dendritic_drive__square_pulse_train, dendritic_drive__exp_pls_train__LR, dendrite_time_stepper
from _plotting import plot_dendritic_drive, plot_dendritic_integration_loop_current
from util import physical_constants

class input_signal():
    
    _next_uid = 0
    input_signals = dict()
    
    def __init__(self, *args, **kwargs):
        
        #make new input signal
        self.uid = input_signal._next_uid
        input_signal._next_uid += 1
        self.unique_label = 'in'+str(self.uid)
        
        if len(args) > 0:
            if type(args[0]) == str:
                _name = args[0]
            elif type(args[0]) == int or type(args[0]) == float:
                _name = str(args[0])
        else:
            _name = 'unnamed_input_signal'
        self.name = _name
        
        if 'input_temporal_form' in kwargs:
            if (kwargs['input_temporal_form'] == 'single_spike' or 
                kwargs['input_temporal_form'] == 'constant_rate' or 
                kwargs['input_temporal_form'] == 'arbitrary_spike_train' or
                kwargs['input_temporal_form'] == 'analog_dendritic_drive'):
                _temporal_form = kwargs['input_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid input signal temporal form to input %s (unique_label = %s)\nThe allowed values of input_temporal_form are ''single_spike'', ''constant_rate'', ''arbitrary_spike_train'', and ''analog_dendritic_drive''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'single_spike'
        self.input_temporal_form =  _temporal_form #'single_spike' by default
        
        if 'spike_times' in kwargs:
            if (self.input_temporal_form == 'single_spike' or self.input_temporal_form == 'arbitrary_spike_train'):
                self.spike_times = kwargs['spike_times']
            elif self.input_temporal_form == 'constant_rate': # in this case, spike_times has the form [rate,time_of_first_spike,time_of_last_spike]
                isi = 1/kwargs['spike_times'][0]
                self.spike_times = np.arange(kwargs['spike_times'][1],kwargs['spike_times'][2]+isi,isi)
                
        if 'stochasticity' in kwargs:
            if kwargs['stochasticity'] == 'gaussian' or kwargs['stochasticity'] == 'none':
                self.stochasticity = kwargs['stochasticity']
            else: 
                raise ValueError('[soens_sim] Tried to assign an invalid stochasticity type to input {} (unique label {}). Available stochasticity forms are presently: ''gaussian'' or ''none''' % (self.name, self.unique_label))
        else:
            self.stochasticity = 'none'
        
        if 'jitter_params' in kwargs:
            if self.stochasticity == 'gaussian':
                self.jitter_params = kwargs['jitter_params'] #[center of gaussian, width of gaussian (standard deviation)]
                if len(self.jitter_params) == 2:
                    for ii in range(len(self.spike_times)):
                        self.spike_times[ii] += np.random.normal(self.jitter_params[0],self.jitter_params[1],1)
                else:
                    raise ValueError('[soens_sim] With gaussian stochasticity, jitter_params must be a two-element list of the form: [center of gaussian, width of gaussian (standard deviation)]')
        
        if self.input_temporal_form == 'analog_dendritic_drive':
            if 'output_inductance' in kwargs:
                self.output_inductance = kwargs['output_inductance']
            else: 
                self.output_inductance = 200e-12
            if 'time_vec' in kwargs:
                self.time_vec = kwargs['time_vec']
            else:
                dt = 5e-9
                tf = 20e-9
                self.time_vec = np.arange(0,tf+dt,dt)
            if 'piecewise_linear' in kwargs:
                self.piecewise_linear = kwargs['piecewise_linear']
            if 'square_pulse_train' in kwargs:
                self.square_pulse_train = kwargs['square_pulse_train'] 
            if 'exponential' in kwargs:
                self.exp_params = kwargs['exponential']
            if 'exponential_pulse_train' in kwargs:
                self.exponential_pulse_train = kwargs['exponential_pulse_train']    
            
        input_signal.input_signals[self.name] = self
            
class synapse():    

    _next_uid = 0
    synapses = dict()
    
    def __init__(self, *args, **kwargs):

        #make new synapse
        # self._instances.add(weakref.ref(self))
        self.uid = synapse._next_uid
        synapse._next_uid += 1
        self.unique_label = 's'+str(self.uid)

        if len(args) > 0:
            if type(args[0]) == str:
                _name = args[0]
            elif type(args[0]) == int or type(args[0]) == float:
                _name = str(args[0])
        else:
            _name = 'unnamed_synapse'
        self.name = _name
        
        if 'num_jjs' in kwargs:
            if 1 <= kwargs['num_jjs'] and kwargs['num_jjs'] <= 3:
                self.num_jjs = kwargs['num_jjs']
            else:
                raise ValueError('[soens_sim] num_jjs must be 1, 2, or 3')
        else:
            self.num_jjs = 3

        if 'inhibitory_or_excitatory' in kwargs:
            if kwargs['inhibitory_or_excitatory'] == 'inhibitory' or kwargs['inhibitory_or_excitatory'] == 'excitatory':
                _i_or_e = kwargs['inhibitory_or_excitatory']
            else:
                raise ValueError('[soens_sim] inhibitory_or_excitatory can either be ''inhibitory'' and ''excitatory''')
        else:
            _i_or_e = 'excitatory'
        self.inhibitory_or_excitatory =  _i_or_e #'excitatory' by default
        
        if 'integration_loop_temporal_form' in kwargs:
            if kwargs['integration_loop_temporal_form'] == 'exponential' or kwargs['integration_loop_temporal_form'] == 'power_law':
                _temporal_form = kwargs['integration_loop_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid integration loop temporal form to synapse %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'exponential'
        self.integration_loop_temporal_form =  _temporal_form #'exponential' or 'power_law'; 'exponential' by default

        if 'integration_loop_time_constant' in kwargs:
            if kwargs['integration_loop_time_constant'] < 0:
                raise ValueError('[soens_sim] time_constant associated with synaptic integration loop decay must be a real number between zero and infinity')
            else:
                self.integration_loop_time_constant = kwargs['integration_loop_time_constant']
        else:
            if self.integration_loop_temporal_form == 'exponential':
                self.integration_loop_time_constant = 200e-9 #default time constant units of seconds
        self.tau_si = self.integration_loop_time_constant
        
        if 'power_law_exponent' in kwargs:
            # if type(kwargs['power_law_exponent']) == int or type(kwargs['power_law_exponent']) == float:
            if kwargs['power_law_exponent'] > 0:
                raise ValueError('[soens_sim] power_law_exponent associated with synaptic integration loop decay must be a real number between negative infinity and zero')
            else:
                 self.power_law_exponent = kwargs['power_law_exponent']
        else:
            if self.integration_loop_time_constant == 'power_law':                
                self.power_law_exponent = -1 #default power law exponent

        if 'integration_loop_self_inductance' in kwargs:
            # if type(kwargs['integration_loop_self_inductance']) == int or type(kwargs['integration_loop_self_inductance']) == float:
            if kwargs['integration_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Integration loop self inductance associated with synaptic integration loop must be a real number between zero and infinity (units of henries)')
            else:
                 self.integration_loop_self_inductance = kwargs['integration_loop_self_inductance']
        else: 
            self.integration_loop_self_inductance = 10e-9 #default value, units of henries
                        
        if 'integration_loop_output_inductance' in kwargs:
            # if type(kwargs['integration_loop_output_inductance']) == int or type(kwargs['integration_loop_output_inductance']) == float:
            if kwargs['integration_loop_output_inductance'] < 0:
                raise ValueError('[soens_sim] Integration loop output inductance associated with coupling between synaptic integration loop and dendrite or neuron must be a real number between zero and infinity (units of henries)')
            else:
                 self.integration_loop_output_inductance = kwargs['integration_loop_output_inductance']
        else: 
            self.integration_loop_output_inductance = 200e-12 #default value, units of henries
        self.integration_loop_total_inductance = self.integration_loop_self_inductance+self.integration_loop_output_inductance
        self.L_si = self.integration_loop_total_inductance

        if 'synaptic_bias_currents' in kwargs:
            # if type(kwargs['synaptic_bias_current']) == int or type(kwargs['synaptic_bias_current']) == float or type(kwargs['synaptic_bias_current']) == np.float64:
            self.synaptic_bias_currents = kwargs['synaptic_bias_currents']
            if self.num_jjs == 1:
                if len(self.synaptic_bias_currents) == 1:
                    self.I_spd = 20e-6
                    self.I_sy = self.synaptic_bias_currents[0]
                if len(self.synaptic_bias_currents) == 2:
                    self.I_spd = self.synaptic_bias_currents[0]
                    self.I_sy = self.synaptic_bias_currents[1]
            if self.num_jjs == 2:
                if len(self.synaptic_bias_currents) == 1:
                    self.I_spd = 20e-6
                    self.I_sy = self.synaptic_bias_currents[0]
                    self.I_sc = 36e-6
                if len(self.synaptic_bias_currents) == 2:
                    self.I_spd = 20e-6
                    self.I_sy = self.synaptic_bias_currents[0]
                    self.I_sc = self.synaptic_bias_currents[1]
                if len(self.synaptic_bias_currents) == 3:
                    self.I_spd = self.synaptic_bias_currents[0]
                    self.I_sy = self.synaptic_bias_currents[1]
                    self.I_sc = self.synaptic_bias_currents[2]
            if self.num_jjs == 3:
                if len(self.synaptic_bias_currents) == 1:
                    self.I_spd = 20e-6
                    self.I_sy = self.synaptic_bias_currents[0]
                    self.I_jtl = 36e-6
                    self.I_sc = 35e-6
                if len(self.synaptic_bias_currents) == 2:
                    self.I_spd = 20e-6
                    self.I_sy = self.synaptic_bias_currents[0]
                    self.I_jtl = 36e-6
                    self.I_sc = self.synaptic_bias_currents[1]
                if len(self.synaptic_bias_currents) == 3:
                    self.I_spd = 20e-6
                    self.I_sy = self.synaptic_bias_currents[0]
                    self.I_jtl = self.synaptic_bias_currents[1]
                    self.I_sc = self.synaptic_bias_currents[2]
                if len(self.synaptic_bias_currents) == 4:
                    self.I_spd = self.synaptic_bias_currents[0]
                    self.I_sy = self.synaptic_bias_currents[1]
                    self.I_jtl = self.synaptic_bias_currents[2]
                    self.I_sc = self.synaptic_bias_currents[3]                    
        else:
            self.synaptic_bias_currents = [20e-6,28e-6,36e-6,35e-6] #units of amps
            self.I_spd = self.synaptic_bias_currents[0]
            self.I_sy = self.synaptic_bias_currents[1]
            self.I_jtl = self.synaptic_bias_currents[2]
            self.I_sc = self.synaptic_bias_currents[3]
            
        if 'jtl_inductance' in kwargs:            
            self.jtl_inductance = kwargs['jtl_inductance']
            if self.num_jjs == 2:
                self.L_jtl = self.jtl_inductance
            if self.num_jjs == 3:
                if len(self.jtl_inductance) == 1:
                    self.L_jtl1 = self.jtl_inductance
                    self.L_jtl2 = self.jtl_inductance
                if len(self.jtl_inductance) == 2:
                    self.L_jtl1 = self.jtl_inductance[0]
                    self.L_jtl2 = self.jtl_inductance[1]
        else:
            if self.num_jjs == 2:
                self.L_jtl = 77.5e-12 #units of henries
            if self.num_jjs == 3:
                self.L_jtl1 = 77.5e-12 #units of henries
                self.L_jtl2 = 77.5e-12 #units of henries
                
        self.L_spd = 247.5e-9

        if 'integration_loop_bias_current' in kwargs:
            # if type(kwargs['loop_bias_current']) == int or type(kwargs['loop_bias_current']) == float:
            if kwargs['integration_loop_bias_current'] < 0:
                raise ValueError('[soens_sim] loop_bias_current associated with synaptic integration loop must be a real number between xx and yy (units of amps)')
            else:
                 self.integration_loop_bias_current = kwargs['integration_loop_bias_current']
        else:
            self.integration_loop_bias_current = 30e-6 #units of amps
            
        if 'synapse_model_params' in kwargs:
            self.sim_params = kwargs['synapse_model_params']
            sim_params = self.sim_params            
        else:
            sim_params = dict()
            sim_params['dt'] = 0.1e-9 # units of seconds
            sim_params['tf'] = 1e-6 # units of seconds
            sim_params['synapse_model'] = 'lookup_table'
            self.sim_params = sim_params
            
        if 'input_signal_name' in kwargs:
            if kwargs['input_signal_name'] != '':
                self.input_signal_name = kwargs['input_signal_name']
            
        # attach external input signal to synapse
        if hasattr(self, 'input_signal_name'):
            self.input_signal = input_signal.input_signals[self.input_signal_name]
            # print('added input signal with name {} to synapse {}'.format(self.input_signal_name,self.name))                               
        
        self.st_ind_last = 0 # initialize for time stepping
        self.spd_current_memory = 0 # initialize for time stepping
        
        synapse.synapses[self.name] = self
                
        # print('synapse created')
        
        return

    def __del__(self):
        # print('synapse deleted')
        return
        
    def run_sim(self):

        #setup time_vec and input signal
        sim_params = self.sim_params
        
        tf = sim_params['tf']
        dt = sim_params['dt']        
        
        if self.sim_params['synapse_model'] == 'ode':
            unit_factor = 1
        if self.sim_params['synapse_model'] == 'ode__spd_delta':
            unit_factor = 1            
        elif self.sim_params['synapse_model'] == 'lookup_table':
            unit_factor = 1e6
        time_vec = unit_factor*np.arange(0,tf+dt,dt)
            
        if hasattr(self,'input_signal'):
            self.input_spike_times = copy.deepcopy(self.input_signal.spike_times)
        else:
            self.input_spike_times = []
        for ii in range(len(self.input_spike_times)):
            # self.input_spike_times[ii] = self.input_spike_times[ii]*1e6
            self.input_spike_times[ii] = self.input_spike_times[ii]*unit_factor

        # tau_fall = copy.deepcopy(self.integration_loop_time_constant)*1e6
        tau_fall = copy.deepcopy(self.integration_loop_time_constant)
        
        # currents are in uA for time stepper
        # inductances are in pH for time stepper
        # if self.num_jjs == 1:
        #     I_bias_list = [copy.deepcopy(self.I_sy)*1e6]
        #     L_list = [copy.deepcopy(self.L_si)*1e12]

        if self.num_jjs == 1:
            
            if self.sim_params['synapse_model'] == 'ode':
            
                L_list = [copy.deepcopy(self.L_spd),copy.deepcopy(self.L_si)]
                r_list = [8.25,L_list[1]/tau_fall]
                I_bias_list = [copy.deepcopy(self.I_spd),copy.deepcopy(self.I_sy)]
                I_si_vec, I_sf_vec, j_sf_state = synapse_time_stepper__1jj_ode(time_vec,self.input_spike_times,L_list,r_list,I_bias_list)
                
                self.I_si = I_si_vec
                self.I_sf = I_sf_vec
                # self.I_spd = I_sf_vec+I_si_vec-copy.deepcopy(self.I_sy)
                # print(I_sf_vec[0])
                # print(copy.deepcopy(self.I_sy))
                # print(copy.deepcopy(self.I_si[0]))
                # self.I_spd = I_sf_vec-copy.deepcopy(self.I_sy)+copy.deepcopy(self.I_si)
                self.I_spd = I_sf_vec-copy.deepcopy(self.I_sy)
                # print(self.I_spd[0])
                self.time_vec = time_vec
                for ii in range(len(self.input_spike_times)):
                    self.input_spike_times[ii] = self.input_spike_times[ii]
             
            if self.sim_params['synapse_model'] == 'ode__spd_delta':
            
                L_list = [copy.deepcopy(self.L_spd),copy.deepcopy(self.L_si)]
                r_list = [8.25,L_list[1]/tau_fall]
                I_bias_list = [copy.deepcopy(self.I_spd),copy.deepcopy(self.I_sy)]
                I_si_vec, I_sf_vec = synapse_time_stepper__Isf_ode__spd_delta(time_vec,self.input_spike_times,L_list,r_list,I_bias_list)
                
                self.I_si = I_si_vec
                self.I_sf = I_sf_vec
                # self.I_spd = I_sf_vec+I_si_vec-copy.deepcopy(self.I_sy)
                self.I_spd = I_sf_vec-copy.deepcopy(self.I_sy)
                self.time_vec = time_vec
                for ii in range(len(self.input_spike_times)):
                    self.input_spike_times[ii] = self.input_spike_times[ii]                           
                    
            if self.sim_params['synapse_model'] == 'lookup_table':
                
                I_bias_list = [copy.deepcopy(self.I_sy)*1e6]
                L_list = [copy.deepcopy(self.L_si)*1e12,]
                I_spd_vec, I_si_vec, I_sf_vec = synapse_time_stepper(time_vec,self.input_spike_times,self.num_jjs,L_list,I_bias_list,tau_fall*1e6)
                
                self.I_si = I_si_vec*1e-6
                self.I_spd = I_spd_vec*1e-6
                self.I_sf = I_sf_vec*1e-6
                # self.j_sf_state = j_sf_state
                self.time_vec = time_vec*1e-6
                # self.I_c = 1e-6*I_c
                # self.I_reset = 1e-6*I_reset
                for ii in range(len(self.input_spike_times)):
                    self.input_spike_times[ii] = self.input_spike_times[ii]*1e-6
        
        if self.num_jjs == 2:
            
            if self.sim_params['synapse_model'] == 'ode':
                
                L_list = [copy.deepcopy(self.L_spd),copy.deepcopy(self.L_jtl),copy.deepcopy(self.L_si)]
                r_list = [8.25,L_list[2]/tau_fall]
                I_bias_list = [copy.deepcopy(self.I_spd),copy.deepcopy(self.I_sy),copy.deepcopy(self.I_sc)]
                
                # print(L_list)
                # print(r_list)
                # print(I_bias_list)
                
                I_si1_vec, I_si2_vec, I_sf_vec = synapse_time_stepper__2jj__ode(time_vec,self.input_spike_times,L_list,r_list,I_bias_list)
                
                self.I_si = I_si2_vec
                self.I_sf = I_sf_vec
                # self.I_spd = I_sf_vec+I_si_vec-copy.deepcopy(self.I_sy)
                I_spd_vec = np.zeros([len(I_sf_vec)])
                I_sy = copy.deepcopy(self.I_sy)
                I_sc = copy.deepcopy(self.I_sc)
                for ii in range(len(I_sf_vec)):
                    I_spd_vec[ii] = I_sf_vec[ii]+I_si1_vec[ii]-I_sy-I_sc
                self.I_spd = I_spd_vec
                self.time_vec = time_vec
                for ii in range(len(self.input_spike_times)):
                    self.input_spike_times[ii] = self.input_spike_times[ii]
                    
            if self.sim_params['synapse_model'] == 'lookup_table':
                
                I_bias_list = [copy.deepcopy(self.I_sy)*1e6,copy.deepcopy(self.I_sc)*1e6]
                L_list = [copy.deepcopy(self.L_jtl)*1e12,copy.deepcopy(self.L_si)*1e12]
                
                I_spd_vec, I_si_vec, I_sf_vec = synapse_time_stepper(time_vec,self.input_spike_times,self.num_jjs,L_list,I_bias_list,tau_fall*1e6)
                
                self.I_si = I_si_vec*1e-6
                self.I_spd = I_spd_vec*1e-6
                self.I_sf = I_sf_vec*1e-6
                self.time_vec = time_vec*1e-6
                for ii in range(len(self.input_spike_times)):
                    self.input_spike_times[ii] = self.input_spike_times[ii]*1e-6
                    
        if self.num_jjs == 3:
            
            if self.sim_params['synapse_model'] == 'lookup_table':
                
                I_bias_list = [copy.deepcopy(self.I_sy)*1e6,copy.deepcopy(self.I_jtl)*1e6,copy.deepcopy(self.I_sc)*1e6]
                L_list = [copy.deepcopy(self.L_jtl1)*1e12,copy.deepcopy(self.L_jtl2)*1e12,copy.deepcopy(self.L_si)*1e12]               
            
                I_spd_vec, I_si_vec, I_sf_vec = synapse_time_stepper(time_vec,self.input_spike_times,self.num_jjs,L_list,I_bias_list,tau_fall*1e6)
                                
                self.I_si = I_si_vec*1e-6
                self.I_spd = I_spd_vec*1e-6
                self.I_sf = I_sf_vec*1e-6
                self.time_vec = time_vec*1e-6
                for ii in range(len(self.input_spike_times)):
                    self.input_spike_times[ii] = self.input_spike_times[ii]*1e-6
        
        return self
    

class dendrite():
    
    _next_uid = 0
    dendrites = dict()

    def __init__(self, *args, **kwargs):
        
        #make new dendrite
        self.uid = dendrite._next_uid
        self.unique_label = 'd'+str(self.uid)
        dendrite._next_uid += 1
        
        if len(args) > 0:
            if type(args[0]) == str:
                _name = args[0]
            elif type(args[0]) == int or type(args[0]) == float:
                _name = str(args[0])
        else:
            _name = 'unnamed_dendrite'
        self.name = _name
        
        if 'num_jjs' in kwargs:
            if kwargs['num_jjs'] == 2 or kwargs['num_jjs'] == 4:
                self.num_jjs = kwargs['num_jjs']
            else:
                raise ValueError('[soens_sim] num_jjs must be 2, or 4 for a dendrite')
        else:
            self.num_jjs = 4
                    
        if 'inhibitory_or_excitatory' in kwargs:
            if kwargs['inhibitory_or_excitatory'] == 'inhibitory' or kwargs['inhibitory_or_excitatory'] == 'excitatory':
                _i_or_e = kwargs['inhibitory_or_excitatory']
            else:
                raise ValueError('[soens_sim] inhibitory_or_excitatory can either be ''inhibitory'' and ''excitatory''')
        else:
            _i_or_e = 'excitatory'
        self.inhibitory_or_excitatory =  _i_or_e #'excitatory' by default
            
        if 'input_synaptic_connections' in kwargs:
            self.input_synaptic_connections = kwargs['input_synaptic_connections']            
        else:
            self.input_synaptic_connections = []        

        if 'input_synaptic_inductances' in kwargs:
            if type(kwargs['input_synaptic_inductances']) == list:
                self.input_synaptic_inductances = dict()
                for ii in range(len(self.input_synaptic_connections)):
                    self.input_synaptic_inductances[self.input_synaptic_connections[ii]] = kwargs['input_synaptic_inductances'][ii]
                # self.input_synaptic_inductances = kwargs['input_synaptic_inductances']
            else:
                raise ValueError('[soens_sim] Input synaptic inductances to dendrites are specified as a list of pairs of real numbers with one pair per synaptic connection. The first element of the pair is the inductance on the dendritic receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the synapse and the dendritic receiving loop.')
        else:
            self.input_synaptic_inductances =  [[]]
                        
        if 'input_dendritic_connections' in kwargs:
            self.input_dendritic_connections = kwargs['input_dendritic_connections']            
        else:
            self.input_dendritic_connections = []        

        if 'input_dendritic_inductances' in kwargs:
            # print('{}'.format(kwargs['input_dendritic_inductances']))
            if type(kwargs['input_dendritic_inductances']) == list:
                self.input_dendritic_inductances = dict()
                for ii in range(len(self.input_dendritic_connections)):
                    self.input_dendritic_inductances[self.input_dendritic_connections[ii]] = kwargs['input_dendritic_inductances'][ii]
                # self.input_dendritic_inductances = kwargs['input_dendritic_inductances']
            else:
                raise ValueError('[soens_sim] Input dendritic inductances to dendrites are specified as a list of pairs of real numbers with one pair per dendritic connection. The first element of the pair is the inductance on the dendritic receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the input dendrite and the dendritic receiving loop.')
        else:
            self.input_dendritic_inductances =  []
                                    
        if 'input_direct_connections' in kwargs:
            self.input_direct_connections = kwargs['input_direct_connections']            
        else:
            self.input_direct_connections = []             

        if 'input_direct_inductances' in kwargs:
            if type(kwargs['input_direct_inductances']) == list:
                self.input_direct_inductances = dict()
                for ii in range(len(self.input_direct_connections)):
                    self.input_direct_inductances[self.input_direct_connections[ii]] = kwargs['input_direct_inductances'][ii]
                # self.input_direct_inductances = kwargs['input_direct_inductances']
            else:
                raise ValueError('[soens_sim] Input direct inductances to dendrites are specified as a list of pairs of real numbers with one pair per direct connection. The first element of the pair is the inductance on the dendritic receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the input direct signal and the dendritic receiving loop.')
        else:
            self.input_direct_inductances =  []
                    
        if 'circuit_inductances' in kwargs:
            if type(kwargs['circuit_inductances']) == list and len(kwargs['circuit_inductances']) == 4:
                self.circuit_inductances = kwargs['circuit_inductances']
            else:
                raise ValueError('[soens_sim] circuit_inductances is a list of four real numbers greater than zero with units of henries. The first element is the self inductance of the left branch of the DR loop, excluding the JJ and any mutual inductor inputs. The second element is the right branch of the DR loop, excluding the JJ and any mutual inductor inputs. The third element is the inductor to the right of the DR loop that goes to the JTL. The fourth element is the inductor in the JTL. All other contributions to DR loop inductance (JJs and MIs) will be handled separately, as will the inductance of the DI loop.')
        else:
            self.circuit_inductances = [20e-12, 20e-12, 200e-12, 77.5e-12]            

        if 'thresholding_junction_critical_current' in kwargs:
            _Ic = kwargs['thresholding_junction_critical_current']
            # else:
                # raise ValueError('[soens_sim] Thresholding junction critical current must be a real number with units of amps')
        else:
            _Ic = 40e-6 #default J_th Ic = 40 uA
        self.thresholding_junction_critical_current =  _Ic
            
        if 'bias_currents' in kwargs:
            _Ib = kwargs['bias_currents']
        else:
            _Ib = [72e-6, 29e-6, 35e-6] #[bias to DR loop (J_th), bias to JTL, bias to DI loop]
        self.bias_currents =  _Ib
            
        if 'integration_loop_self_inductance' in kwargs:
            # if type(kwargs['integration_loop_self_inductance']) == int or type(kwargs['integration_loop_self_inductance']) == float:
            if kwargs['integration_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Integration loop self inductance associated with dendritic integration loop must be a real number between zero and infinity (units of henries)')
            else:
                 self.integration_loop_self_inductance = kwargs['integration_loop_self_inductance']
        else: 
            self.integration_loop_self_inductance = 10e-9 #default value, units of henries
                        
        if 'integration_loop_output_inductance' in kwargs:
            if type(kwargs['integration_loop_output_inductance']) == int or type(kwargs['integration_loop_output_inductance']) == float:
                if kwargs['integration_loop_output_inductance'] < 0:
                    raise ValueError('[soens_sim] Integration loop output inductance associated with coupling between synaptic integration loop and dendrite or neuron must be a real number between zero and infinity (units of henries)')
                else:
                     self.integration_loop_output_inductance = kwargs['integration_loop_output_inductance']
                     ti = self.integration_loop_output_inductance
            if type(kwargs['integration_loop_output_inductance']) == list:
                self.integration_loop_output_inductance = kwargs['integration_loop_output_inductance']
                ti = sum(self.integration_loop_output_inductance)
        else: 
            self.integration_loop_output_inductance = 200e-12 #default value, units of henries
        self.integration_loop_total_inductance = self.integration_loop_self_inductance+ti
        
        if 'integration_loop_temporal_form' in kwargs:
            if kwargs['integration_loop_temporal_form'] == 'exponential' or kwargs['loop_temporal_form'] == 'power_law':
                _temporal_form = kwargs['integration_loop_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid integration loop temporal form to dendrite %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'exponential'
        self.integration_loop_temporal_form =  _temporal_form #'exponential' or 'power_law'; 'exponential' by default
        
        if 'integration_loop_time_constant' in kwargs:
            if kwargs['integration_loop_time_constant'] < 0:
                raise ValueError('[soens_sim] integration_loop_time_constant associated with dendritic decay must be a real number between zero and infinity')
            else:
                self.integration_loop_time_constant = kwargs['integration_loop_time_constant']
        else:
            if self.integration_loop_temporal_form == 'exponential':
                self.integration_loop_time_constant = 200e-9 #default time constant units of seconds
             
        if 'integration_loop_power_law_exponent' in kwargs:
            if kwargs['integration_loop_power_law_exponent'] > 0:
                raise ValueError('[soens_sim] power_law_exponent associated with dendritic decay must be a real number between negative infinity and zero')
            else:
                 self.power_law_exponent = kwargs['integration_loop_power_law_exponent']
        else:
            if self.integration_loop_temporal_form == 'power_law':                
                self.integration_loop_power_law_exponent = -1 #default power law exponent
                
        if 'dendrite_model_params' in kwargs:
            self.dendrite_model_params = kwargs['dendrite_model_params']                 
        else:
            sim_params = dict()
            # sim_params['amp'] = 9.27586207
            # sim_params['mu1'] = 1.92758621
            # sim_params['mu2'] = 0.45344828
            # sim_params['mu3'] = 0.87959184
            # sim_params['mu4'] = 0.59591837
            sim_params['dt'] = 0.1e-9
            sim_params['tf'] = 1e-6
            self.dendrite_model_params = sim_params
 
        dendrite.dendrites[self.name] = self
            
        # print('dendrite created')        
        return      
    
    def make_connections(self):

        #add synapses to dendrite
        self.synapses = []
        for name_1 in self.input_synaptic_connections:
            self.synapses.append(synapse.synapses[name_1])
               
        #add dendrites to dendrite
        self.dendrites = []
        for name_1 in self.input_dendritic_connections:
            self.dendrites.append(dendrite.dendrites[name_1])
                   
        #then add direct connections to dendrite
        self.direct_connections = []
        for name_1 in self.input_direct_connections:
            self.direct_connections.append(input_signal.input_signals[name_1])
            
        return self    
    
    def run_sim(self):
        
        # set up time vec
        dt = self.dendrite_model_params['dt']
        tf = self.dendrite_model_params['tf']
        time_vec = np.arange(0,tf+dt,dt)
        
        # attach synapses, dendrites, and direct connections to dendrite
        self.make_connections()        
        # print('simulating dendrite with {:d} synapses, {:d} dendrites, and {:d} direct connections\n\n'.format(len(self.synapses),len(self.dendrites),len(self.direct_connections)))
            
        self.time_vec = time_vec
        
        if hasattr(self.direct_connections[0],'piecewise_linear'):
            dendritic_drive = dendritic_drive__piecewise_linear(time_vec,self.direct_connections[0].piecewise_linear)
        if hasattr(self.direct_connections[0],'square_pulse_train'):
            dendritic_drive = dendritic_drive__square_pulse_train(time_vec,self.direct_connections[0].square_pulse_train)
        if hasattr(self.direct_connections[0],'exponential_pulse_train'):
            dendritic_drive = dendritic_drive__exp_pls_train__LR(time_vec,self.direct_connections[0].exponential_pulse_train)

        # plot_dendritic_drive(time_vec, dendritic_drive)
        self.dendritic_drive = dendritic_drive
        
        tau_di = self.integration_loop_time_constant
        L3 = self.integration_loop_self_inductance+self.integration_loop_output_inductance
        I_di_vec = dendrite_time_stepper(time_vec,self) # dendritic_drive,L3,tau_di
        
        self.I_di = I_di_vec        
        
        return self

    def __del__(self):
        # print('dendrite deleted')
        return


class neuron():

    _next_uid = 0
    neurons = dict()

    def __init__(self, *args, **kwargs):

        #make new neuron
        self.uid = neuron._next_uid
        self.unique_label = 'n'+str(self.uid)
        # self._instances.add(weakref.ref(self))
        neuron._next_uid += 1

        if len(args) > 0:
            if type(args[0]) == str:
                _name = args[0]
            elif type(args[0]) == int or type(args[0]) == float:
                _name = str(args[0])
        else:
            _name = 'unnamed_neuron'
        self.name = _name
        
        if 'num_jjs' in kwargs:
            if kwargs['num_jjs'] == 2 or kwargs['num_jjs'] == 4:
                self.num_jjs = kwargs['num_jjs']
            else:
                raise ValueError('[soens_sim] num_jjs must be 2, or 4 for a dendrite')
        else:
            self.num_jjs = 4
        
        if 'circuit_inductances' in kwargs:
            if type(kwargs['circuit_inductances']) == list and len(kwargs['circuit_inductances']) == 4:
                self.circuit_inductances = kwargs['circuit_inductances']
            else:
                raise ValueError('[soens_sim] circuit_inductances is a list of four real numbers greater than zero with units of henries. The first element is the self inductance of the left branch of the DR loop, excluding the JJ and any mutual inductor inputs. The second element is the right branch of the DR loop, excluding the JJ and any mutual inductor inputs. The third element is the inductor to the right of the DR loop that goes to the JTL. The fourth element is the inductor in the JTL. All other contributions to DR loop inductance (JJs and MIs) will be handled separately, as will the inductance of the DI loop.')
        else:
            self.circuit_inductances = [20e-12, 20e-12, 200e-12, 77.5e-12]
        
        if 'input_direct_connections' in kwargs:
            self.input_direct_connections = kwargs['input_direct_connections']            
        else:
            self.input_direct_connections = []        

        if 'input_direct_inductances' in kwargs:
            if type(kwargs['input_direct_inductances']) == list:
                    self.input_direct_inductances = kwargs['input_direct_inductances']
            else:
                raise ValueError('[soens_sim] Input direct inductances to neurons are specified as a list of pairs of real numbers with one pair per direct connection. The first element of the pair is the inductance on the neuronal receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the direct connection and the neuronal receiving loop.')
        else:
            self.input_direct_inductances =  [[]]
            
        if 'input_synaptic_connections' in kwargs:
            self.input_synaptic_connections = kwargs['input_synaptic_connections']            
        else:
            self.input_synaptic_connections = []        

        if 'input_synaptic_inductances' in kwargs:
            if type(kwargs['input_synaptic_inductances']) == list:
                    self.input_synaptic_inductances = kwargs['input_synaptic_inductances']
            else:
                raise ValueError('[soens_sim] Input synaptic inductances to neurons are specified as a list of pairs of real numbers with one pair per synaptic connection. The first element of the pair is the inductance on the neuronal receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the synapse and the neuronal receiving loop.')
        else:
            self.input_synaptic_inductances =  [[]]
            
        if 'input_dendritic_connections' in kwargs:
            self.input_dendritic_connections = kwargs['input_dendritic_connections']            
        else:
            self.input_dendritic_connections = []        

        if 'input_dendritic_inductances' in kwargs:
            if type(kwargs['input_dendritic_inductances']) == list:
                    self.input_dendritic_inductances = kwargs['input_dendritic_inductances']
            else:
                raise ValueError('[soens_sim] Input dendritic inductances to neurons are specified as a list of pairs of real numbers with one pair per dendritic connection. The first element of the pair is the inductance on the neuronal receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the dendritic and the neuronal receiving loop.')
        else:
            self.input_dendritic_inductances =  [[]]
            
        if 'thresholding_junction_critical_current' in kwargs:
            # if type(kwargs['thresholding_junction_critical_current']) == float:
            _Ic = kwargs['thresholding_junction_critical_current']
            # else:
                # raise ValueError('[soens_sim] Thresholding junction critical current must be a real number with units of amps')
        else:
            _Ic = 40e-6 #default J_th Ic = 40 uA
        self.thresholding_junction_critical_current =  _Ic
                    
        if 'bias_currents' in kwargs:
            _Ib = kwargs['bias_currents']
        else:
            _Ib = [74e-6, 36e-6, 35e-6] #[bias to NR loop (J_th), bias to JTL, bias to NI loop]
        self.bias_currents =  _Ib
        
        if 'integration_loop_self_inductance' in kwargs:
            # if type(kwargs['integration_loop_self_inductance']) == int or type(kwargs['integration_loop_self_inductance']) == float:
            if kwargs['integration_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Integration loop self inductance associated with neuronal integration loop must be a real number between zero and infinity (units of henries)')
            else:
                 self.integration_loop_self_inductance = kwargs['integration_loop_self_inductance']
        else: 
            self.integration_loop_self_inductance = 775e-12 #default value, units of henries
                        
        if 'integration_loop_output_inductances' in kwargs:
            if type(kwargs['integration_loop_output_inductances']) == list:
                self.integration_loop_output_inductances = kwargs['integration_loop_output_inductances']
        else:
            self.integration_loop_output_inductances = [[400e-12,1],[200e-12,1]] # defaults; [[inudctor_to_latching_jj, k_to_latching_jj],[inductor_to_refractory_dendrite, k_to_refractory_dendrite]] 
        ti = 0
        for ii in range(len(self.integration_loop_output_inductances)):
            ti += self.integration_loop_output_inductances[ii][0]        
        self.integration_loop_total_inductance = self.integration_loop_self_inductance+ti
        
        if 'integration_loop_temporal_form' in kwargs:
            if kwargs['integration_loop_temporal_form'] == 'exponential' or kwargs['loop_temporal_form'] == 'power_law':
                _temporal_form = kwargs['integration_loop_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid integration loop temporal form to neuron %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'exponential'
        self.integration_loop_temporal_form =  _temporal_form #'exponential' or 'power_law'; 'exponential' by default
        
        if 'integration_loop_time_constant' in kwargs:
            if kwargs['integration_loop_time_constant'] < 0:
                raise ValueError('[soens_sim] integration_loop_time_constant associated with neuronal decay must be a real number between zero and infinity')
            else:
                self.integration_loop_time_constant = kwargs['integration_loop_time_constant']
        else:
            if self.integration_loop_temporal_form == 'exponential':
                self.integration_loop_time_constant = 5e-9 #default time constant units of seconds        
                
        if 'integration_loop_power_law_exponent' in kwargs:
            if kwargs['integration_loop_power_law_exponent'] > 0:
                raise ValueError('[soens_sim] power_law_exponent associated with dendritic decay must be a real number between negative infinity and zero')
            else:
                 self.power_law_exponent = kwargs['integration_loop_power_law_exponent']
        else:
            if self.integration_loop_temporal_form == 'power_law':                
                self.integration_loop_power_law_exponent = -1 #default power law exponent
                
        if 'time_params' in kwargs:
            self.time_params = kwargs['time_params']                 
        else:
            time_params = dict()
            time_params['dt'] = 0.1e-9
            time_params['tf'] = 1e-6
            self.time_params = time_params
                         
        if 'refractory_temporal_form' in kwargs:
            if kwargs['refractory_temporal_form'] == 'exponential' or kwargs['refractory_temporal_form'] == 'power_law':
                _temporal_form = kwargs['refractory_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid loop temporal form to neuron %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'exponential'
        self.refractory_temporal_form =  _temporal_form #'exponential' or 'power_law'; 'exponential' by default
                
        if 'refractory_time_constant' in kwargs:
            # if type(kwargs['refractory_time_constant']) == int or type(kwargs['refractory_time_constant']) == float:
            if kwargs['refractory_time_constant'] < 0:
                raise ValueError('[soens_sim] time_constant associated with neuronal refraction must be a real number between zero and infinity')
            else:
                self.refractory_time_constant = kwargs['refractory_time_constant']
        else:
            if self.refractory_temporal_form == 'exponential':
                self.refractory_time_constant = 50e-9 #default time constant, units of seconds
            
        if 'refractory_thresholding_junction_critical_current' in kwargs:
            _Ic = kwargs['refractory_thresholding_junction_critical_current']
        else:
            _Ic = 40e-6 #default J_th Ic = 40 uA
        self.refractory_thresholding_junction_critical_current = _Ic
        
        if 'refractory_loop_circuit_inductances' in kwargs:
            if type(kwargs['refractory_loop_circuit_inductances']) == list and len(kwargs['refractory_loop_circuit_inductances']) == 4:
                self.refractory_loop_circuit_inductances = kwargs['refractory_loop_circuit_inductances']
            else:
                raise ValueError('[soens_sim] refractory_loop_circuit_inductances is a list of four real numbers greater than zero with units of henries. The first element is the self inductance of the left branch of the DR loop, excluding the JJ and any mutual inductor inputs. The second element is the right branch of the DR loop, excluding the JJ and any mutual inductor inputs. The third element is the inductor to the right of the DR loop that goes to the JTL. The fourth element is the inductor in the JTL. All other contributions to DR loop inductance (JJs and MIs) will be handled separately, as will the inductance of the DI loop.')
        else:
            self.refractory_loop_circuit_inductances = [20e-12, 20e-12, 200e-12, 77.5e-12]
            
        if 'refractory_loop_self_inductance' in kwargs:
            # if type(kwargs['refractory_loop_self_inductance']) == int or type(kwargs['refractory_loop_self_inductance']) == float:
            if kwargs['refractory_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Refractory loop self inductance associated with refractory suppression loop must be a real number between zero and infinity (units of henries)')
            else:
                 self.refractory_loop_self_inductance = kwargs['refractory_loop_self_inductance']
        else: 
            self.refractory_loop_self_inductance = 1e-9 #default value, units of henries
                        
        if 'refractory_loop_output_inductance' in kwargs:
            # if type(kwargs['refractory_loop_output_inductance']) == int or type(kwargs['refractory_loop_output_inductance']) == float:
            if kwargs['refractory_loop_output_inductance'] < 0:
                raise ValueError('[soens_sim] Refractory loop output inductance associated with coupling between refractory suppression loop and neuron must be a real number between zero and infinity (units of henries)')
            else:
                 self.refractory_loop_output_inductance = kwargs['refractory_loop_output_inductance']
        else: 
            self.refractory_loop_output_inductance = 200e-12 #default value, units of henries 
            
        if 'refractory_bias_currents' in kwargs:
            _Ib = kwargs['refractory_bias_currents']
        else:
            _Ib = [74e-6, 36e-6, 35e-6] #[bias to NR loop (J_th), bias to JTL, bias to NI loop]
        self.refractory_bias_currents =  _Ib
        
        if 'refractory_receiving_input_inductance' in kwargs:
            if type(kwargs['refractory_receiving_input_inductance']) == list:
                    self.refractory_receiving_input_inductance = kwargs['refractory_receiving_input_inductance']
            else:
                raise ValueError('[soens_sim] refractory_receiving_input_inductance is specified as a pair of real numbers. The first element of the pair is the inductance on the refractory receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the refractory dendrite and the neuronal integration loop.')
        else:
            self.refractory_receiving_input_inductance =  [20e-12,1]
            
        if 'neuronal_receiving_input_refractory_inductance' in kwargs:
            if type(kwargs['neuronal_receiving_input_refractory_inductance']) == list:
                    self.neuronal_receiving_input_refractory_inductance = kwargs['neuronal_receiving_input_refractory_inductance']
            else:
                raise ValueError('[soens_sim] neuronal_receiving_input_refractory_inductance is specified as a pair of real numbers. The first element of the pair is the inductance on the neuronal receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the refractory dendrite and the neuronal receiving loop.')
        else:
            self.neuronal_receiving_input_refractory_inductance =  [20e-12,1]
                         
        if 'homeostatic_temporal_form' in kwargs:
            if kwargs['homeostatic_temporal_form'] == 'exponential' or kwargs['homeostatic_temporal_form'] == 'power_law':
                _temporal_form = kwargs['homeostatic_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid loop temporal form to neuron %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'exponential'
        self.homeostatic_temporal_form =  _temporal_form #'exponential' or 'power_law'; 'exponential' by default
                
        if 'homeostatic_time_constant' in kwargs:
            # if type(kwargs['refractory_time_constant']) == int or type(kwargs['refractory_time_constant']) == float:
            if kwargs['homeostatic_time_constant'] < 0:
                raise ValueError('[soens_sim] time_constant associated with neuronal homeostasis must be a real number between zero and infinity')
            else:
                self.homeostatic_time_constant = kwargs['homeostatic_time_constant']
        else:
            if self.homeostatic_temporal_form == 'exponential':
                self.homeostatic_time_constant = 50e-9 #default time constant, units of seconds
            
        if 'homeostatic_thresholding_junction_critical_current' in kwargs:
            _Ic = kwargs['homeostatic_thresholding_junction_critical_current']
        else:
            _Ic = 40e-6 #default J_th Ic = 40 uA
        self.homeostatic_thresholding_junction_critical_current = _Ic
        
        if 'homeostatic_loop_circuit_inductances' in kwargs:
            if type(kwargs['homeostatic_loop_circuit_inductances']) == list and len(kwargs['homeostatic_loop_circuit_inductances']) == 4:
                self.homeostatic_loop_circuit_inductances = kwargs['homeostatic_loop_circuit_inductances']
            else:
                raise ValueError('[soens_sim] homeostatic_loop_circuit_inductances is a list of four real numbers greater than zero with units of henries. The first element is the self inductance of the left branch of the DR loop, excluding the JJ and any mutual inductor inputs. The second element is the right branch of the DR loop, excluding the JJ and any mutual inductor inputs. The third element is the inductor to the right of the DR loop that goes to the JTL. The fourth element is the inductor in the JTL. All other contributions to DR loop inductance (JJs and MIs) will be handled separately, as will the inductance of the DI loop.')
        else:
            self.homeostatic_loop_circuit_inductances = [20e-12, 20e-12, 200e-12, 77.5e-12]
            
        if 'homeostatic_loop_self_inductance' in kwargs:
            if kwargs['homeostatic_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Homeostatic loop self inductance associated with homeostasis must be a real number between zero and infinity (units of henries)')
            else:
                 self.homeostatic_loop_self_inductance = kwargs['homeostatic_loop_self_inductance']
        else: 
            self.homeostatic_loop_self_inductance = 1e-9 #default value, units of henries
                        
        if 'homeostatic_loop_output_inductance' in kwargs:
            if kwargs['refractory_loop_output_inductance'] < 0:
                raise ValueError('[soens_sim] Homeostatic loop output inductance associated with coupling between homeostatic loop and neuron must be a real number between zero and infinity (units of henries)')
            else:
                 self.homeostatic_loop_output_inductance = kwargs['homeostatic_loop_output_inductance']
        else: 
            self.homeostatic_loop_output_inductance = 200e-12 #default value, units of henries 
            
        if 'homeostatic_bias_currents' in kwargs:
            _Ib = kwargs['homeostatic_bias_currents']
        else:
            _Ib = [74e-6, 36e-6, 35e-6] #[bias to NR loop (J_th), bias to JTL, bias to NI loop]
        self.homeostatic_bias_currents =  _Ib
        
        if 'homeostatic_receiving_input_inductance' in kwargs:
            if type(kwargs['homeostatic_receiving_input_inductance']) == list:
                    self.homeostatic_receiving_input_inductance = kwargs['homeostatic_receiving_input_inductance']
            else:
                raise ValueError('[soens_sim] homeostatic_receiving_input_inductance is specified as a pair of real numbers. The first element of the pair is the inductance on the homeostatic receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the homeostatic dendrite and the neuronal integration loop.')
        else:
            self.homeostatic_receiving_input_inductance =  [20e-12,1]
            
        if 'neuronal_receiving_input_homeostatic_inductance' in kwargs:
            if type(kwargs['neuronal_receiving_input_homeostatic_inductance']) == list:
                    self.neuronal_receiving_input_homeostatic_inductance = kwargs['neuronal_receiving_input_homeostatic_inductance']
            else:
                raise ValueError('[soens_sim] neuronal_receiving_input_homeostatic_inductance is specified as a pair of real numbers. The first element of the pair is the inductance on the neuronal receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the homeostatic dendrite and the neuronal receiving loop.')
        else:
            self.neuronal_receiving_input_homeostatic_inductance =  [20e-12,1]            
        
        # make neuron cell body as dendrite
        temp_list_1 = self.input_dendritic_connections
        temp_str = '{}__r'.format(self.name)
        temp_list_1.append(temp_str)
        temp_list_2 = self.input_dendritic_inductances
        temp_list_2.append(self.neuronal_receiving_input_refractory_inductance)
        # print('self.circuit_inductances = {}'.format(self.circuit_inductances))
        neuron_dendrite = dendrite('{}__d'.format(self.name),
                                   num_jjs = self.num_jjs,
                                   inhibitory_or_excitatory = 'excitatory', 
                                   circuit_inductances = self.circuit_inductances,
                                   input_direct_connections = self.input_direct_connections,
                                   input_direct_inductances = self.input_direct_inductances,
                                   input_synaptic_connections = self.input_synaptic_connections, 
                                   input_synaptic_inductances = self.input_synaptic_inductances,
                                   input_dendritic_connections = temp_list_1, 
                                   input_dendritic_inductances = temp_list_2,
                                   thresholding_junction_critical_current = self.thresholding_junction_critical_current, 
                                   bias_currents = self.bias_currents,
                                   integration_loop_self_inductance = self.integration_loop_self_inductance,
                                   integration_loop_output_inductance = self.integration_loop_output_inductances[1][0],
                                   integration_loop_temporal_form = 'exponential', 
                                   integration_loop_time_constant = self.integration_loop_time_constant,
                                   dendrite_model_params = self.time_params)                
                 
        # make refractory dendrite
        refractory_loop = dendrite('{}__r'.format(self.name),
                                  num_jjs = self.num_jjs,
                                  inhibitory_or_excitatory = 'inhibitory',
                                  circuit_inductances = self.refractory_loop_circuit_inductances,
                                  thresholding_junction_critical_current = self.refractory_thresholding_junction_critical_current,
                                  input_dendritic_connections = ['{}__d'.format(self.name)],
                                  input_dendritic_inductances = [self.refractory_receiving_input_inductance], # [self.integration_loop_output_inductances[1]],
                                  bias_currents = self.refractory_bias_currents,
                                  integration_loop_temporal_form = self.refractory_temporal_form,
                                  integration_loop_time_constant = self.refractory_time_constant,
                                  integration_loop_self_inductance = self.refractory_loop_self_inductance,
                                  integration_loop_output_inductance = self.refractory_loop_output_inductance,
                                  dendrite_model_params = self.time_params)
        
        # refractory_loop.unique_label = self.unique_label+'_rs'
        # neuron.refractory_loop = refractory_loop
        # self.input_dendritic_connections.append(refractory_loop.name)        
                
        # print('neuron created')
        neuron.neurons[self.name] = self
        
        return

    def __del__(self):
        # print('neuron deleted')
        return
    
    def make_connections(self):

        #add dendrites to neuron
        self.dendrites = dict()
        for name_1 in self.input_dendritic_connections:
            
            #first add synapses to dendrites
            dendrite.dendrites[name_1].synapses = dict()
            for name_2 in dendrite.dendrites[name_1].input_synaptic_connections:
                if hasattr(synapse.synapses[name_2],'input_signal'):
                    synapse.synapses[name_2].input_spike_times = copy.deepcopy(synapse.synapses[name_2].input_signal.spike_times)
                else:
                    synapse.synapses[name_2].input_spike_times = []
                dendrite.dendrites[name_1].synapses[synapse.synapses[name_2].name] = synapse.synapses[name_2]
                self.synapses[name_2] = synapse.synapses[name_2]
                
            #then add direct connections to dendrites
            dendrite.dendrites[name_1].direct_connections = dict()
            for name_3 in dendrite.dendrites[name_1].input_direct_connections:                
                dendrite.dendrites[name_1].direct_connections[input_signal.input_signals[name_3].name] = input_signal.input_signals[name_3]
                
            #then add dendrites to dendrites
            dendrite.dendrites[name_1].dendrites = dict()
            for name_2 in dendrite.dendrites[name_1].input_dendritic_connections:
                dendrite.dendrites[name_1].dendrites[dendrite.dendrites[name_2].name] = dendrite.dendrites[name_2]
                self.dendrites[name_2] = dendrite.dendrites[name_2]
            
            #then add dendrites to neuron
            self.dendrites[dendrite.dendrites[name_1].name] = dendrite.dendrites[name_1]
        
        #then add synapses to neuron
        self.synapses = dict()
        for name_1 in self.input_synaptic_connections:
            if hasattr(synapse.synapses[name_1],'input_signal'):
                synapse.synapses[name_1].input_spike_times = copy.deepcopy(synapse.synapses[name_1].input_signal.spike_times)
            else:
                synapse.synapses[name_1].input_spike_times = []
            self.synapses[synapse.synapses[name_1].name] = synapse.synapses[name_1]
            
        #also add direct connections to neuron
        self.direct_connections = dict()
        for name_1 in self.input_direct_connections:
            self.direct_connections[input_signal.input_signals[name_1].name] = input_signal.input_signals[name_1]                      
            
        #finally, add self to self as dendrite
        self.dendrites['{}__d'.format(self.name)] = dendrite.dendrites['{}__d'.format(self.name)]
                        
        return self   
    
    
    def sum_inductances(self):  
        
        print_progress = False
    
        # go through all dendrites in the neuron. remember the neuron itself is a dendrite, so it is included here
        # currently set up so all excitatory connections are on the left, all inhibitory on the right branch of the DR loop. is that good?
        for name_dendrite in self.dendrites:
            
            if print_progress == True:
                print('name_dendrite = {}'.format(name_dendrite))
            
            self.dendrites[name_dendrite].L_left = self.dendrites[name_dendrite].circuit_inductances[0]
            self.dendrites[name_dendrite].L_right = self.dendrites[name_dendrite].circuit_inductances[1]
            
            if print_progress == True:
                print('1: self.dendrites[name_dendrite].L_left = {}'.format(self.dendrites[name_dendrite].L_left))
                print('1: self.dendrites[name_dendrite].L_right = {}'.format(self.dendrites[name_dendrite].L_right))
            
            for name_direct in self.dendrites[name_dendrite].input_direct_connections:
                
                if self.dendrites[name_dendrite].inhibitory_or_excitatory == 'excitatory':
                    self.dendrites[name_dendrite].L_left += self.dendrites[name_dendrite].input_direct_inductances[name_direct][0]
                    
                elif self.dendrites[name_dendrite].inhibitory_or_excitatory == 'inhibitory':
                    self.dendrites[name_dendrite].L_right += self.dendrites[name_dendrite].input_direct_inductances[name_direct][0]
                    
                if print_progress == True:
                    print('2: self.dendrites[name_dendrite].L_left = {}'.format(self.dendrites[name_dendrite].L_left))
                    print('2: self.dendrites[name_dendrite].L_right = {}'.format(self.dendrites[name_dendrite].L_right))
            
            for name_dendrite_in in self.dendrites[name_dendrite].input_dendritic_connections:
                
                if self.dendrites[name_dendrite_in].inhibitory_or_excitatory == 'excitatory':
                    self.dendrites[name_dendrite].L_left += self.dendrites[name_dendrite].input_dendritic_inductances[name_dendrite_in][0]
                    
                elif self.dendrites[name_dendrite_in].inhibitory_or_excitatory == 'inhibitory':
                    self.dendrites[name_dendrite].L_right += self.dendrites[name_dendrite].input_dendritic_inductances[name_dendrite_in][0]
                    
                if print_progress == True:
                    print('3: self.dendrites[name_dendrite].L_left = {}'.format(self.dendrites[name_dendrite].L_left))
                    print('3: self.dendrites[name_dendrite].L_right = {}'.format(self.dendrites[name_dendrite].L_right))
                
            for name_synapse in self.dendrites[name_dendrite].input_synaptic_connections:
                
                if print_progress == True:
                    print('name_synapse = {}'.format(name_synapse))                
                
                if self.dendrites[name_dendrite].inhibitory_or_excitatory == 'excitatory':
                    self.dendrites[name_dendrite].L_left += self.dendrites[name_dendrite].input_synaptic_inductances[name_synapse][0]
                    
                elif self.dendrites[name_dendrite].inhibitory_or_excitatory == 'inhibitory':
                    self.dendrites[name_dendrite].L_right += self.dendrites[name_dendrite].input_synaptic_inductances[name_synapse][0]
    
                if print_progress == True:
                    print('4: self.dendrites[name_dendrite].L_left = {}'.format(self.dendrites[name_dendrite].L_left))
                    print('4: self.dendrites[name_dendrite].L_right = {}'.format(self.dendrites[name_dendrite].L_right))
                             
            if print_progress == True:
                print('5: self.dendrites[name_dendrite].L_left = {}'.format(self.dendrites[name_dendrite].L_left))
                print('5: self.dendrites[name_dendrite].L_right = {}'.format(self.dendrites[name_dendrite].L_right))
                
        return self
    
    def construct_dendritic_drives(self):
        
        for de_name in self.dendrites:
            
            for dir_sig in self.dendrites[de_name].input_direct_connections:
                
                if hasattr(self.dendrites[de_name].direct_connections[dir_sig],'piecewise_linear'):
                    dendritic_drive = dendritic_drive__piecewise_linear(self.time_vec,self.dendrites[de_name].direct_connections[dir_sig].piecewise_linear)
                    
                if hasattr(self.dendrites[de_name].direct_connections[dir_sig],'square_pulse_train'):
                    dendritic_drive = dendritic_drive__square_pulse_train(self.time_vec,self.dendrites[de_name].direct_connections[dir_sig].square_pulse_train)
                    
                if hasattr(self.dendrites[de_name].direct_connections[dir_sig],'exponential_pulse_train'):
                    dendritic_drive = dendritic_drive__exp_pls_train__LR(self.time_vec,self.dendrites[de_name].direct_connections[dir_sig].exponential_pulse_train)
    
                # plot_dendritic_drive(time_vec, dendritic_drive)
                self.dendrites[de_name].direct_connections[dir_sig].drive_signal = dendritic_drive
    
        
    def run_sim(self):
        
        # set up time vec
        dt = self.time_params['dt']
        tf = self.time_params['tf']
                
        time_vec = np.arange(0,tf+dt,dt)    
        self.time_vec = time_vec            

        # attach synapses, dendrites, and direct connections to neuron
        self.make_connections()        
        print('\nsimulating neuron with {:d} synapse(s) and {:d} dendrite(s)\n'.format(len(self.synapses),len(self.dendrites)))
        
        self.sum_inductances()
        
        self.construct_dendritic_drives()
        
        self = neuron_time_stepper(time_vec,self)
        
        # calculate spike times
        self.output_voltage = self.I_ni_vec # self.integration_loop_output_inductances[0][0]*np.diff(self.I_ni_vec)        
        self.voltage_peaks, _ = find_peaks(self.output_voltage, distance = 5e-9/dt) # , height = min_peak_height, )
        self.spike_times = self.time_vec[self.voltage_peaks]
        
        # calculate receiving loop inductance
        # self.receiving_loop_total_inductance = self.receiving_loop_self_inductance
        # for ii in range(len(self.synapses)):
        #     # print(self.synapses[ii].unique_label)
        #     self.receiving_loop_total_inductance += self.input_synaptic_inductances[ii][0]
        # for ii in range(len(self.dendrites)):
        #     # print(self.dendrites[ii].unique_label)
        #     self.receiving_loop_total_inductance += self.input_dendritic_inductances[ii][0]
                                     
        # #find index of refractory suppresion loop
        # for ii in range(len(self.dendrites)):
        #     if self.dendrites[ii].name == self.name+'__r':
        #         rs_index = ii
                
        # for ii in range(len(self.synapses)):
        #     if self.synapses[ii].inhibitory_or_excitatory == 'excitatory':
        #         self.synapses[ii]._inh = 1
        #     elif self.synapses[ii].inhibitory_or_excitatory == 'inhibitory':
        #         self.synapses[ii]._inh = -1
        #     _I_sy = 1e6*self.synapses[ii].synaptic_bias_current
            
        #     mutual_inductance = self.input_synaptic_inductances[ii][1]*np.sqrt(self.input_synaptic_inductances[ii][0]*self.synapses[ii].integration_loop_output_inductance)
        #     self.synapses[ii].coupling_factor = mutual_inductance/self.receiving_loop_total_inductance
        #     self.synapses[ii].I_si = np.zeros([len(time_vec),1])
        #     if ii == rs_index:
        #         self.synapses[ii].I_si_sat = I_si_sat__rs_loop
        #     else:
        #         self.synapses[ii].I_si_sat = I_si_sat__nom                
        #     if hasattr(self.synapses[ii],'input_signal'):
        #         self.synapses[ii].input_spike_times = self.synapses[ii].input_signal.spike_times+num_dt_pre*dt
        #     else:
        #         self.synapses[ii].input_spike_times = []            
        #     self.synapses[ii].spike_vec = np.zeros([len(time_vec),1])
        #     for jj in range(len(self.synapses[ii].input_spike_times)):
        #         spike_ind = (np.abs(np.asarray(time_vec)-self.synapses[ii].input_spike_times[jj])).argmin()
        #         self.synapses[ii].spike_vec[spike_ind] = 1
        
        # self.cell_body_circulating_current = self.thresholding_junction_bias_current*np.ones([len(time_vec),1])
        # self.state = 'sub_threshold'
        # self.spike_vec = np.zeros([len(time_vec),1])
        # self.spike_times = []#np.array([]) #list of real numbers (times cell_body_circulating_current crossed threshold current with positive derivative; the main dynamical variable and output of the neuron)        
        # self.t_obs = t_obs
        # self.time_vec = time_vec
        # for ii in range(len(time_vec)):
        #     for jj in range(len(self.synapses)):                       
        #         self.synapses[jj].I_si[ii] = synaptic_time_stepper(time_vec,ii,self.synapses[jj].input_spike_times,self.synapses[jj].I_0,self.synapses[jj].I_si_sat,gamma1,gamma2,gamma3,self.synapses[jj].tau_rise,self.synapses[jj].integration_loop_time_constant)
        #         self.cell_body_circulating_current[ii] += self.synapses[jj]._inh*self.synapses[jj].coupling_factor*self.synapses[jj].I_si[ii]
        #     if ii > 0:
        #         if (self.cell_body_circulating_current[ii] > self.thresholding_junction_critical_current 
        #             and self.cell_body_circulating_current[ii-1] < self.thresholding_junction_critical_current 
        #             and self.state == 'sub_threshold'):
        #             self.state = 'spiking'
        #             self.spike_times.append(time_vec[ii])
        #             self.spike_vec[ii] = 1
        #             self.synapses[rs_index].input_spike_times.append(time_vec[ii])
        #         if self.cell_body_circulating_current[ii] < self.thresholding_junction_critical_current:
        #             self.state = 'sub_threshold' 
                    
        #calculate output rate in various ways by looking at spikes in observation_duration
        # self.inter_spike_intervals = np.diff(self.spike_times)
        # idx_obs_start = (np.abs(time_vec-t_obs)).argmin()
        # idx_obs_end = (np.abs(time_vec-t_sim_total)).argmin()
        # self.num_spikes = sum(self.spike_vec[idx_obs_start:idx_obs_end+1])
        # for ii in range(len(self.synapses)):
        #     self.synapses[ii].num_spikes = sum(self.synapses[ii].spike_vec[idx_obs_start:idx_obs_end+1])
        # if len(self.spike_times) > 1:
        #     idx_avg_start = (np.abs(np.asarray(self.spike_times)-t_obs)).argmin()
        #     idx_avg_end = (np.abs(np.asarray(self.spike_times)-t_sim_total)).argmin()            
        #     self.isi_output__last_two = self.spike_times[-1]-self.spike_times[-2]
        #     self.isi_output__avg = np.mean(self.inter_spike_intervals[idx_avg_start:idx_avg_end])
        
        # self.idx_obs_start = idx_obs_start
        
        return self
    
    def save_neuron_data(self,save_string):
        
        tt = time.time()     
        s_str = 'neuron_data__'+save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.dat'
        self.save_string = s_str
        
        with open('data/'+s_str, 'wb') as data_file:
            pickle.dump(self, data_file)
  
        return self
        
        #neuron.is_output_neuron = True or False (whether or not the neuron communicates to the outside world)

# class network:
#     network.unique_label = 'net'+int
#     network.neurons = {unique_label} (list of neuron labels)
#     network.networks = {unique_label} (list of sub-network labels)
#     network.num_neurons = int (N_tot, total number of neurons in network)
#     network.adjacency_matrix = A (N_tot x N_tot adjacency matrix of the network. can be obtained from list of all neuron synaptic connections)
#     network.graph_metrics.{degree_distribution, clustering_coefficient, avg_path_length, etc}





