import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import time
import pickle

from _functions import synaptic_response_function, synaptic_time_stepper

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
                kwargs['input_temporal_form'] == 'arbitrary_spike_train'):
                _temporal_form = kwargs['input_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid input signal temporal form to input %s (unique_label = %s)\nThe allowed values of input_temporal_form are ''single_spike'', ''constant_rate'', and ''arbitrary_spike_train''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'single_spike'
        self.input_temporal_form =  _temporal_form #'single_pulse' by default
        
        if 'spike_times' in kwargs:
            if (self.input_temporal_form == 'single_spike' or self.input_temporal_form == 'arbitrary_spike_train'):
                self.spike_times = kwargs['spike_times']
            elif self.input_temporal_form == 'constant_rate': # in this case, pulse_times has the form [rate,time_of_last_spike]
                isi = 1/kwargs['spike_times'][0]
                self.spike_times = np.arange(0,kwargs['spike_times'][1],isi)
                
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
                _temporal_form = kwargs['loop_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid integration loop temporal form to synapse %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'exponential'
        self.integration_loop_temporal_form =  _temporal_form #'exponential' or 'power_law'; 'exponential' by default

        if 'integration_loop_time_constant' in kwargs:
            # print(type(kwargs['time_constant']))
            # print(kwargs['time_constant'])
            # if type(kwargs['time_constant']) == int or type(kwargs['time_constant']) == float or type(kwargs['time_constant']) == np.float64:
            if kwargs['integration_loop_time_constant'] < 0:
                raise ValueError('[soens_sim] time_constant associated with synaptic integration loop decay must be a real number between zero and infinity')
            else:
                self.integration_loop_time_constant = kwargs['integration_loop_time_constant']
        else:
            if self.integration_loop_temporal_form == 'exponential':
                self.integration_loop_time_constant = 200e-9 #default time constant units of seconds
        
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

        if 'synaptic_bias_current' in kwargs:
            # if type(kwargs['synaptic_bias_current']) == int or type(kwargs['synaptic_bias_current']) == float or type(kwargs['synaptic_bias_current']) == np.float64:
            if kwargs['synaptic_bias_current'] < 34e-6 or kwargs['synaptic_bias_current'] > 100e-6:
                raise ValueError('[soens_sim] synaptic_bias_current associated with synaptic integration loop must be a real number between 34e-6 and 39e-6 (units of amps)')
            else:
                 self.synaptic_bias_current = kwargs['synaptic_bias_current']
        else:
            _synaptic_bias_current_default = 35e-6 #units of amps
            self.synaptic_bias_current = _synaptic_bias_current_default

        if 'integration_loop_bias_current' in kwargs:
            # if type(kwargs['loop_bias_current']) == int or type(kwargs['loop_bias_current']) == float:
            if kwargs['integration_loop_bias_current'] < 0:
                raise ValueError('[soens_sim] loop_bias_current associated with synaptic integration loop must be a real number between xx and yy (units of amps)')
            else:
                 self.integration_loop_bias_current = kwargs['integration_loop_bias_current']
        else:
            self.integration_loop_bias_current = 30e-6 #units of amps
            
        if 'input_signal_name' in kwargs:
            if kwargs['input_signal_name'] != '':
                self.input_signal_name = kwargs['input_signal_name']
            
        # attach external input signal to synapse
        if hasattr(self, 'input_signal_name'):
            self.input_signal = input_signal.input_signals[self.input_signal_name]
            # print('added input signal with name {} to synapse {}'.format(self.input_signal_name,self.name))                               
        
        synapse.synapses[self.name] = self
                
        # print('synapse created')
        
        return

    def __del__(self):
        # print('synapse deleted')
        return
        
    def run_sim(self,time_vec):

        input_spike_times = self.input_spike_times
        
        #here currents are in uA. they are converted to A before passing back
        I_sy = self.synaptic_bias_current*1e6

        #these values obtained by fitting to spice simulations
        #see matlab scripts in a4/calculations/nC/phenomenological_modeling...
        gamma1 = 0.9
        gamma2 = 0.158
        gamma3 = 3/4

        #these fits were obtained by comparing to spice simulations
        #see matlab scripts in a4/calculations/nC/phenomenological_modeling...
        tau_rise = (1.294*I_sy-43.01)*1e-9
        self.integration_loop_total_inductance = self.integration_loop_self_inductance+self.integration_loop_output_inductance
        _reference_inductance = 50e-9 #inductance at which I_0 fit was performed
        _scale_factor = _reference_inductance/self.integration_loop_total_inductance
        I_0 = (0.06989*I_sy**2-3.948*I_sy+53.73)*_scale_factor
        
        #I_si_sat is actually a function of I_b (loop_current_bias). The fit I_si_sat(I_b) has not yet been performed (20200319)
        I_si_sat = 13

        tau_fall = self.time_constant

        # I_si_vec = synaptic_response_function(time_vec,input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall)
        I_si_vec = np.zeros([len(time_vec),1])
        for ii in range(len(time_vec)):
            I_si_vec[ii] = synaptic_time_stepper(time_vec,ii,input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall)

        self.I_si = I_si_vec*1e-6

        return self
    

class dendrite():
    _next_uid = 0

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
                    self.input_synaptic_inductances = kwargs['input_synaptic_inductances']
            else:
                raise ValueError('[soens_sim] Input synaptic inductances to dendrites are specified as a list of pairs of real numbers with one pair per synaptic connection. The first element of the pair is the inductance on the dendritic receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the synapse and the dendritic receiving loop.')
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
                raise ValueError('[soens_sim] Input dendritic inductances to dendrites are specified as a list of pairs of real numbers with one pair per dendritic connection. The first element of the pair is the inductance on the dendritic receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the input dendrite and the dendritic receiving loop.')
        else:
            self.input_dendritic_inductances =  [[]]            
                    
        if 'receiving_loop_self_inductance' in kwargs:
            if kwargs['receiving_loop_self_inductance'] > 0:
                self.receiving_loop_self_inductance = kwargs['receiving_loop_self_inductance']
            else:
                raise ValueError('[soens_sim] Receiving loop self inductance is a real number greater than zero with units of henries. This includes the total inductance of the dendritic receiving loop excluding any input mutual inductors.')
        else:
            self.receiving_loop_self_inductance = 20e-12            

        if 'thresholding_junction_critical_current' in kwargs:
            _Ic = kwargs['thresholding_junction_critical_current']
            # else:
                # raise ValueError('[soens_sim] Thresholding junction critical current must be a real number with units of amps')
        else:
            _Ic = 40e-6 #default J_th Ic = 40 uA
        self.thresholding_junction_critical_current =  _Ic
            
        if 'thresholding_junction_bias_current' in kwargs:
            _Ib = kwargs['thresholding_junction_bias_current']
        else:
            _Ib = 35e-6 #default J_th Ic = 40 uA
        self.thresholding_junction_bias_current =  _Ib
            
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

        dendrite.dendrites[self.name] = self
        
        if 'integration_loop_temporal_form' in kwargs:
            if kwargs['loop_temporal_form'] == 'exponential' or kwargs['loop_temporal_form'] == 'power_law':
                _temporal_form = kwargs['loop_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid loop temporal form to synapse %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'exponential'
        self.loop_temporal_form =  _temporal_form #'exponential' or 'power_law'; 'exponential' by default
        
        if 'integration_loop_time_constant' in kwargs:
            if kwargs['time_constant'] < 0:
                raise ValueError('[soens_sim] time_constant associated with dendritic decay must be a real number between zero and infinity')
            else:
                self.integration_loop_time_constant = kwargs['time_constant']
        else:
            if self.loop_temporal_form == 'exponential':
                self.integration_loop_time_constant = 200e-9 #default time constant units of seconds
             
        if 'integration_loop_power_law_exponent' in kwargs:
            if kwargs['integration_loop_power_law_exponent'] > 0:
                raise ValueError('[soens_sim] power_law_exponent associated with dendritic decay must be a real number between negative infinity and zero')
            else:
                 self.power_law_exponent = kwargs['integration_loop_power_law_exponent']
        else:
            if self.loop_temporal_form == 'power_law':                
                self.integration_loop_power_law_exponent = -1 #default power law exponent
                
        if 'integration_loop_bias_current' in kwargs:
            if kwargs['integration_loop_bias_current'] < 0:
                raise ValueError('[soens_sim] integration_loop_bias_current associated with dendritic integration loop must be a real number between xx and yy (units of amps)')
            else:
                 self.integration_loop_bias_current = kwargs['integration_loop_bias_current']
        else:            
            self.integration_loop_bias_current = 30e-6 #units of amps 
            
        # print('dendrite created')        
        return      

    def __del__(self):
        # print('dendrite deleted')
        return


class neuron():

    _next_uid = 0

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

        if 'receiving_loop_self_inductance' in kwargs:
            # if type(kwargs['receiving_loop_self_inductance']) == float:
            if kwargs['receiving_loop_self_inductance'] > 0:
                self.receiving_loop_self_inductance = kwargs['receiving_loop_self_inductance']
            else:
                raise ValueError('[soens_sim] Receiving loop self inductance is a real number greater than zero with units of henries. This includes the total inductance of the neuronal receiving loop excluding any input mutual inductors.')
        else:
            self.receiving_loop_self_inductance = 20e-12
            
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
                raise ValueError('[soens_sim] Input dendritic inductances to neurons are specified as a list of pairs of real numbers with one pair per dendritic connection. The first element of the pair is the inductance on the neuronal receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the dendrite and the neuronal receiving loop.')
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
            
        if 'thresholding_junction_bias_current' in kwargs:
            # if type(kwargs['threshold_bias_current']) == float:
            _Ib = kwargs['thresholding_junction_bias_current']
            # else:
                # raise ValueError('[soens_sim] Thresholding junction bias current must be a real number with units of amps')
        else:
            _Ib = 35e-6 #default J_th Ic = 40 uA
        self.thresholding_junction_bias_current =  _Ib
                
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
            
        if 'refractory_loop_synaptic_bias_current' in kwargs:
            self.refractory_synaptic_bias_current = kwargs['refractory_loop_synaptic_bias_current']
        else:
            self.refractory_synaptic_bias_current = 39e-6
            
        if 'refractory_loop_saturation_current' in kwargs:
            self.refractory_loop_saturation_current = kwargs['refractory_loop_saturation_current']
        else:
            self.refractory_loop_saturation_current = 100e-6
                 
        #set up refractory loop as synapse
        refractory_loop_bias_current = 31e-6 #doesn't matter now as synapse saturation is independent of bias; will matter with future updates, I hope
        refractory_loop = synapse(self.name+'__refractory_suppression_synapse', loop_temporal_form = 'exponential', 
                                  time_constant = self.refractory_time_constant,
                                  integration_loop_self_inductance = self.refractory_loop_self_inductance,
                                  integration_loop_output_inductance = self.refractory_loop_output_inductance,
                                  synaptic_bias_current = self.refractory_synaptic_bias_current,
                                  loop_bias_current = refractory_loop_bias_current)
        
        refractory_loop.unique_label = self.unique_label+'_rs'
        refractory_loop.input_spike_times = []
        neuron.refractory_loop = refractory_loop
        self.input_connections.append(refractory_loop.name)        
                
        # print('neuron created')
        
        return

    def __del__(self):
        # print('neuron deleted')
        return
    
    def make_connections(self):

        #add dendrites to neuron
        self.dendrites = []
        for name_1 in self.input_dendritic_connections:
            #first add synapses to dendrites
            for name_2 in dendrite.dendrites[name_1].input_synaptic_connections:
                dendrite.dendrites[name_1].synapses.append(synapse.synapses[name_2])
            #then add dendrites to dendrites
            for name_2 in dendrite.dendrites[name_1].input_dendritic_connections:
                dendrite.dendrites[name_1].dendrites.append(dendrite.dendrites[name_2])
            #then add dendrites to neuron
            self.dendrites.append(dendrite.dendrites[name_1])
        
        #then add synapses to neuron
        self.synapses = []
        for name in self.input_synaptic_connections:             
            self.synapses.append(synapse.synapses[name])
            
        return self
        
    def run_sim(self):
        
        # set up time vec
        num_dt_pre = 10
        dt = self.sim_params['dt']
        t_obs = dt*np.round((self.sim_params['pre_observation_duration']+num_dt_pre*dt)/dt,decimals = 0)
        t_sim_total = dt*np.round((t_obs+self.sim_params['observation_duration'])/dt,decimals = 0)
        time_vec = np.arange(0,t_sim_total+dt,dt)           

        # attach synapses to neuron
        self.make_connections()        
        # print('simulating neuron with {} synapses\n\n'.format(len(self.synapses)))
        
        # calculate receiving loop inductance
        self.receiving_loop_total_inductance = self.receiving_loop_self_inductance
        
        for ii in range(len(self.synapses)):
            # print(self.synapses[ii].unique_label)
            self.receiving_loop_total_inductance += self.input_inductances[ii][0]
            
        #info for synaptic response        
        #these values obtained by fitting to spice simulations
        #see matlab scripts in a4/calculations/nC/phenomenological_modeling...
        gamma1 = 0.9
        gamma2 = 0.158
        gamma3 = 3/4
        _reference_inductance = 50e-9 #inductance at which I_0 fit was performed
        #I_si_sat is actually a function of I_b (loop_current_bias). The fit I_si_sat(I_b) has not yet been performed (20200319)
        I_si_sat__nom = 13
        I_si_sat__rs_loop = self.refractory_loop_saturation_current*1e6

        for ii in range(len(self.synapses)):
            _I_sy = 1e6*self.synapses[ii].synaptic_bias_current
            self.synapses[ii].tau_rise = (1.294*_I_sy-43.01)*1e-9
            _scale_factor = _reference_inductance/self.synapses[ii].integration_loop_total_inductance
            self.synapses[ii].I_0 = (0.06989*_I_sy**2-3.948*_I_sy+53.73)*_scale_factor
            mutual_inductance = self.input_synaptic_inductances[ii][1]*np.sqrt(self.input_synaptic_inductances[ii][0]*self.synapses[ii].integration_loop_output_inductance)
            self.synapses[ii].coupling_factor = mutual_inductance/self.receiving_loop_total_inductance
            self.synapses[ii].I_si = np.zeros([len(time_vec),1])
            if hasattr(self.synapses[ii],'input_signal'):
                self.synapses[ii].input_spike_times = self.synapses[ii].input_signal.spike_times+num_dt_pre*dt
            else:
                self.synapses[ii].input_spike_times = []            
            self.synapses[ii].spike_vec = np.zeros([len(time_vec),1])
            for jj in range(len(self.synapses[ii].input_spike_times)):
                spike_ind = (np.abs(np.asarray(time_vec)-self.synapses[ii].input_spike_times[jj])).argmin()
                self.synapses[ii].spike_vec[spike_ind] = 1
                          
        #find index of refractory suppresion loop
        for ii in range(len(self.synapses)):
            if self.synapses[ii].unique_label == self.unique_label+'_rs':
                rs_index = ii
        
        self.cell_body_circulating_current = self.threshold_bias_current*np.ones([len(time_vec),1])
        self.state = 'sub_threshold'
        self.spike_vec = np.zeros([len(time_vec),1])
        self.spike_times = []#np.array([]) #list of real numbers (times cell_body_circulating_current crossed threshold current with positive derivative; the main dynamical variable and output of the neuron)        
        self.t_obs = t_obs
        self.time_vec = time_vec
        for ii in range(len(time_vec)):
            for jj in range(len(self.synapses)):
                if jj == rs_index:
                    I_si_sat = I_si_sat__rs_loop
                    _inh = -1
                else:
                    I_si_sat = I_si_sat__nom
                    _inh = 1                       
                self.synapses[jj].I_si[ii] = synaptic_time_stepper(time_vec,ii,self.synapses[jj].input_spike_times,self.synapses[jj].I_0,I_si_sat,gamma1,gamma2,gamma3,self.synapses[jj].tau_rise,self.synapses[jj].time_constant)
                self.cell_body_circulating_current[ii] += _inh*self.synapses[jj].coupling_factor*self.synapses[jj].I_si[ii]
            if ii > 0:
                if (self.cell_body_circulating_current[ii] > self.thresholding_junction_critical_current 
                    and self.cell_body_circulating_current[ii-1] < self.thresholding_junction_critical_current 
                    and self.state == 'sub_threshold'):
                    self.state = 'spiking'
                    self.spike_times.append(time_vec[ii])
                    self.spike_vec[ii] = 1
                    self.synapses[rs_index].input_spike_times.append(time_vec[ii])
                if self.cell_body_circulating_current[ii] < self.thresholding_junction_critical_current:
                    self.state = 'sub_threshold' 
                    
        #calculate output rate in various ways by looking at spikes in observation_duration
        self.inter_spike_intervals = np.diff(self.spike_times)
        idx_obs_start = (np.abs(time_vec-t_obs)).argmin()
        idx_obs_end = (np.abs(time_vec-t_sim_total)).argmin()
        self.num_spikes = sum(self.spike_vec[idx_obs_start:idx_obs_end+1])
        for ii in range(len(self.synapses)):
            self.synapses[ii].num_spikes = sum(self.synapses[ii].spike_vec[idx_obs_start:idx_obs_end+1])
        if len(self.spike_times) > 1:
            idx_avg_start = (np.abs(np.asarray(self.spike_times)-t_obs)).argmin()
            idx_avg_end = (np.abs(np.asarray(self.spike_times)-t_sim_total)).argmin()            
            self.isi_output__last_two = self.spike_times[-1]-self.spike_times[-2]
            self.isi_output__avg = np.mean(self.inter_spike_intervals[idx_avg_start:idx_avg_end])
        
        self.idx_obs_start = idx_obs_start
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





