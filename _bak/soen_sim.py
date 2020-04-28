import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import time
import pickle

from _functions import synaptic_response_function, synapse_time_stepper, dendritic_drive__piecewise_linear, dendritic_time_stepper, Ljj, dendritic_drive__square_pulse_train, dendritic_drive__exp_pls_train__LR, dendrite_time_stepper
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
            if kwargs['synaptic_bias_current'] < 22e-6 or kwargs['synaptic_bias_current'] > 40e-6:
                raise ValueError('[soens_sim] synaptic_bias_current associated with synaptic integration loop must be a real number between 22e-6 and 40e-6 (units of amps)')
            else:
                 self.synaptic_bias_current = kwargs['synaptic_bias_current']
        else:
            _synaptic_bias_current_default = 28e-6 #units of amps
            self.synaptic_bias_current = _synaptic_bias_current_default

        if 'integration_loop_bias_current' in kwargs:
            # if type(kwargs['loop_bias_current']) == int or type(kwargs['loop_bias_current']) == float:
            if kwargs['integration_loop_bias_current'] < 0:
                raise ValueError('[soens_sim] loop_bias_current associated with synaptic integration loop must be a real number between xx and yy (units of amps)')
            else:
                 self.integration_loop_bias_current = kwargs['integration_loop_bias_current']
        else:
            self.integration_loop_bias_current = 30e-6 #units of amps
            
        if 'synapse_model_params' in kwargs:
            self.synapse_model_params = kwargs['synapse_model_params']
            sim_params = self.synapse_model_params            
        else:
            sim_params = dict()
            sim_params['dt'] = 1e-9 # units of seconds
            sim_params['tf'] = 1e-6 # units of seconds
            self.synapse_model_params = sim_params
            
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
        
    def run_sim(self):

        sim_params = self.synapse_model_params
        tf = sim_params['tf']
        dt = sim_params['dt']
        time_vec = np.arange(0,tf+dt,dt)
        p = physical_constants()
        # input_spike_times = self.input_spike_times
        
        #setup input signal
        if hasattr(self,'input_signal'):
            self.input_spike_times = self.input_signal.spike_times
        else:
            self.input_spike_times = []
                
        #here currents are in uA. they are converted to A before passing back
        I_sy = self.synaptic_bias_current*1e6

        #these values obtained by fitting to spice simulations
        #see matlab scripts in a4/calculations/nC/phenomenological_modeling...
        if 'gamma1' in sim_params:
            gamma1 = sim_params['gamma1']
        else:
            gamma1 = 0.9
        if 'gamma2' in sim_params:
            gamma2 = sim_params['gamma2']
        else:
            gamma2 = 0.158
        if 'gamma3' in sim_params:
            gamma3 = sim_params['gamma3']
        else:
            gamma3 = 3/4

        #these fits were obtained by comparing to spice simulations
        #see matlab scripts in a4/calculations/nC/phenomenological_modeling...
        
        # tau_rise = (1.294*I_sy-43.01)*1e-9
        tau_rise = (0.038359*I_sy**2-0.778850*I_sy-0.441682)*1e-9
        self.integration_loop_total_inductance = self.integration_loop_self_inductance+self.integration_loop_output_inductance
        _reference_inductance = 775e-9 #inductance at which I_0 fit was performed
        _scale_factor = _reference_inductance/self.integration_loop_total_inductance
        #I_0 = (0.06989*I_sy**2-3.948*I_sy+53.73)*_scale_factor #from earlier model assuming 10uA spd
        
        # I_0 = (0.006024*I_sy**2-0.202821*I_sy+1.555543)*_scale_factor # in terms of currents
        I_0 = 1e6*(2.257804*I_sy**2-76.01606*I_sy+583.005805)*p['Phi0']/self.integration_loop_total_inductance # in terms of n_fq
        
        #I_si_sat is actually a function of I_b (loop_current_bias). The fit I_si_sat(I_b) has not yet been performed (20200319)
        I_si_sat = 19.7

        tau_fall = self.integration_loop_time_constant

        # I_si_vec = synaptic_response_function(time_vec,input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall)
        I_si_vec = np.zeros([len(time_vec),1])
        for ii in range(len(time_vec)):
            I_si_vec[ii] = synapse_time_stepper(time_vec,ii,self.input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall)

        self.I_si = I_si_vec*1e-6
        self.time_vec = time_vec

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
                                    
        if 'input_direct_connections' in kwargs:
            self.input_direct_connections = kwargs['input_direct_connections']            
        else:
            self.input_direct_connections = []             

        if 'input_direct_inductances' in kwargs:
            if type(kwargs['input_direct_inductances']) == list:
                    self.input_direct_inductances = kwargs['input_direct_inductances']
            else:
                raise ValueError('[soens_sim] Input direct inductances to dendrites are specified as a list of pairs of real numbers with one pair per direct connection. The first element of the pair is the inductance on the dendritic receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the input direct signal and the dendritic receiving loop.')
        else:
            self.input_direct_inductances =  [[]] 
                    
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
            
        if 'integration_loop_saturation_current' in kwargs:
            # if type(kwargs['integration_loop_self_inductance']) == int or type(kwargs['integration_loop_self_inductance']) == float:
            if kwargs['integration_loop_saturation_current'] < 0:
                raise ValueError('[soens_sim] Integration loop saturation current associated with dendritic integration loop must be a real number between zero and infinity (units of amps)')
            else:
                 self.integration_loop_saturation_current = kwargs['integration_loop_saturation_current']
        else: 
            self.integration_loop_saturation_current = 13e-6 #default value, units of amps
                        
        if 'integration_loop_output_inductance' in kwargs:
            # if type(kwargs['integration_loop_output_inductance']) == int or type(kwargs['integration_loop_output_inductance']) == float:
            if kwargs['integration_loop_output_inductance'] < 0:
                raise ValueError('[soens_sim] Integration loop output inductance associated with coupling between synaptic integration loop and dendrite or neuron must be a real number between zero and infinity (units of henries)')
            else:
                 self.integration_loop_output_inductance = kwargs['integration_loop_output_inductance']
        else: 
            self.integration_loop_output_inductance = 200e-12 #default value, units of henries
        self.integration_loop_total_inductance = self.integration_loop_self_inductance+self.integration_loop_output_inductance
        
        if 'integration_loop_temporal_form' in kwargs:
            if kwargs['integration_loop_temporal_form'] == 'exponential' or kwargs['loop_temporal_form'] == 'power_law':
                _temporal_form = kwargs['integration_loop_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid integration loop temporal form to synapse %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.name, self.unique_label))
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
            sim_params['amp'] = 9.27586207
            sim_params['mu1'] = 1.92758621
            sim_params['mu2'] = 0.45344828
            sim_params['mu3'] = 0.87959184
            sim_params['mu4'] = 0.59591837
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
        
        # calculate receiving loop inductance
        # self.receiving_loop_total_inductance = self.circuit_inductances[0]+self.circuit_inductances[1]
        
        # for ii in range(len(self.synapses)):
        #     self.receiving_loop_total_inductance += self.input_synaptic_inductances[ii][0]
        
        # for ii in range(len(self.dendrites)):
        #     self.receiving_loop_total_inductance += self.input_dendritic_inductances[ii][0]
                            
        # for ii in range(len(self.direct_connections)):
        #     self.receiving_loop_total_inductance += self.input_direct_inductances[ii][0]
            
        self.time_vec = time_vec
        
        if hasattr(self.direct_connections[0],'piecewise_linear'):
            dendritic_drive = dendritic_drive__piecewise_linear(time_vec,self.direct_connections[0].piecewise_linear)
        if hasattr(self.direct_connections[0],'square_pulse_train'):
            dendritic_drive = dendritic_drive__square_pulse_train(time_vec,self.direct_connections[0].square_pulse_train)
        if hasattr(self.direct_connections[0],'exponential_pulse_train'):
            dendritic_drive = dendritic_drive__exp_pls_train__LR(time_vec,self.direct_connections[0].exponential_pulse_train)
        # if hasattr(self.direct_connections[0],'slope'):
        #     dendritic_drive = dendritic_drive__linear_ramp(time_vec, time_on = self.direct_connections[0].time_on, slope = self.direct_connections[0].slope)
        # dendritic_drive = dendritic_drive__step_function(time_vec, amplitude = self.direct_connections[0].amplitude, time_on = self.direct_connections[0].time_on)
        # 
        # plot_dendritic_drive(time_vec, dendritic_drive)
        self.dendritic_drive = dendritic_drive
        
        # I_b = self.bias_currents
        # Ic = self.thresholding_junction_critical_current
        # I_di_sat = self.integration_loop_saturation_current
        tau_di = self.integration_loop_time_constant
        # mu_1 = self.dendrite_model_params['mu1']
        # mu_2 = self.dendrite_model_params['mu2']
        # mu_3 = self.dendrite_model_params['mu3']
        # mu_4 = self.dendrite_model_params['mu4']
        # print('mu1 = {}'.format(mu_1))
        # print('mu2 = {}'.format(mu_2))
        # print('mu3 = {}'.format(mu_3))
        # print('mu4 = {}'.format(mu_4))
        # M_direct = self.input_direct_inductances[0][1]*np.sqrt(self.input_direct_inductances[0][0]*self.direct_connections[0].output_inductance)
        # Lm2 = self.input_direct_inductances[0][0]
        # Ldr1 = self.circuit_inductances[0]
        # Ldr2 = self.circuit_inductances[1]
        # L1 = self.circuit_inductances[2]
        # L2 = self.circuit_inductances[3]
        L3 = self.integration_loop_self_inductance+self.integration_loop_output_inductance
        # L_reference = 10e-6
        # A_prefactor = self.dendrite_model_params['amp']*L_reference/L3 #9.27586207*L_reference/L3#self.sim_params['A']*L_reference/L3 #
        # print(A_prefactor)
        # tau_di = self.integration_loop_time_constant
        #I_di_vec = dendritic_time_stepper_old2(time_vec,A_prefactor,dendritic_drive,I_b,I_th,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,tau_di,mu_1,mu_2,mu_3,mu_4)
        # R = 4.125
        # I_di_vec = dendritic_time_stepper(time_vec,R,dendritic_drive,I_b,Ic,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,tau_di,mu_1,mu_2)
        # dendrite_time_stepper(time_vec,I_drive,L3,tau_di)
        I_di_vec = dendrite_time_stepper(time_vec,dendritic_drive,L3,tau_di)
        
        self.I_di = I_di_vec        
        
        return self

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
        refractory_loop = synapse(self.name+'__refractory_suppression_synapse', 
                                  inhibitory_or_excitatory = 'inhibitory',
                                  integration_loop_temporal_form = 'exponential', 
                                  integration_loop_time_constant = self.refractory_time_constant,
                                  integration_loop_self_inductance = self.refractory_loop_self_inductance,
                                  integration_loop_output_inductance = self.refractory_loop_output_inductance,
                                  synaptic_bias_current = self.refractory_synaptic_bias_current,
                                  integration_loop_bias_current = refractory_loop_bias_current)
        
        refractory_loop.unique_label = self.unique_label+'_rs'
        refractory_loop.input_spike_times = []
        neuron.refractory_loop = refractory_loop
        self.input_synaptic_connections.append(refractory_loop.name)        
                
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
            #then add direct connections to dendrites
            for name_3 in dendrite.dendrites[name_1].input_direct_connections:
                dendrite.dendrites[name_1].direct_connections.append(input_signal.input_signals[name_3])
            #then add dendrites to neuron
            self.dendrites.append(dendrite.dendrites[name_1])
        
        #then add synapses to neuron
        self.synapses = []
        for name in self.input_synaptic_connections:             
            self.synapses.append(synapse.synapses[name])
            
        #then add direct connections to neuron
            
        return self
    
    def configure_synapses(self):
        
        return self
        
    def configure_dendrites(self):
        
        return self    
        
    def run_sim(self):
        
        # set up time vec
        num_dt_pre = 10
        dt = self.sim_params['dt']
        t_obs = dt*np.round((self.sim_params['pre_observation_duration']+num_dt_pre*dt)/dt,decimals = 0)
        t_sim_total = dt*np.round((t_obs+self.sim_params['observation_duration'])/dt,decimals = 0)
        time_vec = np.arange(0,t_sim_total+dt,dt)           

        # attach synapses, dendrites, and direct connections to neuron
        self.make_connections()        
        # print('simulating neuron with {} synapses\n\n'.format(len(self.synapses)))
        
        # calculate receiving loop inductance
        self.receiving_loop_total_inductance = self.receiving_loop_self_inductance
        
        for ii in range(len(self.synapses)):
            # print(self.synapses[ii].unique_label)
            self.receiving_loop_total_inductance += self.input_synaptic_inductances[ii][0]
            
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
                          
        #find index of refractory suppresion loop
        for ii in range(len(self.synapses)):
            if self.synapses[ii].unique_label == self.unique_label+'_rs':
                rs_index = ii
                
        for ii in range(len(self.synapses)):
            if self.synapses[ii].inhibitory_or_excitatory == 'excitatory':
                self.synapses[ii]._inh = 1
            elif self.synapses[ii].inhibitory_or_excitatory == 'inhibitory':
                self.synapses[ii]._inh = -1
            _I_sy = 1e6*self.synapses[ii].synaptic_bias_current
            self.synapses[ii].tau_rise = (1.294*_I_sy-43.01)*1e-9
            _scale_factor = _reference_inductance/self.synapses[ii].integration_loop_total_inductance
            self.synapses[ii].I_0 = (0.06989*_I_sy**2-3.948*_I_sy+53.73)*_scale_factor
            mutual_inductance = self.input_synaptic_inductances[ii][1]*np.sqrt(self.input_synaptic_inductances[ii][0]*self.synapses[ii].integration_loop_output_inductance)
            self.synapses[ii].coupling_factor = mutual_inductance/self.receiving_loop_total_inductance
            self.synapses[ii].I_si = np.zeros([len(time_vec),1])
            if ii == rs_index:
                self.synapses[ii].I_si_sat = I_si_sat__rs_loop
            else:
                self.synapses[ii].I_si_sat = I_si_sat__nom                
            if hasattr(self.synapses[ii],'input_signal'):
                self.synapses[ii].input_spike_times = self.synapses[ii].input_signal.spike_times+num_dt_pre*dt
            else:
                self.synapses[ii].input_spike_times = []            
            self.synapses[ii].spike_vec = np.zeros([len(time_vec),1])
            for jj in range(len(self.synapses[ii].input_spike_times)):
                spike_ind = (np.abs(np.asarray(time_vec)-self.synapses[ii].input_spike_times[jj])).argmin()
                self.synapses[ii].spike_vec[spike_ind] = 1
        
        self.cell_body_circulating_current = self.thresholding_junction_bias_current*np.ones([len(time_vec),1])
        self.state = 'sub_threshold'
        self.spike_vec = np.zeros([len(time_vec),1])
        self.spike_times = []#np.array([]) #list of real numbers (times cell_body_circulating_current crossed threshold current with positive derivative; the main dynamical variable and output of the neuron)        
        self.t_obs = t_obs
        self.time_vec = time_vec
        for ii in range(len(time_vec)):
            for jj in range(len(self.synapses)):                       
                self.synapses[jj].I_si[ii] = synaptic_time_stepper(time_vec,ii,self.synapses[jj].input_spike_times,self.synapses[jj].I_0,self.synapses[jj].I_si_sat,gamma1,gamma2,gamma3,self.synapses[jj].tau_rise,self.synapses[jj].integration_loop_time_constant)
                self.cell_body_circulating_current[ii] += self.synapses[jj]._inh*self.synapses[jj].coupling_factor*self.synapses[jj].I_si[ii]
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





