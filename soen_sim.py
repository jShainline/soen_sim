import numpy as np
# from pylab import *
import time
import pickle
import copy
# from scipy.signal import find_peaks

from _functions import neuron_time_stepper, dendritic_drive__piecewise_linear, dendritic_drive__square_pulse_train, dendritic_drive__exp_pls_train__LR, Ljj_pH
# from util import physical_constants

class input_signal():
    
    _next_uid = 0
    input_signals = dict()
    
    def __init__(self, **kwargs):
        
        #make new input signal
        self.uid = input_signal._next_uid
        input_signal._next_uid += 1
        self.unique_label = 'in'+str(self.uid)
        
        # name the synapse
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'unnamed_input_signal__{}'.format(self.unique_label)
        # end name 
        
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
                raise ValueError('[soens_sim] Tried to assign an invalid stochasticity type to input {} (unique label {}). Available stochasticity forms are presently: ''gaussian'' or ''none'''.format(self.name, self.unique_label))
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
                self.output_inductance = 200 # pH
            if 'time_vec' in kwargs:
                self.time_vec = kwargs['time_vec']
            else:
                dt = 5 # ns
                tf = 20 # ns
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
    
    def __init__(self, **kwargs):

        #make new synapse
        # self._instances.add(weakref.ref(self))
        self.uid = synapse._next_uid
        synapse._next_uid += 1
        self.unique_label = 's{}'.format(self.uid)

        # name the synapse
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'unnamed_synapse__{}'.format(self.unique_label)
        # end name  
        
        # synaptic receiver spd circuit specification
        if 'synaptic_circuit_inductors' in kwargs:
            if type(kwargs['synaptic_circuit_inductors']) == list:
                if len(kwargs['synaptic_circuit_inductors']) == 3:
                    self.synaptic_circuit_inductors = kwargs['synaptic_circuit_inductors']
                else: ValueError('[soens_sim] len(synaptic_circuit_inductors) must be three (see circuit diagram in documentation)')
            else:
                raise ValueError('[soens_sim] synaptic_circuit_inductors must be a list of length three')
        else:
            self.synaptic_circuit_inductors = [100e3,100e3,400] # three inductors with units of picohenries
        
        if 'synaptic_circuit_resistors' in kwargs:
            if type(kwargs['synaptic_circuit_resistors']) == list:
                if len(kwargs['synaptic_circuit_resistors']) == 2:
                    self.synaptic_circuit_resistors = kwargs['synaptic_circuit_resistors']
                else: ValueError('[soens_sim] len(synaptic_circuit_resistors) must be two (see circuit diagram in documentation)')
            else:
                raise ValueError('[soens_sim] synaptic_circuit_resistors must be a list of length two')
        else:
            self.synaptic_circuit_resistors = [5e6,5e3] # three resistors with units of mOhms (pH/ns)
        
        if 'synaptic_hotspot_duration' in kwargs:
            self.synaptic_hotspot_duration = kwargs['synaptic_hotspot_duration']
        else:
            self.synaptic_hotspot_duration = 0.2 # real number with units of nanoeconds
        
        if 'synaptic_spd_current' in kwargs:
            self.synaptic_spd_current = kwargs['synaptic_spd_current']
        else:
            self.synaptic_spd_current = 10 # real number with units of microamps   
        # end synaptic receiver spd circuit specification
        
        # input signals                
        if 'input_direct_connections' in kwargs:
            self.input_direct_connections = kwargs['input_direct_connections']
        else:
            self.input_direct_connections = []
        for connection_name in self.input_direct_connections:
            # print('{}'.format(connection_name))
            self.input_signal = input_signal.input_signals[connection_name]
        
        if 'input_neuronal_connections' in kwargs:
            self.input_neuronal_connections = kwargs['input_neuronal_connections']
        else:
            self.input_neuronal_connections = []
        # end input signals
        
        # synaptic dendrite specification        
        if 'num_jjs' in kwargs:
            if kwargs['num_jjs'] == 2 or kwargs['num_jjs'] == 4:
                self.num_jjs = kwargs['num_jjs']
            else:
                raise ValueError('[soens_sim] num_jjs must be 2 or 4')
        else:
            self.num_jjs = 4
        
        if 'inhibitory_or_excitatory' in kwargs:
            if kwargs['inhibitory_or_excitatory'] == 'inhibitory' or kwargs['inhibitory_or_excitatory'] == 'excitatory':
                _i_or_e = kwargs['inhibitory_or_excitatory']
            else:
                raise ValueError('[soens_sim] inhibitory_or_excitatory can either be ''inhibitory'' and ''excitatory''')
        else:
            _i_or_e = 'excitatory' # 'excitatory' by default
        self.inhibitory_or_excitatory =  _i_or_e #'excitatory' by default
                            
        if 'synaptic_dendrite_circuit_inductances' in kwargs:
            if type(kwargs['synaptic_dendrite_circuit_inductances']) == list:
                self.synaptic_dendrite_circuit_inductances = kwargs['synaptic_dendrite_circuit_inductances']            
        else:
            self.synaptic_dendrite_circuit_inductances = [20, 20, 200, 77.5]            
        
        if 'synaptic_dendrite_input_synaptic_inductance' in kwargs:
            self.synaptic_dendrite_input_synaptic_inductance = kwargs['synaptic_dendrite_input_synaptic_inductance']
        else:
            self.synaptic_dendrite_input_synaptic_inductance =  [20,0.5] # [inductance (units of picohenries), mutual inductance efficiency (k)]
            
        if 'junction_critical_current' in kwargs:
            self.junction_critical_current =  kwargs['junction_critical_current']
        else:
            self.junction_critical_current =  40 #default Ic = 40 uA
            
        if 'bias_currents' in kwargs:
            self.bias_currents = kwargs['bias_currents']
        else:
            self.bias_currents = [72, 36, 35] #[bias to DR loop (J_th), bias to JTL, bias to DI loop]        
            
        if 'integration_loop_self_inductance' in kwargs:
            # if type(kwargs['integration_loop_self_inductance']) == int or type(kwargs['integration_loop_self_inductance']) == float:
            if kwargs['integration_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Integration loop self inductance associated with dendritic integration loop must be a real number between zero and infinity (units of picohenries)')
            else:
                 self.integration_loop_self_inductance = kwargs['integration_loop_self_inductance']
        else: 
            self.integration_loop_self_inductance = 77.5e3 #default value, units of picohenries
                        
        if 'integration_loop_output_inductance' in kwargs:
            if type(kwargs['integration_loop_output_inductance']) != list:
                self.integration_loop_output_inductance = kwargs['integration_loop_output_inductance']
                ti = self.integration_loop_output_inductance
            if type(kwargs['integration_loop_output_inductance']) == list:
                self.integration_loop_output_inductance = kwargs['integration_loop_output_inductance']
                ti = sum(self.integration_loop_output_inductance)
        else: 
            self.integration_loop_output_inductance = 200 #default value, units of pH
        self.integration_loop_total_inductance = self.integration_loop_self_inductance+ti
        
        if 'integration_loop_time_constant' in kwargs:
            self.integration_loop_time_constant = kwargs['integration_loop_time_constant']
        else:
            self.integration_loop_time_constant = 250 #default time constant units of ns
        # end synaptic dendrite specification   
        
        # make synaptic dendrite
        synapse_dendrite = dendrite(name = '{}__d'.format(self.name),
                                   num_jjs = self.num_jjs,
                                   inhibitory_or_excitatory = self.inhibitory_or_excitatory, 
                                   circuit_inductances = self.synaptic_dendrite_circuit_inductances,
                                   junction_critical_current = self.junction_critical_current, 
                                   bias_currents = self.bias_currents,
                                   input_synaptic_connections = [self.name], 
                                   input_synaptic_inductances = [self.synaptic_dendrite_input_synaptic_inductance],
                                   integration_loop_self_inductance = self.integration_loop_self_inductance,
                                   integration_loop_output_inductance = self.integration_loop_output_inductance,
                                   integration_loop_time_constant = self.integration_loop_time_constant)                        
                                   
        self.dendrite = synapse_dendrite
        # end make synaptic dendrite  
        
        # configure spd
        self.L_tot = np.sum(self.synaptic_circuit_inductors)
        self.tau_plus = self.L_tot/(np.sum(self.synaptic_circuit_resistors))
        self.tau_minus = self.L_tot/self.synaptic_circuit_resistors[1]
        self.spd_duration = 6*self.tau_minus
        self.t0 = self.synaptic_hotspot_duration
        self.I_spd = self.synaptic_spd_current
        self.M_self = self.synaptic_dendrite_input_synaptic_inductance[1]*np.sqrt(self.synaptic_dendrite_input_synaptic_inductance[0]*self.synaptic_circuit_inductors[2])
        self.r1 = self.synaptic_circuit_resistors[0]
        self.r2 = self.synaptic_circuit_resistors[1]
        # end configure spd
        
        synapse.synapses[self.name] = self
                
        # print('synapse created')
        
        return

    def __del__(self):
        # print('synapse deleted')
        return
                
        return self
    

class dendrite():
    
    _next_uid = 0
    dendrites = dict()

    def __init__(self, *args, **kwargs):
        
        #make new dendrite
        self.uid = dendrite._next_uid
        self.unique_label = 'd'+str(self.uid)
        dendrite._next_uid += 1
        
        # name the dendrite
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'unnamed_dendrite__{}'.format(self.unique_label)
        # end name 
        
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
            _i_or_e = 'excitatory' # 'excitatory' by default
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
                raise ValueError('[soens_sim] Input synaptic inductances to dendrites are specified as a list of pairs of real numbers with one pair per synaptic connection. The first element of the pair is the inductance on the dendritic receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the synapse and the dendritic receiving loop.')
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
                raise ValueError('[soens_sim] Input dendritic inductances to dendrites are specified as a list of pairs of real numbers with one pair per dendritic connection. The first element of the pair is the inductance on the dendritic receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the input dendrite and the dendritic receiving loop.')
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
                raise ValueError('[soens_sim] Input direct inductances to dendrites are specified as a list of pairs of real numbers with one pair per direct connection. The first element of the pair is the inductance on the dendritic receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the input direct signal and the dendritic receiving loop.')
        else:
            self.input_direct_inductances =  []
                    
        if 'circuit_inductances' in kwargs:
            if type(kwargs['circuit_inductances']) == list and len(kwargs['circuit_inductances']) == 4:
                self.circuit_inductances = kwargs['circuit_inductances']
            else:
                raise ValueError('[soens_sim] circuit_inductances is a list of four real numbers greater than zero with units of picohenries. The first element is the self inductance of the left branch of the DR loop, excluding the JJ and any mutual inductor inputs. The second element is the right branch of the DR loop, excluding the JJ and any mutual inductor inputs. The third element is the inductor to the right of the DR loop that goes to the JTL. The fourth element is the inductor in the JTL. All other contributions to DR loop inductance (JJs and MIs) will be handled separately, as will the inductance of the DI loop.')
        else:
            self.circuit_inductances = [20, 20, 200, 77.5]            

        if 'junction_critical_current' in kwargs:
            self.junction_critical_current =  kwargs['junction_critical_current']
        else:
            self.junction_critical_current =  40 #default Ic = 40 uA
        self.I_c = self.junction_critical_current
            
        if 'bias_currents' in kwargs:
            self.bias_currents = kwargs['bias_currents']
        else:
            self.bias_currents = [72, 29, 35] #[bias to DR loop (J_th), bias to JTL, bias to DI loop]        
            
        if 'integration_loop_self_inductance' in kwargs:
            # if type(kwargs['integration_loop_self_inductance']) == int or type(kwargs['integration_loop_self_inductance']) == float:
            if kwargs['integration_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Integration loop self inductance associated with dendritic integration loop must be a real number between zero and infinity (units of picohenries)')
            else:
                 self.integration_loop_self_inductance = kwargs['integration_loop_self_inductance']
        else: 
            self.integration_loop_self_inductance = 10e3 #default value, units of pH
                        
        if 'integration_loop_output_inductance' in kwargs:
            if type(kwargs['integration_loop_output_inductance']) != list:
                self.integration_loop_output_inductance = kwargs['integration_loop_output_inductance']
                ti = self.integration_loop_output_inductance
            if type(kwargs['integration_loop_output_inductance']) == list:
                self.integration_loop_output_inductance = kwargs['integration_loop_output_inductance']
                ti = sum(self.integration_loop_output_inductance)
        else: 
            self.integration_loop_output_inductance = 200 #default value, units of pH
        self.integration_loop_total_inductance = self.integration_loop_self_inductance+ti
        
        if 'integration_loop_time_constant' in kwargs:
            self.integration_loop_time_constant = kwargs['integration_loop_time_constant']
        else:
            self.integration_loop_time_constant = 250 #default time constant units of ns
 
        dendrite.dendrites[self.name] = self
            
        # print('dendrite created')        
        return      

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

        # if len(args) > 0:
        #     if type(args[0]) == str:
        #         _name = args[0]
        #     elif type(args[0]) == int or type(args[0]) == float:
        #         _name = str(args[0])
        # else:
        #     _name = 'unnamed_neuron'
        # self.name = _name
        
        # name the neuron
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = 'unnamed_neuron__{}'.format(self.unique_label)
        # end name
        
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
                raise ValueError('[soens_sim] circuit_inductances is a list of four real numbers greater than zero with units of picohenries. The first element is the self inductance of the left branch of the DR loop, excluding the JJ and any mutual inductor inputs. The second element is the right branch of the DR loop, excluding the JJ and any mutual inductor inputs. The third element is the inductor to the right of the DR loop that goes to the JTL. The fourth element is the inductor in the JTL. All other contributions to DR loop inductance (JJs and MIs) will be handled separately, as will the inductance of the DI loop.')
        else:
            self.circuit_inductances = [20, 20, 200, 77.5]
        
        if 'input_direct_connections' in kwargs:
            self.input_direct_connections = kwargs['input_direct_connections']            
        else:
            self.input_direct_connections = []        

        if 'input_direct_inductances' in kwargs:
            if type(kwargs['input_direct_inductances']) == list:
                    self.input_direct_inductances = kwargs['input_direct_inductances']
            else:
                raise ValueError('[soens_sim] Input direct inductances to neurons are specified as a list of pairs of real numbers with one pair per direct connection. The first element of the pair is the inductance on the neuronal receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the direct connection and the neuronal receiving loop.')
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
                raise ValueError('[soens_sim] Input synaptic inductances to neurons are specified as a list of pairs of real numbers with one pair per synaptic connection. The first element of the pair is the inductance on the neuronal receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the synapse and the neuronal receiving loop.')
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
                raise ValueError('[soens_sim] Input dendritic inductances to neurons are specified as a list of pairs of real numbers with one pair per dendritic connection. The first element of the pair is the inductance on the neuronal receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the dendritic and the neuronal receiving loop.')
        else:
            self.input_dendritic_inductances =  []
            
        if 'junction_critical_current' in kwargs:
            self.junction_critical_current = kwargs['junction_critical_current']
        else:
            self.junction_critical_current = 40 #default Ic = 40 uA
        self.I_c = self.junction_critical_current
                    
        if 'bias_currents' in kwargs:
            _Ib = kwargs['bias_currents']
        else:
            _Ib = [74, 36, 35] #[bias to NR loop (J_th), bias to JTL, bias to NI loop]
        self.bias_currents =  _Ib
        
        if 'integration_loop_self_inductance' in kwargs:
            # if type(kwargs['integration_loop_self_inductance']) == int or type(kwargs['integration_loop_self_inductance']) == float:
            if kwargs['integration_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Integration loop self inductance associated with neuronal integration loop must be a real number between zero and infinity (units of picohenries)')
            else:
                 self.integration_loop_self_inductance = kwargs['integration_loop_self_inductance']
        else: 
            self.integration_loop_self_inductance = 775 #default value, units of picohenries
                        
        if 'integration_loop_output_inductance' in kwargs:
            self.integration_loop_output_inductance = kwargs['integration_loop_output_inductance']
        else:
            self.integration_loop_output_inductance = [400,1] # defaults; [inudctor_to_latching_jj, k_to_latching_jj]        
        self.integration_loop_total_inductance = self.integration_loop_self_inductance+self.integration_loop_output_inductance[0]        
        
        if 'integration_loop_time_constant' in kwargs:
            self.integration_loop_time_constant = kwargs['integration_loop_time_constant']
        else:
            self.integration_loop_time_constant = 50 #default time constant units of ns
                
        if 'time_params' in kwargs:
            self.time_params = kwargs['time_params']                 
        else:
            time_params = dict()
            time_params['dt'] = 0.1
            time_params['tf'] = 1e3
            self.time_params = time_params
            
        if 'refractory_dendrite_num_jjs' in kwargs:
            self.refractory_dendrite_num_jjs = kwargs['refractory_dendrite_num_jjs']
        else:
            self.refractory_dendrite_num_jjs = 4
                
        if 'refractory_time_constant' in kwargs:
            # if type(kwargs['refractory_time_constant']) == int or type(kwargs['refractory_time_constant']) == float:
            if kwargs['refractory_time_constant'] < 0:
                raise ValueError('[soens_sim] time_constant associated with neuronal refraction must be a real number between zero and infinity')
            else:
                self.refractory_time_constant = kwargs['refractory_time_constant']
        else:
            self.refractory_time_constant = 50 #default time constant, units of ns
            
        if 'refractory_junction_critical_current' in kwargs:
            self.refractory_junction_critical_current = kwargs['refractory_junction_critical_current']
        else:
            self.refractory_junction_critical_current = 40 #default J_th Ic = 40 uA        
        
        if 'refractory_loop_circuit_inductances' in kwargs:
            if type(kwargs['refractory_loop_circuit_inductances']) == list and len(kwargs['refractory_loop_circuit_inductances']) == 4:
                self.refractory_loop_circuit_inductances = kwargs['refractory_loop_circuit_inductances']
            else:
                raise ValueError('[soens_sim] refractory_loop_circuit_inductances is a list of four real numbers greater than zero with units of picohenries. The first element is the self inductance of the left branch of the DR loop, excluding the JJ and any mutual inductor inputs. The second element is the right branch of the DR loop, excluding the JJ and any mutual inductor inputs. The third element is the inductor to the right of the DR loop that goes to the JTL. The fourth element is the inductor in the JTL. All other contributions to DR loop inductance (JJs and MIs) will be handled separately, as will the inductance of the DI loop.')
        else:
            self.refractory_loop_circuit_inductances = [20, 20, 200, 77.5]
            
        if 'refractory_loop_self_inductance' in kwargs:
            # if type(kwargs['refractory_loop_self_inductance']) == int or type(kwargs['refractory_loop_self_inductance']) == float:
            if kwargs['refractory_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Refractory loop self inductance associated with refractory suppression loop must be a real number between zero and infinity (units of picohenries)')
            else:
                 self.refractory_loop_self_inductance = kwargs['refractory_loop_self_inductance']
        else: 
            self.refractory_loop_self_inductance = 1e3 #default value, units of picohenries
                        
        if 'refractory_loop_output_inductance' in kwargs:
            # if type(kwargs['refractory_loop_output_inductance']) == int or type(kwargs['refractory_loop_output_inductance']) == float:
            if kwargs['refractory_loop_output_inductance'] < 0:
                raise ValueError('[soens_sim] Refractory loop output inductance associated with coupling between refractory suppression loop and neuron must be a real number between zero and infinity (units of picohenries)')
            else:
                 self.refractory_loop_output_inductance = kwargs['refractory_loop_output_inductance']
        else: 
            self.refractory_loop_output_inductance = 200 #default value, units of picohenries 
            
        if 'refractory_bias_currents' in kwargs:
            _Ib = kwargs['refractory_bias_currents']
        else:
            _Ib = [74, 36, 35] #[bias to NR loop (J_th), bias to JTL, bias to NI loop]
        self.refractory_bias_currents =  _Ib
        
        if 'refractory_receiving_input_inductance' in kwargs:
            if type(kwargs['refractory_receiving_input_inductance']) == list:
                    self.refractory_receiving_input_inductance = kwargs['refractory_receiving_input_inductance']
            else:
                raise ValueError('[soens_sim] refractory_receiving_input_inductance is specified as a pair of real numbers. The first element of the pair is the inductance on the refractory receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the refractory dendrite and the neuronal integration loop.')
        else:
            self.refractory_receiving_input_inductance =  [20,1]
            
        if 'neuronal_receiving_input_refractory_inductance' in kwargs:
            if type(kwargs['neuronal_receiving_input_refractory_inductance']) == list:
                    self.neuronal_receiving_input_refractory_inductance = kwargs['neuronal_receiving_input_refractory_inductance']
            else:
                raise ValueError('[soens_sim] neuronal_receiving_input_refractory_inductance is specified as a pair of real numbers. The first element of the pair is the inductance on the neuronal receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the refractory dendrite and the neuronal receiving loop.')
        else:
            self.neuronal_receiving_input_refractory_inductance =  [20,1]
                
        if 'homeostatic_time_constant' in kwargs:
            self.homeostatic_time_constant = kwargs['homeostatic_time_constant']
        else:
            self.homeostatic_time_constant = 50 #default time constant, units of ns
            
        if 'homeostatic_junction_critical_current' in kwargs:
            _Ic = kwargs['homeostatic_junction_critical_current']
        else:
            _Ic = 40 #default J_th Ic = 40 uA
        self.homeostatic_junction_critical_current = _Ic
        
        if 'homeostatic_loop_circuit_inductances' in kwargs:
            if type(kwargs['homeostatic_loop_circuit_inductances']) == list and len(kwargs['homeostatic_loop_circuit_inductances']) == 4:
                self.homeostatic_loop_circuit_inductances = kwargs['homeostatic_loop_circuit_inductances']
            else:
                raise ValueError('[soens_sim] homeostatic_loop_circuit_inductances is a list of four real numbers greater than zero with units of picohenries. The first element is the self inductance of the left branch of the DR loop, excluding the JJ and any mutual inductor inputs. The second element is the right branch of the DR loop, excluding the JJ and any mutual inductor inputs. The third element is the inductor to the right of the DR loop that goes to the JTL. The fourth element is the inductor in the JTL. All other contributions to DR loop inductance (JJs and MIs) will be handled separately, as will the inductance of the DI loop.')
        else:
            self.homeostatic_loop_circuit_inductances = [20, 20, 200, 77.5]
            
        if 'homeostatic_loop_self_inductance' in kwargs:
            if kwargs['homeostatic_loop_self_inductance'] < 0:
                raise ValueError('[soens_sim] Homeostatic loop self inductance associated with homeostasis must be a real number between zero and infinity (units of picohenries)')
            else:
                 self.homeostatic_loop_self_inductance = kwargs['homeostatic_loop_self_inductance']
        else: 
            self.homeostatic_loop_self_inductance = 1e3 #default value, units of picohenries
                        
        if 'homeostatic_loop_output_inductance' in kwargs:
            if kwargs['refractory_loop_output_inductance'] < 0:
                raise ValueError('[soens_sim] Homeostatic loop output inductance associated with coupling between homeostatic loop and neuron must be a real number between zero and infinity (units of picohenries)')
            else:
                 self.homeostatic_loop_output_inductance = kwargs['homeostatic_loop_output_inductance']
        else: 
            self.homeostatic_loop_output_inductance = 200 #default value, units of picohenries 
            
        if 'homeostatic_bias_currents' in kwargs:
            _Ib = kwargs['homeostatic_bias_currents']
        else:
            _Ib = [74, 36, 35] #[bias to NR loop (J_th), bias to JTL, bias to NI loop] # units of uA
        self.homeostatic_bias_currents =  _Ib
        
        if 'homeostatic_receiving_input_inductance' in kwargs:
            if type(kwargs['homeostatic_receiving_input_inductance']) == list:
                    self.homeostatic_receiving_input_inductance = kwargs['homeostatic_receiving_input_inductance']
            else:
                raise ValueError('[soens_sim] homeostatic_receiving_input_inductance is specified as a pair of real numbers. The first element of the pair is the inductance on the homeostatic receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the homeostatic dendrite and the neuronal integration loop.')
        else:
            self.homeostatic_receiving_input_inductance =  [20,1]
            
        if 'neuronal_receiving_input_homeostatic_inductance' in kwargs:
            if type(kwargs['neuronal_receiving_input_homeostatic_inductance']) == list:
                    self.neuronal_receiving_input_homeostatic_inductance = kwargs['neuronal_receiving_input_homeostatic_inductance']
            else:
                raise ValueError('[soens_sim] neuronal_receiving_input_homeostatic_inductance is specified as a pair of real numbers. The first element of the pair is the inductance on the neuronal receiving loop side with units of picohenries. The second element of the pair is the mutual inductance coupling factor k (M = k*sqrt(L1*L2)) between the homeostatic dendrite and the neuronal receiving loop.')
        else:
            self.neuronal_receiving_input_homeostatic_inductance =  [20,1] 
          
        if 'threshold_circuit_inductances' in kwargs:
            self.threshold_circuit_inductances = kwargs['threshold_circuit_inductances']
        else:
            self.threshold_circuit_inductances = [10,0,20] # [inductor receiving M from NI loop, inductor coupling to transmitter, inductor coupling back to NR for refraction]
        self.threshold_circuit_total_inductance = np.sum(self.threshold_circuit_inductances)
            
        if 'threshold_circuit_resistance' in kwargs:
            self.threshold_circuit_resistance = kwargs['threshold_circuit_resistance']
        else:
            self.threshold_circuit_resistance = 0.8 # units of mOhms to match with pH and ns
        
        if 'threshold_circuit_bias_current' in kwargs:
            self.threshold_circuit_bias_current = kwargs['threshold_circuit_bias_current']
        else:
            self.threshold_circuit_bias_current = 35 # uA
            
        if 'threshold_junction_critical_current' in kwargs:
            self.threshold_junction_critical_current = kwargs['threshold_junction_critical_current']
        else:
            self.threshold_junction_critical_current = 40 # uA    
        
        # make neuron cell body as dendrite
        temp_list_1 = self.input_dendritic_connections
        # temp_str = '{}__r'.format(self.name)
        # temp_list_1.append(temp_str)
        temp_list_2 = self.input_dendritic_inductances
        temp_list_2.append(self.neuronal_receiving_input_refractory_inductance)
        # print('self.circuit_inductances = {}'.format(self.circuit_inductances))
        neuron_dendrite = dendrite(name = '{}__d'.format(self.name),
                                   num_jjs = self.num_jjs,
                                   inhibitory_or_excitatory = 'excitatory', 
                                   circuit_inductances = self.circuit_inductances,
                                   input_direct_connections = self.input_direct_connections,
                                   input_direct_inductances = self.input_direct_inductances,
                                   input_synaptic_connections = self.input_synaptic_connections, 
                                   input_synaptic_inductances = self.input_synaptic_inductances,
                                   input_dendritic_connections = temp_list_1, 
                                   input_dendritic_inductances = temp_list_2,
                                   junction_critical_current = self.junction_critical_current, 
                                   bias_currents = self.bias_currents,
                                   integration_loop_self_inductance = self.integration_loop_self_inductance,
                                   integration_loop_output_inductance = self.integration_loop_output_inductance[0],
                                   integration_loop_time_constant = self.integration_loop_time_constant)                
                 
        # make refractory dendrite
        # refractory_loop = dendrite(name = '{}__r'.format(self.name),
        #                           num_jjs = self.refractory_dendrite_num_jjs,
        #                           inhibitory_or_excitatory = 'inhibitory',
        #                           circuit_inductances = self.refractory_loop_circuit_inductances,
        #                           junction_critical_current = self.refractory_junction_critical_current,
        #                           input_dendritic_connections = ['{}__d'.format(self.name)],
        #                           input_dendritic_inductances = [self.refractory_receiving_input_inductance], # [self.integration_loop_output_inductances[1]],
        #                           bias_currents = self.refractory_bias_currents,
        #                           integration_loop_time_constant = self.refractory_time_constant,
        #                           integration_loop_self_inductance = self.refractory_loop_self_inductance,
        #                           integration_loop_output_inductance = self.refractory_loop_output_inductance)                     
                
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
                self.dendrites[self.synapses[name_2].dendrite.name] = self.synapses[name_2].dendrite
                
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
            self.dendrites[self.synapses[name_1].dendrite.name] = self.synapses[name_1].dendrite
            
        #also add direct connections to neuron
        self.direct_connections = dict()
        for name_1 in self.input_direct_connections:
            self.direct_connections[input_signal.input_signals[name_1].name] = input_signal.input_signals[name_1]                      
            
        #finally, add self to self as dendrite
        self.dendrites['{}__d'.format(self.name)] = dendrite.dendrites['{}__d'.format(self.name)]
                        
        return self   
    
    
    def sum_inductances(self):  
        
        print_progress = False
    
        # go through all dendrites in the neuron. remember the neuron itself is a dendrite, so it is included here. synapses are dendrites, so they are included, too.
        # currently set up so all excitatory connections are on the left, all inhibitory on the right branch of the DR loop. is that good?
        for name_dendrite in self.dendrites:
            # print('{}'.format(name_dendrite))
            
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
                
                # print('{}'.format(name_dendrite_in))
                
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
                
        self.dendrites['{}__d'.format(self.name)].L_right = self.circuit_inductances[1] + self.neuronal_receiving_input_refractory_inductance[0]
        self.L_nr = self.dendrites['{}__d'.format(self.name)].L_right+self.dendrites['{}__d'.format(self.name)].L_left+2*Ljj_pH(self.junction_critical_current,0)
            
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
    
        return self
        
    def run_sim(self):
        
        # set up time vec
        dt = self.time_params['dt']
        tf = self.time_params['tf']
                
        time_vec = np.arange(0,tf+dt,dt)    
        self.time_vec = time_vec            

        # attach synapses, dendrites, and direct connections to neuron
        self.make_connections()
        
        # find total inductances of all loops
        self.sum_inductances()
        
        # make drive signals
        self.construct_dendritic_drives()
        
        # simulate neuron in time         
        self = neuron_time_stepper(self)
        
        # calculate spike times
        self.output_voltage = self.I_ni_vec # self.integration_loop_output_inductances[0][0]*np.diff(self.I_ni_vec)
        self.voltage_peaks = self.spike_times
        # self.voltage_peaks, _ = find_peaks(self.output_voltage, distance = 10e-9/dt, height = 10) # , height = min_peak_height, ) # , distance = 10e-9/dt
        # self.spike_times = self.time_vec[self.voltage_peaks]
        self.spike_times = np.asarray(self.spike_times)
        self.interspike_intervals = np.diff(self.spike_times)
        if len(self.interspike_intervals) > 0:
            self.max_rate = 1/np.min(self.interspike_intervals)
        else:
            self.max_rate = 0
        
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





