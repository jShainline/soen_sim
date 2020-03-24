import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import time

from _functions import synaptic_response_function, synaptic_time_stepper, plot_params
pp = plot_params()
plt.rcParams['figure.figsize'] = pp['fig_size']

import pickle

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
            if (kwargs['input_temporal_form'] == 'single_pulse' or 
                kwargs['input_temporal_form'] == 'constant_rate' or 
                kwargs['input_temporal_form'] == 'arbitrary_pulse_train'):
                _temporal_form = kwargs['input_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid input signal temporal form to input %s (unique_label = %s)\nThe allowed values of input_temporal_form are ''single_pulse'', ''constant_rate'', and ''arbitrary_pulse_train''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'single_pulse'
        self.input_temporal_form =  _temporal_form #'single_pulse' by default
        
        if 'spike_times' in kwargs:
            if (self.input_temporal_form == 'single_pulse' or self.input_temporal_form == 'arbitrary_pulse_train'):
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

        if 'loop_temporal_form' in kwargs:
            if kwargs['loop_temporal_form'] == 'exponential' or kwargs['loop_temporal_form'] == 'power_law':
                _temporal_form = kwargs['loop_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid loop temporal form to synapse %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.name, self.unique_label))
        else:
            _temporal_form = 'exponential'
        self.loop_temporal_form =  _temporal_form #'exponential' or 'power_law'; 'exponential' by default

        if 'time_constant' in kwargs:
            # print(type(kwargs['time_constant']))
            # print(kwargs['time_constant'])
            # if type(kwargs['time_constant']) == int or type(kwargs['time_constant']) == float or type(kwargs['time_constant']) == np.float64:
            if kwargs['time_constant'] < 0:
                raise ValueError('[soens_sim] time_constant associated with synaptic decay must be a real number between zero and infinity')
            else:
                self.time_constant = kwargs['time_constant']
        else:
            if self.loop_temporal_form == 'exponential':
                self.time_constant = 200e-9 #default time constant units of seconds
        
        if 'power_law_exponent' in kwargs:
            # if type(kwargs['power_law_exponent']) == int or type(kwargs['power_law_exponent']) == float:
            if kwargs['power_law_exponent'] > 0:
                raise ValueError('[soens_sim] power_law_exponent associated with synaptic decay must be a real number between negative infinity and zero')
            else:
                 self.power_law_exponent = kwargs['power_law_exponent']
        else:
            if self.loop_temporal_form == 'power_law':                
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
            if kwargs['synaptic_bias_current'] < 34e-6 or kwargs['synaptic_bias_current'] > 39e-6:
                raise ValueError('[soens_sim] synaptic_bias_current associated with synaptic integration loop must be a real number between 34e-6 and 39e-6 (units of amps)')
            else:
                 self.synaptic_bias_current = kwargs['synaptic_bias_current']
        else:
            _synaptic_bias_current_default = 35e-6 #units of amps
            self.synaptic_bias_current = _synaptic_bias_current_default

        if 'loop_bias_current' in kwargs:
            # if type(kwargs['loop_bias_current']) == int or type(kwargs['loop_bias_current']) == float:
            if kwargs['loop_bias_current'] < 0:
                raise ValueError('[soens_sim] loop_bias_current associated with synaptic integration loop must be a real number between xx and yy (units of amps)')
            else:
                 self.loop_bias_current = kwargs['loop_bias_current']
        else:
            _loop_bias_current_default = 30e-6 #units of amps
            self.loop_bias_current = _loop_bias_current_default
            
        if 'input_signal_name' in kwargs:
            if kwargs['input_signal_name'] != '':
                self.input_signal_name = kwargs['input_signal_name']
            
        # attach external input signals to synapse
        if hasattr(self, 'input_signal_name'):
            self.input_signal = input_signal.input_signals[self.input_signal_name]
            # print('added input signal with name {} to synapse {}'.format(self.input_signal_name,self.name))                               
        
        synapse.synapses[self.name] = self
        
        # self.neuronal_connections = {} #[unique_label (input_neuron or input_signal), unique_label (output_neuron)] (label of neurons from which synapse receives and to which synapse connects)
        # self.input_spike_times = {} #list of real numbers (obtained from spike_times of neuronal_connection)
        # self.loop_integrated_current = {} #real function of time (I_si, amps; output variable)
        
        # print('synapse created')
        
        return

    def __del__(self):
        # print('synapse deleted')
        return
        
        
    # def add_time_constant(self, tau):
    #     if self.loop_temporal_form == 'exponential':
    #         self.time_constant = tau
    #     else:
    #         raise ValueError('[soens_sim] Tried to assign a time constant to a synapse without exponential leak in synapse %s (unique_label = %s)' % (self.name, self.unique_label))
    #     return

    # def add_power_law_exponent(self, gamma):
    #     if self.loop_temporal_form == 'power_law':
    #         self.power_law_exponent = gamma
    #     else:
    #         raise ValueError('[soens_sim] Tried to assign a power-law exponent to a synapse without power-law leak in synapse %s (unique_label = %s)' % (self.name, self.unique_label))
    # @classmethod
    # def get_instances(cls):
    #     dead = set()
    #     for ref in cls._instances:
    #         obj = ref()
    #         if obj is not None:
    #             yield obj
    #         else:
    #             dead.add(ref)
    #     cls._instances -= dead
        
    # @classmethod
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

    # @classmethod
    def plot_integration_loop_current(self,time_vec):
        
        input_spike_times = self.input_spike_times

        plt.rcParams['figure.figsize'] = pp['fig_size']
        fig, axes = plt.subplots(1,1)
        # axes.plot(time_vec*1e6,input_spike_times, 'o-', linewidth = 1, markersize = 3, label = 'input pulses'.format())
        axes.plot(time_vec*1e6,self.I_si*1e6, 'o-', linewidth = 1, markersize = 3, label = 'synaptic response'.format())
        axes.set_xlabel(r'Time [us]', fontsize = pp['axes_labels_font_size'])
        axes.set_ylabel(r'Isi [uA]', fontsize = pp['axes_labels_font_size'])
        axes.tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])

        #ylim((ymin_plot,ymax_plot))
        #xlim((xmin_plot,xmax_plot))

        axes.legend(loc='best')
        grid(True,which='both')
        title('Synapse: '+self.name+' ('+self.unique_label+')'+\
              '\nI_sy = '+str(self.synaptic_bias_current*1e6)+' uA'+\
              '; tau_si = '+str(self.time_constant*1e9)+' ns'+\
              '; L_si = '+str(self.integration_loop_total_inductance*1e9)+' nH',fontsize = pp['title_font_size'])
        plt.show()

        return

# class dendrite:
# 	dendrite.unique_label = 'd'+int
# 	dendrite.input_connections = {unique_label} (list of input synapse/dendrite labels)
# 	dendrite.input_mutual_inductances = real (list of mutual inductances between synapses/dendrites and dendrite)
# 	dendrite.loop_temporal_form = 'exponential' or 'power_law'
#     dendrite.time_constant = real (if 'exponential')
#     dendrite.power_law_exponent = real (-inf,0) (if 'power_law')
#     dendrite.integration_loop_inductance = real (L_si, henries)
#     dendrite.synaptic_bias_current = real (I_sy, amps)
#     dendrite.loop_bias_current = real (I_b, amps)
#     dendrite.loop_integrated_current = real function of time (I_si, amps; output variable)

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
            
        if 'input_connections' in kwargs:
            # if type(kwargs['input_connections']) == set:
            #     if all(isinstance(x, str) for x in kwargs['input_connections']):
                self.input_connections = kwargs['input_connections']
            # else:
            #     raise ValueError('[soens_sim] Input connections to neurons are specified as a set of strings referencing the synapse/dendrite unique labels')
        else:
            self.input_connections = [[{}]]        

        if 'input_inductances' in kwargs:
            if type(kwargs['input_inductances']) == list:
                    self.input_inductances = kwargs['input_inductances']
            else:
                raise ValueError('[soens_sim] Input inductances to neurons are specified as a list of pairs of real numbers with one pair per synaptic or dendritic connection. The first element of the pair is the inductance on the neuronal receiving loop side with units of henries. The second element of the pair is the mutual inductance coupling factor (M = k*sqrt(L1*L2)) between the synapse or dendrite and the neuronal receiving loop.')
        else:
            self.input_inductances =  [[]]

        if 'thresholding_junction_critical_current' in kwargs:
            # if type(kwargs['thresholding_junction_critical_current']) == float:
            _Ic = kwargs['thresholding_junction_critical_current']
            # else:
                # raise ValueError('[soens_sim] Thresholding junction critical current must be a real number with units of amps')
        else:
            _Ic = 40e-6 #default J_th Ic = 40 uA
        self.thresholding_junction_critical_current =  _Ic
            
        if 'threshold_bias_current' in kwargs:
            # if type(kwargs['threshold_bias_current']) == float:
            _Ib = kwargs['threshold_bias_current']
            # else:
                # raise ValueError('[soens_sim] Thresholding junction bias current must be a real number with units of amps')
        else:
            _Ib = 35e-6 #default J_th Ic = 40 uA
        self.threshold_bias_current =  _Ib
                
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
                 
        #set up refractory loop as synapse
        refractory_synaptic_bias_current = 39e-6 #default for now; want it to saturate with each spike
        refractory_loop_bias_current = 31e-6 #doesn't matter now as synapse saturation is independent of bias; will matter with future updates, I hope
        refractory_loop = synapse(self.name+'__refractory_suppression_synapse', loop_temporal_form = 'exponential', time_constant = self.refractory_time_constant,
                    integration_loop_self_inductance = self.refractory_loop_self_inductance, integration_loop_output_inductance = self.refractory_loop_output_inductance, 
                    synaptic_bias_current = refractory_synaptic_bias_current, loop_bias_current = refractory_loop_bias_current)
        
        refractory_loop.unique_label = self.unique_label+'_rs'
        refractory_loop.input_spike_times = []
        neuron.refractory_loop = refractory_loop
        self.input_connections.append(refractory_loop.name)        
                
        # print('neuron created')
        
        return

    def __del__(self):
        # print('neuron deleted')
        return
    
    # @classmethod
    # def get_instances(cls):
    #     dead = set()
    #     for ref in cls._instances:
    #         obj = ref()
    #         if obj is not None:
    #             yield obj
    #         else:
    #             dead.add(ref)
    #     cls._instances -= dead
    
    def add_synapses_to_neuron(self):
        
        self.synapses = []
        for name in self.input_connections:             
            self.synapses.append(synapse.synapses[name])
            
        return

        # self.synapses = []
        # for obj in synapse.get_instances():
        #     # print('synapse {} has time constant = {}'.format(obj.unique_label,obj.time_constant))
        #     if obj.name in self.input_connections:
        #         self.synapses.append(obj)
                
        # for ii in range(len(self.synapses)):
        #     print(len(self.synapses[ii].input_spike_times))

        return self
        
    def run_sim(self):
        
        # set up time vec
        num_dt_pre = 10
        dt = self.sim_params['dt']
        t_obs = dt*np.round((self.sim_params['pre_observation_duration']+num_dt_pre*dt)/dt,decimals = 0)
        t_sim_total = dt*np.round((t_obs+self.sim_params['observation_duration'])/dt,decimals = 0)
        time_vec = np.arange(0,t_sim_total+dt,dt)           

        # attach synapses to neuron
        self.add_synapses_to_neuron()        
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
        I_si_sat__rs_loop = 100

        for ii in range(len(self.synapses)):
            _I_sy = 1e6*self.synapses[ii].synaptic_bias_current
            # print('I_sy = {} uA'.format(_I_sy))
            self.synapses[ii].tau_rise = (1.294*_I_sy-43.01)*1e-9
            # print('tau_rise = {} ns'.format(self.synapses[ii].tau_rise*1e9))
            _scale_factor = _reference_inductance/self.synapses[ii].integration_loop_total_inductance
            # print('_scale_factor = {}'.format(_scale_factor))
            self.synapses[ii].I_0 = (0.06989*_I_sy**2-3.948*_I_sy+53.73)*_scale_factor
            # print('I_0 = {} uA'.format(self.synapses[ii].I_0))
            mutual_inductance = self.input_inductances[ii][1]*np.sqrt(self.input_inductances[ii][0]*self.synapses[ii].integration_loop_output_inductance)
            self.synapses[ii].coupling_factor = mutual_inductance/self.receiving_loop_total_inductance
            # print('coupling_Factor = {}\n\n'.format(self.synapses[ii].coupling_factor))
            self.synapses[ii].I_si = np.zeros([len(time_vec),1])
            if hasattr(self.synapses[ii],'input_signal'):
                # cc = self.sim_params['num_tau_sim']*self.synapses[ii].time_constant
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
    
    def plot_receiving_loop_current(self,plot_save_string = ''):
    
        tt = time.time()   
        save_str = 'receiving_loop_current__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.png'
        
        time_vec = self.time_vec
        plt.rcParams['figure.figsize'] = pp['fig_size']
        #nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
        fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
        fig.suptitle('Current in the neuronal receiving loop versus time', fontsize = pp['title_font_size'])
        
        #upper panel, total I_nr
        axs[0].plot(time_vec*1e6,self.cell_body_circulating_current*1e6, 'o-', linewidth = 1, markersize = 3, label = 'Neuron: '+self.name+' ('+self.unique_label+')'.format())        
        #spike times
        ylim = axs[0].get_ylim()
        for ii in range(len(self.spike_times)):
            if ii == len(self.spike_times):
                axs[0].plot([self.spike_times[ii]*1e6, self.spike_times[ii]*1e6], [ylim[0], ylim[1]], 'r-', linewidth = 0.5, label = 'spike times'.format())
            else:
                axs[0].plot([self.spike_times[ii]*1e6, self.spike_times[ii]*1e6], [ylim[0], ylim[1]], 'r-', linewidth = 0.5)
        #threshold
        xlim = axs[0].get_xlim()
        axs[0].plot([xlim[0],xlim[1]], [self.thresholding_junction_critical_current*1e6,self.thresholding_junction_critical_current*1e6], 'g-', linewidth = 1.5, label = 'Threshold')
        axs[0].plot([self.time_vec[self.idx_obs_start]*1e6, self.time_vec[self.idx_obs_start]*1e6], [ylim[0], ylim[1]], 'b-', linewidth = 1.5, label = 'Begin observation')
        axs[0].set_xlabel(r'Time [$\mu$s]', fontsize = pp['axes_labels_font_size'])
        axs[0].set_ylabel(r'$I_{nr}$ [uA]', fontsize = pp['axes_labels_font_size'])
        axs[0].tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        axs[0].set_title('Total current in NR loop')
        axs[0].legend(loc = 'best')
        axs[0].grid(b = True, which='major', axis='both')

        #lower panel, scaled contributions from each synapse
        for ii in range(len(self.synapses)): 
            axs[1].plot(time_vec*1e6,self.synapses[ii].coupling_factor*self.synapses[ii].I_si*1e6, 'o-', linewidth = 1, markersize = 3, label = self.synapses[ii].unique_label.format())#'Synapse: '+self.synapses[ii].name+' ('+self.synapses[ii].unique_label+')'.format()
        axs[1].set_xlabel(r'Time [$\mu$s]', fontsize = pp['axes_labels_font_size'])
        axs[1].set_ylabel(r'Contribution to $I_{nr}$ [uA]', fontsize = pp['axes_labels_font_size'])
        axs[1].tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        axs[1].set_title('Scaled contribution from each synapse')
        axs[1].legend(loc = 'best')
        axs[1].grid(b = True, which='major', axis='both')        
        
        plt.show()       
        fig.savefig('figures/'+save_str)

        return
    
    def plot_rate_transfer_function(self,plot_save_string = ''):
        
        tt = time.time()        
        save_str = 'rate_transfer_function__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.png'
        # print(save_str)
        
        #nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
        fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
        fig.suptitle('Output rate versus input rate', fontsize = pp['title_font_size'])
        plt.title(plot_save_string, fontsize = pp['subtitle_font_size'])
        
        for qq in range(len(self.tau_ref_vec)):
            for jj in range(len(self.I_sy_vec)):
                for ii in range(len(self.tau_si_vec)):
                    # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = 1, markersize = 3, label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
                    # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = 1, markersize = 3, label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
                    ax.plot(self.rate_vec*1e-6,1e-6*1/self.isi_output_avg_mat[:,ii,jj,qq], 'o-', linewidth = 1, markersize = 3, label = 'mean firing rate; tau_si = {:2.2f}ns, tau_ref = {:2.2f}ns, I_sy = {:2.2f}uA'.format(self.tau_si_vec[ii]*1e9,self.tau_ref_vec[qq]*1e9,self.I_sy_vec[jj]*1e6))
        ylim = ax.get_ylim()
        # print(ylim)
        ax.plot(self.rate_vec*1e-6,self.rate_vec*1e-6, '-', linewidth = 0.5, label = 'rate-out equals rate-in')        
        ax.plot(self.rate_vec*1e-6,self.rate_vec*1e-6/2, '-', linewidth = 0.5, label = 'rate-out equals rate-in/2')        
        ax.plot(self.rate_vec*1e-6,self.rate_vec*1e-6/3, '-', linewidth = 0.5, label = 'rate-out equals rate-in/3')        
        ax.plot(self.rate_vec*1e-6,self.rate_vec*1e-6/4, '-', linewidth = 0.5, label = 'rate-out equals rate-in/4')        
        # ylim = ax.get_ylim()
        ax.set_xlabel(r'Input rate [MHz]', fontsize = pp['axes_labels_font_size'])
        ax.set_ylabel(r'Output rate [MHz]', fontsize = pp['axes_labels_font_size'])
        ax.tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        # ax.set_title('Total current in NR loop')
        ax.legend(loc = 'best')
        ax.grid(b = True, which='major', axis='both')        
        ax.set_ylim([ylim[0],ylim[1]])
        
        plt.show()
        fig.savefig('figures/'+save_str)
        
        return
    
    def plot_rate_transfer_function__no_lines(self,plot_save_string = ''):
        
        tt = time.time()        
        save_str = 'rate_transfer_function__'+plot_save_string+'__no_lines__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.png'
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
        fig.suptitle('Output rate versus input rate', fontsize = pp['title_font_size'])
        plt.title(plot_save_string, fontsize = pp['subtitle_font_size'])
        
        for jj in range(len(self.I_sy_vec)):
            for ii in range(len(self.tau_si_vec)):
                # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = 1, markersize = 3, label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
                # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = 1, markersize = 3, label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
                ax.plot(self.rate_vec*1e-6,1e-6*1/self.isi_output_avg_mat[:,ii,jj], 'o-', linewidth = 1, markersize = 3, label = 'mean firing rate; tau_si = {:2.2f}ns, I_sy = {:2.2f}uA'.format(self.tau_si_vec[ii]*1e9,self.I_sy_vec[jj]*1e6))        
        ax.set_xlabel(r'Input rate [MHz]', fontsize = pp['axes_labels_font_size'])
        ax.set_ylabel(r'Output rate [MHz]', fontsize = pp['axes_labels_font_size'])
        ax.tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        ax.legend(loc = 'best')
        ax.grid(b = True, which='major', axis='both')    
        
        plt.show()
        fig.savefig('figures/'+save_str)  

        return
    
    def plot_num_spikes(self,plot_save_string = ''):
        
        tt = time.time()        
        save_str = 'num_spikes_transfer_function__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.png'
        # print(save_str)
        
        #nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
        fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
        fig.suptitle('Output rate versus input rate\n'+plot_save_string, fontsize = pp['title_font_size'])
        
        #upper panel, rate_out vs rate_in
        for jj in range(len(self.I_sy_vec)):
            for ii in range(len(self.tau_si_vec)):
                axs[0].plot(self.rate_vec*1e-6,1e-6*1/self.isi_output_avg_mat[:,ii,jj], 'o-', linewidth = 1, markersize = 3, label = 'mean firing rate; tau_si = {:2.2f}ns, I_sy = {:2.2f}uA'.format(self.tau_si_vec[ii]*1e9,self.I_sy_vec[jj]*1e6))
        ylim = axs[0].get_ylim()
        axs[0].plot(self.rate_vec*1e-6,self.rate_vec*1e-6, 'o-', linewidth = 0.5, markersize = 3, label = 'rate-out equals rate-in')
        axs[0].set_xlabel(r'Input rate [MHz]', fontsize = pp['axes_labels_font_size'])
        axs[0].set_ylabel(r'Output rate [MHz]', fontsize = pp['axes_labels_font_size'])
        axs[0].tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        # ax.set_title('Total current in NR loop')
        axs[0].legend(loc = 'best')
        axs[0].grid(b = True, which='major', axis='both')
        axs[0].set_ylim([ylim[0],ylim[1]])
        
        #lower panel, num_spikes_out vs num_spikes_in
        for jj in range(len(self.I_sy_vec)):
            for ii in range(len(self.tau_si_vec)):
                # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = 1, markersize = 3, label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
                # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = 1, markersize = 3, label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
                axs[1].plot(self.rate_vec*1e-6,self.num_spikes_in_mat[:,ii,jj], 'o-', linewidth = 1, markersize = 3, label = 'num_spikes_in')
                axs[1].plot(self.rate_vec*1e-6,self.num_spikes_out_mat[:,ii,jj], 'o-', linewidth = 1, markersize = 3, label = 'num_spikes_out; tau_si = {:2.2f}ns, I_sy = {:2.2f}uA'.format(self.tau_si_vec[ii]*1e9,self.I_sy_vec[jj]*1e6))        
        axs[1].set_xlabel(r'Input rate [MHz]', fontsize = pp['axes_labels_font_size'])
        axs[1].set_ylabel(r'Number of spikes', fontsize = pp['axes_labels_font_size'])
        axs[1].tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        # ax.set_title('Total current in NR loop')
        axs[1].legend(loc = 'best')
        axs[1].grid(b = True, which='major', axis='both')
        
        plt.show()
        fig.savefig('figures/'+save_str)

        return
    
    def plot_rate_and_isi(self,plot_save_string = ''):
        
        pp = plot_params()
        tt = time.time()        
        save_str = 'rate_and_isi__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.png'
        # print(save_str)
        
        #nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
        fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
        fig.suptitle('Output versus input transfer function', fontsize = pp['title_font_size'])
        
        #upper panel, rate transfer function
        for ii in range(len(self.tau_si_vec)):
            # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = 1, markersize = 3, label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
            # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = 1, markersize = 3, label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
            axs[0].plot(self.rate_vec*1e-6,1e-6*1/self.isi_output_avg_mat[:,ii], 'o-', linewidth = 1, markersize = 3, label = 'mean firing rate; tau_si = '+str(self.tau_si_vec[ii]*1e9)+' ns'.format())            
        axs[0].set_xlabel(r'Input rate [MHz]', fontsize = pp['axes_labels_font_size'])
        axs[0].set_ylabel(r'Output rate [MHz]', fontsize = pp['axes_labels_font_size'])
        axs[0].tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        # ax.set_title('Total current in NR loop')
        axs[0].legend(loc = 'best')
        axs[0].grid(b = True, which='major', axis='both')
                
        #lower panel, inter-spike interval
        isi_vec = 1/self.rate_vec
        for ii in range(len(self.tau_si_vec)):
            # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = 1, markersize = 3, label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
            # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = 1, markersize = 3, label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
            axs[1].plot(isi_vec*1e6,1e6*self.isi_output_avg_mat[:,ii], 'o-', linewidth = 1, markersize = 3, label = 'mean inter-spike interval; tau_si = '+str(self.tau_si_vec[ii]*1e9)+' ns'.format())            
        axs[1].set_xlabel(r'Input inter-spike interval [$\mu$s]', fontsize = pp['axes_labels_font_size'])
        axs[1].set_ylabel(r'Output inter-spike interval [$\mu$s]', fontsize = pp['axes_labels_font_size'])
        axs[1].tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        # ax.set_title('Total current in NR loop')
        axs[1].legend(loc = 'best')
        axs[1].grid(b = True, which='major', axis='both')        
                
        plt.show()
        fig.savefig('figures/'+save_str)

        return 
    
    def plot_spike_train(self,plot_save_string):
        
        pp = plot_params()
        tt = time.time()        
        save_str = 'spike_train__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.png'
        # print(save_str)
        time_vec = self.time_vec
        
        #nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
        fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
        fig.suptitle('Input and output spike trains', fontsize = pp['title_font_size'])
        
        #upper panel, total I_nr
        axs[0].plot(time_vec*1e6,self.cell_body_circulating_current*1e6, 'o-', linewidth = 1, markersize = 3, label = 'Neuron: '+self.name+' ('+self.unique_label+')'.format())        

        #threshold
        xlim = axs[0].get_xlim()
        ylim = axs[0].get_ylim()
        axs[0].plot([xlim[0],xlim[1]], [self.thresholding_junction_critical_current*1e6,self.thresholding_junction_critical_current*1e6], 'g-', linewidth = 0.5, label = 'Threshold')
        axs[0].plot([self.time_vec[self.idx_obs_start]*1e6, self.time_vec[self.idx_obs_start]*1e6], [ylim[0], ylim[1]], 'b-', linewidth = 0.5, label = 'Begin observation')
        axs[0].set_xlabel(r'Time [$\mu$s]', fontsize = pp['axes_labels_font_size'])
        axs[0].set_ylabel(r'$I_{nr}$ [uA]', fontsize = pp['axes_labels_font_size'])
        axs[0].tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        axs[0].set_title('Total current in NR loop')
        axs[0].legend(loc = 'best')
        axs[0].grid(b = True, which='major', axis='both')
                
        #lower panel, synaptic input spikes, neuronal output spikes
        axs[1].plot(time_vec*1e6,self.spike_vec,linewidth = 1, label = 'neuronal spike times'.format())
        axs[1].plot(time_vec*1e6,self.synapses[0].spike_vec,linewidth = 1, label = 'synaptic spike times'.format())        
        axs[1].set_xlabel(r'Time [$\mu$s]', fontsize = pp['axes_labels_font_size'])
        axs[1].set_ylabel(r'Spikes [a.u.]', fontsize = pp['axes_labels_font_size'])
        axs[1].tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        axs[1].set_title('Synaptic and neuronal spiking')
        axs[1].legend(loc = 'best')
        axs[1].grid(b = True, which='major', axis='both')        
                
        plt.show()
        fig.savefig('figures/'+save_str)
        
        return
    
    def plot_fourier_transform(self,time_vec):
        
        _sv = self.spike_vec
        
        _sv_ft = np.fft.rfft(_sv)
        self.spike_vec__ft = _sv_ft
        num_pts = len(_sv_ft)
        temp_vec = np.linspace(0,1,num_pts)
        
        title_font_size = 20
        axes_labels_font_size = 16
        tick_labels_font_size = 12
        plt.rcParams['figure.figsize'] = (20,16)
        #nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
        fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
        fig.suptitle('Fourier transform of spike vec', fontsize = title_font_size)
        
        #upper panel, spike vec
        axs[0].plot(time_vec*1e6,_sv, 'o-', linewidth = 1, markersize = 3, label = 'Neuron: '+self.name+' ('+self.unique_label+')'.format())        
        axs[0].set_xlabel(r'Time [$\mu$s]', fontsize = axes_labels_font_size)
        axs[0].set_ylabel(r'Spikes [binary]', fontsize = axes_labels_font_size)
        axs[0].tick_params(axis='both', which='major', labelsize = tick_labels_font_size)
        axs[0].set_title('Spike train')
        axs[0].legend(loc = 'best')
        axs[0].grid(b = True, which='major', axis='both')
        
        #lower panel, fourier transform
        axs[1].plot(temp_vec,_sv_ft, 'o-', linewidth = 1, markersize = 3, label = 'Neuron: '+self.name+' ('+self.unique_label+')'.format())        
        axs[1].set_xlabel(r'frequency', fontsize = axes_labels_font_size)
        axs[1].set_ylabel(r'amplitude', fontsize = axes_labels_font_size)
        axs[1].tick_params(axis='both', which='major', labelsize = tick_labels_font_size)
        axs[1].set_title('fft')
        axs[1].legend(loc = 'best')
        axs[1].grid(b = True, which='major', axis='both')
                
        plt.show()
        save_str = 'spike_vec_fourier_transform__'+self.name        
        fig.savefig('figures/'+save_str)

        return self
        
    def plot_rate_vs_num_active_synapses(self,plot_save_string = ''):
        
        tt = time.time()        
        save_str = 'rate_vs_num_active_synapses__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.png'
        # print(save_str)
        
        #nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
        fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
        fig.suptitle('Output rate versus input rate', fontsize = pp['title_font_size'])
        plt.title(plot_save_string, fontsize = pp['subtitle_font_size'])
        
        ax.plot(self.num_active_synapses_vec,1e-6*1/self.isi_output_avg_vec[:], 'o-', linewidth = 1, markersize = 3, label = 'mean firing rate; input_rate = {:2.2f}MHz, tau_si = {:2.2f}ns, tau_ref = {:2.2f}ns, I_sy = {:2.2f}uA'.format(self.rate*1e-6,self.tau_si*1e9,self.tau_ref*1e9,self.I_sy*1e6))

        ax.set_xlabel(r'Number of active synapses', fontsize = pp['axes_labels_font_size'])
        ax.set_ylabel(r'Output rate [MHz]', fontsize = pp['axes_labels_font_size'])
        ax.tick_params(axis='both', which='major', labelsize = pp['tick_labels_font_size'])
        # ax.set_title('Total current in NR loop')
        ax.legend(loc = 'best')
        ax.grid(b = True, which='major', axis='both')      
        
        plt.show()
        fig.savefig('figures/'+save_str)
        
        return
    
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





