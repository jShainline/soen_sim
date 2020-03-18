class synapse():
    #example_synapse = synapse('colloquial_name', 'loop_temporal_form' = 'exponential', 'time_constant' = 100e-9)
    _next_uid = 0
    
    def __init__(self, *args, **kwargs):
        
        #make new synapse
        if len(args) > 0:
            if type(args[0]) == str:
                _name = args[0]
            elif type(args[0]) == int or type(args[0]) == float:
                _name = str(args[0])
        else:
                _name = 'unnamed_synapse'
        self.colloquial_name = _name
            
        if 'loop_temporal_form' in kwargs:
            if kwargs['loop_temporal_form'] == 'exponential' or kwargs['loop_temporal_form'] == 'power_law':
                _temporal_form = kwargs['loop_temporal_form']
            else:
                raise ValueError('[soens_sim] Tried to assign an invalid loop temporal form to synapse %s (unique_label = %s)\nThe allowed values of loop_temporal_form are ''exponential'' and ''power_law''' % (self.colloquial_name, self.unique_label))
        else:
            _temporal_form = 'exponential'
        self.loop_temporal_form =  _temporal_form #'exponential' or 'power_law'; 'exponential' by default
        
        if 'time_constant' in kwargs:
            if type(kwargs['time_constant']) == int or type(kwargs['time_constant']) == float:
                if kwargs['time_constant'] < 0:
                    raise ValueError('[soens_sim] time_constant associated with synaptic decay must be a real number between zero and infinity')
                else:
                    self.add_time_constant(kwargs['time_constant'])
        else:
            if self.loop_temporal_form == 'exponential':
                _tau_default = 100e-9 #units of seconds
                self.add_time_constant(_tau_default)
        
        if 'power_law_exponent' in kwargs:
            if type(kwargs['power_law_exponent']) == int or type(kwargs['power_law_exponent']) == float:
                if kwargs['power_law_exponent'] > 0:
                    raise ValueError('[soens_sim] power_law_exponent associated with synaptic decay must be a real number between negative infinity and zero')
                else:
                     self.add_power_law_exponent(kwargs['power_law_exponent'])
        else:
            if self.loop_temporal_form == 'power_law':
                _gamma_default = -1
                self.add_power_law_exponent(_gamma_default)
                
        if 'integration_loop_inductance' in kwargs:
            if type(kwargs['integration_loop_inductance']) == int or type(kwargs['integration_loop_inductance']) == float:
                if kwargs['integration_loop_inductance'] < 0:
                    raise ValueError('[soens_sim] integration_loop_inductance associated with synaptic integration loop must be a real number between zero and infinity (units of henries)')
                else:
                     self.integration_loop_inductance = kwargs['integration_loop_inductance']
        else:
            _integration_loop_inductance_default = 10e-9 #units of henries
            self.integration_loop_inductance = _integration_loop_inductance_default
                 
        if 'synaptic_bias_current' in kwargs:
            if type(kwargs['synaptic_bias_current']) == int or type(kwargs['synaptic_bias_current']) == float:
                if kwargs['synaptic_bias_current'] < 0:
                    raise ValueError('[soens_sim] synaptic_bias_current associated with synaptic integration loop must be a real number between zero and infinity (units of henries)')
                else:
                     self.synaptic_bias_current = kwargs['synaptic_bias_current']
        else:
            _synaptic_bias_current_default = 35e-6 #units of amps
            self.synaptic_bias_current = _synaptic_bias_current_default
                            
        if 'loop_bias_current' in kwargs:
            if type(kwargs['loop_bias_current']) == int or type(kwargs['loop_bias_current']) == float:
                if kwargs['loop_bias_current'] < 0:
                    raise ValueError('[soens_sim] loop_bias_current associated with synaptic integration loop must be a real number between zero and infinity (units of henries)')
                else:
                     self.loop_bias_current = kwargs['loop_bias_current']
        else:
            _loop_bias_current_default = 30e-6 #units of amps
            self.loop_bias_current = _loop_bias_current_default            
        
        # self.neuronal_connections = {} #[unique_label (input_neuron or input_signal), unique_label (output_neuron)] (label of neurons from which synapse receives and to which synapse connects)
        # self.input_spike_times = {} #list of real numbers (obtained from spike_times of neuronal_connection)
        # self.loop_integrated_current = {} #real function of time (I_si, amps; output variable)

        self.uid = synapse._next_uid
        
        self.unique_label = 's'+str(self.uid)
        
        synapse._next_uid += 1
      
    def add_time_constant(self, tau):
        if self.loop_temporal_form == 'exponential':
            self.time_constant = tau
        else:
            raise ValueError('[soens_sim] Tried to assign a time constant to a synapse without exponential leak in synapse %s (unique_label = %s)' % (self.name, self.unique_label))
      
    def add_power_law_exponent(self, gamma):
        if self.loop_temporal_form == 'power_law':
            self.power_law_exponent = gamma
        else:
            raise ValueError('[soens_sim] Tried to assign a power-law exponent to a synapse without power-law leak in synapse %s (unique_label = %s)' % (self.name, self.unique_label))
        
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
    
# class neuron:
#     neuron.unique_label = 'n'+int
#     neuron.input_connections = {unique_label} (list of synapse/dendrite labels)
#     neuron.input_mutual_inductances = real (list of mutual inductances between synapses/dendrites and neuron)    
#     neuron.refractory_temporal_form = 'exp' or 'power_law' (behaves like synaptic or dendritic connection)
#     neuron.refractory_loop_current = real (I_ref, amps; self-feedback variable)
#     neuron.threshold_bias_current = real (I_th, amps)
#     neuron.cell_body_circulating_current = real (sum of synaptic/dendritic inputs and refractory self feedback)
#     neuron.spike_times = list of real numbers (times cell_body_circulating_current crossed threshold current with positive derivative; the main dynamical variable and output of the neuron)
#     neuron.is_output_neuron = True or False (whether or not the neuron communicates to the outside world)

# class network:
#     network.unique_label = 'net'+int
#     network.neurons = {unique_label} (list of neuron labels)
#     network.networks = {unique_label} (list of sub-network labels)
#     network.num_neurons = int (N_tot, total number of neurons in network)
#     network.adjacency_matrix = A (N_tot x N_tot adjacency matrix of the network. can be obtained from list of all neuron synaptic connections)
#     network.graph_metrics.{degree_distribution, clustering_coefficient, avg_path_length, etc}

# class input_signal:
# 	input_signal.unique_label = 'in'+int
# 	input_signal.temporal_form = 'single_pulse' or 'constant_rate' or 'arbitrary_pulse_train'
# 	input_signal.pulse_times = real (if 'single_pulse') or [start_time,rep_rate] (if 'constant_rate') or list_of_times (if 'arbitrary_pulse_train')