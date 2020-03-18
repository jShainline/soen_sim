class synapse():
    
    _next_uid = 0
    
    def __init__(self):
        
        #make new synapse
        self.unique_label = {} #'s'+int
        self.neuronal_connections = {} #[unique_label (input_neuron or input_signal), unique_label (output_neuron)] (label of neurons from which synapse receives and to which synapse connects)
        self.input_spike_times = {} #list of real numbers (obtained from spike_times of neuronal_connection)
        self.loop_temporal_form = {} #'exponential' or 'power_law'
        self.time_constant = {} #real (if 'exponential')
        self.power_law_exponent = {} #real (-inf,0) (if 'power_law')
        self.integration_loop_inductance = {} #real (L_si, henries)
        self.synaptic_bias_current = {} #real, possibly function of time (I_sy, amps)
        self.loop_bias_current = {} # real, possibly function of time (I_b, amps)
        self.loop_integrated_current = {} #real function of time (I_si, amps; output variable)
        self.uid = synapse._next_uid
        
        synapse._next_uid += 1
        
class dendrite:
	dendrite.unique_label = 'd'+int
	dendrite.input_connections = {unique_label} (list of input synapse/dendrite labels)
	dendrite.input_mutual_inductances = real (list of mutual inductances between synapses/dendrites and dendrite) 
	dendrite.loop_temporal_form = 'exponential' or 'power_law'
    dendrite.time_constant = real (if 'exponential')
    dendrite.power_law_exponent = real (-inf,0) (if 'power_law')
    dendrite.integration_loop_inductance = real (L_si, henries)
    dendrite.synaptic_bias_current = real (I_sy, amps)
    dendrite.loop_bias_current = real (I_b, amps)
    dendrite.loop_integrated_current = real function of time (I_si, amps; output variable)   
    
class neuron:
    neuron.unique_label = 'n'+int
    neuron.input_connections = {unique_label} (list of synapse/dendrite labels)
    neuron.input_mutual_inductances = real (list of mutual inductances between synapses/dendrites and neuron)    
    neuron.refractory_temporal_form = 'exp' or 'power_law' (behaves like synaptic or dendritic connection)
    neuron.refractory_loop_current = real (I_ref, amps; self-feedback variable)
    neuron.threshold_bias_current = real (I_th, amps)
    neuron.cell_body_circulating_current = real (sum of synaptic/dendritic inputs and refractory self feedback)
    neuron.spike_times = list of real numbers (times cell_body_circulating_current crossed threshold current with positive derivative; the main dynamical variable and output of the neuron)
    neuron.is_output_neuron = True or False (whether or not the neuron communicates to the outside world)

class network:
    network.unique_label = 'net'+int
    network.neurons = {unique_label} (list of neuron labels)
    network.networks = {unique_label} (list of sub-network labels)
    network.num_neurons = int (N_tot, total number of neurons in network)
    network.adjacency_matrix = A (N_tot x N_tot adjacency matrix of the network. can be obtained from list of all neuron synaptic connections)
    network.graph_metrics.{degree_distribution, clustering_coefficient, avg_path_length, etc}

class input_signal:
	input_signal.unique_label = 'in'+int
	input_signal.temporal_form = 'single_pulse' or 'constant_rate' or 'arbitrary_pulse_train'
	input_signal.pulse_times = real (if 'single_pulse') or [start_time,rep_rate] (if 'constant_rate') or list_of_times (if 'arbitrary_pulse_train')