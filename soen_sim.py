import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
from pylab import *

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

    def run_sim(self,time_vec,input_spike_times):

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
        I_0 = (0.06989*I_sy**2-3.948*I_sy+53.73)

        #I_si_sat is actually a function of I_b (loop_current_bias). The fit I_si_sat(I_b) has not yet been performed (20200319)
        I_si_sat = 13

        tau_fall = self.time_constant

        I_si_vec = synaptic_response_function(time_vec,input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall)

        self.I_si = I_si_vec*1e-6

        return self

    def plot_time_trace(self,time_vec,input_spike_times):
        
        fig, axes = plt.subplots(1,1)
        # axes.plot(time_vec*1e6,input_spike_times, 'o-', linewidth = 1, markersize = 3, label = 'input pulses'.format())
        axes.plot(time_vec*1e6,self.I_si*1e6, 'o-', linewidth = 1, markersize = 3, label = 'synaptic response'.format())
        axes.set_xlabel(r'Time [us]', fontsize=20)
        axes.set_ylabel(r'Isi [uA]', fontsize=20)
        
        #ylim((ymin_plot,ymax_plot))
        #xlim((xmin_plot,xmax_plot))
        
        axes.legend(loc='best')
        grid(True,which='both')
        title('Synapse: '+self.colloquial_name+' ('+self.unique_label+')'+\
              '\nI_sy = '+str(self.synaptic_bias_current*1e6)+' uA'+\
              '; tau_si = '+str(self.time_constant*1e9)+' ns'+\
              '; L_si = '+str(self.integration_loop_inductance*1e9)+' nH')
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



def synaptic_response_function(time_vec,input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall):

    I_si_mat = np.zeros([len(time_vec),len(input_spike_times)])

    for ii in range(len(input_spike_times)):
        ind_vec = np.argwhere( time_vec > input_spike_times[ii] )
        I_si_vec_temp = np.sum(I_si_mat, axis = 1)
        # I_si_mat(ind_vec(1:end),ii) = f__synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si_vec_temp(ind_vec(ii)-1))*(1-exp(-(time_vec(ind_vec(1:end))-input_spike_times(ii))/tau_rise)).*exp(-(time_vec(ind_vec(1:end))-input_spike_times(ii))/tau_fall);
        for jj in range(len(ind_vec)):
            if time_vec[ind_vec[jj]]-input_spike_times[ii] <= tau_rise:
                I_si_mat[ind_vec[jj],ii] = synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si_vec_temp[ind_vec[jj]-1],tau_rise,tau_fall)*\
                ( (1/tau_rise**gamma3)*( time_vec[ind_vec[jj]] - input_spike_times[ii] )**gamma3 )*\
                np.exp(tau_rise/tau_fall)*\
                np.exp(-(time_vec[ind_vec[jj]]-input_spike_times[ii])/tau_fall)
            else:
                I_si_mat[ind_vec[jj],ii] = synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si_vec_temp[ind_vec[jj]-1],tau_rise,tau_fall)*\
                np.exp(tau_rise/tau_fall)*\
                np.exp(-(time_vec[ind_vec[jj]]-input_spike_times[ii])/tau_fall)
        I_si_vec = np.sum(I_si_mat, axis = 1);

    return I_si_vec

def synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si,tau_rise,tau_fall):

    if I_si >= 0 and I_si < I_si_sat:
        A = I_0
        I_prefactor = min([A*(1-(I_si/I_si_sat)**gamma1)**gamma2, (I_si_sat-I_si)*np.exp(tau_rise/tau_fall)]);
        # I_prefactor = A*(1-log(I_si/I_si_sat)/log(gamma1))^gamma2;
        # I_prefactor = I_0*(I_si_sat-I_si)/I_si_sat
        # I_prefactor = I_0*(1-exp((I_si_sat-I_si)/I_si_sat))
    else:
        I_prefactor = 0

    #print('\n\nI_si = %g',I_si)
    #print('\n\nI_prefactor = %g',I_prefactor)

    return I_prefactor

