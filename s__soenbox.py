import numpy as np
import copy
import time

from f__physical_constants import physical_constants

from soens_sim import synapse

#%%

synapse_1 = synapse('test_synapse__exp', loop_temporal_form = 'exponential', time_constant = 200e-9, integration_loop_inductance = 10e-9, synaptic_bias_current = 33e-6)
synapse_2 = synapse('test_synapse__power_law', loop_temporal_form = 'power_law', power_law_exponent = -1.1, integration_loop_inductance = 100e-9, synaptic_bias_current = 37e-6)



# synapse_2 = synapse('test_name_2')
# synapse_2.add_time_constant(tau = 51e-9)