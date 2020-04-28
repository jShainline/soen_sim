#%%
import numpy as np
import time

from _functions import read_wr_data
from _plotting import plot_wr_data, plot_wr_data__currents_and_voltages

#%%
directory = 'wrspice_data'
file_name = 'dend__cnst_drv__Idrv19.0uA_Ib35.0uA_Ldi0077.5nH_taudi0077.5ms_tsim_100ns.dat'
data_to_plot = ['L3#branch','L4#branch','L8#branch','v(3)']#'L0#branch','L1#branch','L2#branch',
plot_save_string = False

data_dict = read_wr_data(directory+'/'+file_name)
data_dict['file_name'] = file_name
plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string)


