#%%
import numpy as np
import pickle

from util import physical_constants

#%%

_temp_str_1 = '_circuit_data/master_dnd_rate_array_'
_temp_str_2 = '{:1d}jj_Llft{:05.2f}_Lrgt{:05.2f}_Ide{:05.2f}'.format(4,20,20,73)

with open('{}{}.soen'.format(_temp_str_1,_temp_str_2), 'rb') as data_file:         
    data_array_imported = pickle.load(data_file)

I_di_array = data_array_imported['I_di_array']
influx_list = data_array_imported['influx_list']
rate_array = data_array_imported['rate_array']

#%%
ind1 = 4
ind2 = 2

print('influx = {}; I_di = {}; rate = {}'.format(influx_list[ind1],I_di_array[ind1][ind2],rate_array[ind1][ind2]))