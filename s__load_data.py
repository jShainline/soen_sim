#%%
import pickle

#%%
with open('_circuit_data/master__dnd__rate_matrix.soen', 'rb') as data_file:         
        data_array_imported = pickle.load(data_file)
    
I_di_list__imported = data_array_imported['I_di_list']
I_drive_vec__imported__dend = data_array_imported['I_drive_vec']
master_rate_matrix__imported__dend = data_array_imported['master_rate_matrix']

dt = 100e-12
file_string__spd = 'master__syn__spd_response__3jj__dt{:04.0f}ps.soen'.format(dt*1e6)
file_string__rate_array = 'master__syn__rate_array__3jj__Isipad0010nA.soen'
with open('_circuit_data/{}'.format(file_string__rate_array), 'rb') as data_file:         
        data_array__rate = pickle.load(data_file)                        
        
I_si_array = data_array__rate['I_si_array'] # entries have units of uA
I_drive_list__syn = data_array__rate['I_drive_list'] # entries have units of uA
rate_array__syn = data_array__rate['rate_array'] # entries have units of fluxons per microsecond