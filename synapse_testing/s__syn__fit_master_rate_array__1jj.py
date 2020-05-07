#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import pickle
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import numpy.matlib
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_syn_rate_array, plot_syn_rate_array__waterfall, plot_syn_rate_array__fit_cmpr
from _functions import save_session_data, load_session_data, read_wr_data, syn_1jj_rate_fit, V_fq__fit, inter_fluxon_interval__fit, inter_fluxon_interval, inter_fluxon_interval__fit_2, inter_fluxon_interval__fit_3
from util import physical_constants
p = physical_constants()

# plt.close('all')

#%% load master rate array
I_sy = 40
with open('../_circuit_data/master__syn__1jj__rate_array__Isipad0010nA.soen', 'rb') as data_file:         
    data_array_imprt = pickle.load(data_file)
    
# data_array_imported = load_session_data('session_data__master_rate_matrix__syn__2020-04-24_10-24-23.dat')
I_si_array = data_array_imprt['I_si_array']
I_drive_list = data_array_imprt['I_drive_list']
rate_array = data_array_imprt['rate_array']

I_drive_sought = 14.45
I_drive_sought_ind = (np.abs(np.asarray(I_drive_list)-I_drive_sought)).argmin()
I_si_sought = 4.552
I_si_sought_ind = (np.abs(np.asarray(I_si_array[I_drive_sought_ind])-I_si_sought)).argmin()
rate_obtained = rate_array[I_drive_sought_ind][I_si_sought_ind]

print('I_drive_sought = {:7.4f}uA, I_drive_sought_ind = {:d}\nI_si_sought = {:7.4f}uA, I_si_sought_ind = {:d}\nrate_obtained = {:10.4f} fluxons per us'.format(I_drive_sought,I_drive_sought_ind,I_si_sought,I_si_sought_ind,rate_obtained))

#%% fit
mu1_vec = np.zeros([len(I_drive_list)])
mu2_vec = np.zeros([len(I_drive_list)])
V0_vec = np.zeros([len(I_drive_list)])

ii = 11

x_vec = I_sy + I_drive_list[ii] - I_si_array[ii][:]
ind_40 = ( np.abs( x_vec[:]-I_sy ) ).argmin()
y_vec = rate_array[ii][:]

popt, pcov = curve_fit(syn_1jj_rate_fit, x_vec[0:ind_40], y_vec[0:ind_40], bounds = ([0.3, 0.3, 100], [3., 1., 500]))

mu1 = popt[0]
mu2 = popt[1]
V0 = popt[2]
    
#%% plot
plot_syn_rate_array__fit_cmpr(I_si_array,rate_array,I_drive_list,mu1,mu2,V0)
    
#%%
abc_1 = syn_1jj_rate_fit(41,2,0.5,200)

I_drive_sought = 2
I_drive_sought_ind = (np.abs(np.asarray(I_drive_list)-I_drive_sought)).argmin()
I_si_sought = 1
I_si_sought_ind = (np.abs(np.asarray(I_si_array[I_drive_sought_ind])-I_si_sought)).argmin()
abc_2 = rate_array[I_drive_sought_ind][I_si_sought_ind]

