import numpy as np
import networkx as nx
import copy

from _functions_network import A_random
from _plotting_network import plot_A

from matplotlib import pyplot as plt
import powerlaw

from _util import color_dictionary, physical_constants
colors = color_dictionary()

plt.close('all')

#%%

num_levels_hier = 5
num_nodes_0 = 7**2

gamma = 0

num_modules_list = np.zeros([num_levels_hier])
num_nodes_list = np.zeros([num_levels_hier])
num_nodes_list[0] = 1

h_vec = np.arange(1,num_levels_hier+1,1)
for h in h_vec:
    num_modules_list[h-1] = np.round( num_nodes_0 * h**(-gamma) ) # num_nodes_0 * h**(-gamma) # 
    num_nodes_list[h-1] = ( num_nodes_0**h )*( np.prod(np.arange(1,h+1,1)**(-gamma)) ) # num_nodes_0*( np.prod(num_nodes_0*np.arange(2,h+1,1)**(-gamma)) ) # 

num_nodes_per_module = num_nodes_list/num_modules_list    
total_nodes = num_nodes_list[-1]
inter_modular_nodes = num_nodes_list*(1-1/num_modules_list)
    
#%% plot

fig, ax = plt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = False)
fig.suptitle('Population of network hierarchy, power-law construction\nnum_levels_hier = {:d}, num_nodes_0 = {:d}, gamma = {:5.2f}, num_mod_H = {:d}, \nTotal nodes = {:5.2e}'.format(num_levels_hier,num_nodes_0,gamma,num_modules_list[-1].astype(int),total_nodes))

ax[0,0].plot(h_vec,num_modules_list, '-o', color = colors['blue3'])
ax[0,0].set_xlabel(r'Hierarchy Level')
ax[0,0].set_ylabel(r'Num Modules')
# ax[0].set_ylim([0,num_nodes_0*1.1])
# ax[0].legend()

ax[0,1].semilogy(h_vec,num_nodes_per_module, '-o', color = colors['blue3'])
ax[0,1].set_xlabel(r'Hierarchy Level')
ax[0,1].set_ylabel(r'Neurons per module at this level of hierarchy')

ax[1,0].semilogy(h_vec,num_nodes_list, '-o', color = colors['blue3'])
ax[1,0].set_xlabel(r'Hierarchy Level')
ax[1,0].set_ylabel(r'Total neurons at this level of hierarchy')
# ax[1].legend()

ax[1,1].semilogy(h_vec,inter_modular_nodes, '-o', color = colors['blue3'])
ax[1,1].set_xlabel(r'Hierarchy Level')
ax[1,1].set_ylabel(r'Number of inter-modular neurons at this level of hierarchy')

plt.show()