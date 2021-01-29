import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from _functions_network import populate_hierarchy__power_law, generate_degree_distribution, generate_spatial_structure
from _plotting_network import plot_hierarchy, plot_out_degree_distribution, plot_node_degree_vs_space

plt.close('all')

#%%

# this script produces the following data structures:
    # hierarchy
    # out_degree_distribution
    # spatial_information

#%% establish network hierarchy

print('constructing network hierarchy ... ')

# power-law hierarchy ( number of modules at level h of hierarchy: M_h = n_0 * h**(-gamma) )
num_levels_hier = 3
num_nodes_0 = 7**2
gamma = 2

hierarchy = populate_hierarchy__power_law(num_nodes_0,num_levels_hier,float(gamma))
plot_hierarchy(hierarchy)

#%% generate out-degree distribution

print('generating out-degree distribution ... ')

degree_distribution__functional_form = 'power_law' # 'gaussian' or 'power_law'
# note that we can let alpha = 0 in the power-law distribution for a random degree distribution or we can let st_dev = 0 in the gaussian distribution for a delta function (all nodes same degree)

if degree_distribution__functional_form == 'gaussian':
    
    # gaussian degree params
    center = 40 # mean of gaussian distribution
    st_dev = 5 # standard deviation of gaussian distribution    
    out_degree_distribution = generate_degree_distribution(degree_distribution__functional_form, center = center, st_dev = st_dev, num_nodes = hierarchy['total_nodes'])
    
elif degree_distribution__functional_form == 'power_law':
    
    # power-law degree params
    k_out_min = 2 # minimum degree allowed
    alpha = 2 # exponent, p(k) ~ k**(-alpha)
    out_degree_distribution = generate_degree_distribution(degree_distribution__functional_form, k_out_min = k_out_min, alpha = alpha, num_nodes = hierarchy['total_nodes'])
    
num_bins = 10    
plot_out_degree_distribution(out_degree_distribution,num_bins)

# avg_num_edges = 40

# total_num_nodes = hierarchy['total_num_nodes']
# tot_num_edges__target = avg_num_edges*total_nodes
# tot_num_edges__current = np.sum(out_degree_distribution['node_degrees'])

# diff = tot_num_edges__target-tot_num_edges__current
# print('diff = {}'.format(diff))
    

#%% place nodes in space

print('assigning spatial coordinates ... ')

spatial_information = generate_spatial_structure(hierarchy,out_degree_distribution)
    
plot_node_degree_vs_space()    
    

#%% establish corner-referred indexing

index__corner_referred = np.zeros([total_nodes])

for ii in range(total_nodes):





#%% make connections

# exponential decay params
decay_length = 5 # units of lattice constant

num_bins = 20 # for plotting





#%% rescale out-degree distribution to achieve exact desired number of edges 

# avg_num_edges = 40

# total_nodes = hierarchy['total_nodes']
# tot_num_edges__target = avg_num_edges*total_nodes
# tot_num_edges__current = np.sum(out_degree_distribution['node_degrees'])

# diff = tot_num_edges__target-tot_num_edges__current
# print('diff = {}'.format(diff))

# scale_factor = tot_num_edges__current/tot_num_edges__target
# out_degree_distribution['node_degrees'] = np.round(out_degree_distribution['node_degrees']/scale_factor)
# tot_num_edges__current = np.sum(out_degree_distribution['node_degrees'])

# diff = tot_num_edges__target-tot_num_edges__current
# print('diff = {}'.format(diff))

# indices = np.random.randint(0,total_nodes-1,np.round(np.abs(diff)).astype(int))
# out_degree_distribution['node_degrees'][indices] += np.sign(diff) 
# tot_num_edges__current = np.sum(out_degree_distribution['node_degrees'])

# diff = tot_num_edges__target-tot_num_edges__current
# print('diff = {}'.format(diff))

# plot_out_degree_distribution(out_degree_distribution,num_bins)




