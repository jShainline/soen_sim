import numpy as np
# from matplotlib import pyplot as plt
import powerlaw

from _util import color_dictionary # physical_constants
colors = color_dictionary()

#%%

def A_random(num_nodes,num_edges):
    
    A = np.zeros([num_nodes,num_nodes])
    connections = np.random.randint(0,num_nodes,size=[2,num_edges])
    print('shape(connections) = {}'.format(np.shape(connections)))
    print(connections)
    for ii in range(num_edges):
        A[connections[0,ii],connections[1,ii]] = 1
    # print('shape(A) = {}'.format(np.shape(A)))    
    
    return A


def populate_hierarchy__power_law(num_nodes_0,num_levels_hier,gamma, plot = True): # power-law hierarchy ( number of modules at level h of hierarchy: M_h = n_0 * h**(-gamma) )
    
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
    
    hierarchy = dict()
    
    hierarchy['num_nodes_0'] = num_nodes_0
    hierarchy['num_levels_hier'] = num_levels_hier
    hierarchy['gamma'] = gamma
    hierarchy['h_vec'] = h_vec
    
    hierarchy['num_modules_list'] = num_modules_list # number of modules at each level of hierarchy
    hierarchy['num_nodes_list'] = num_nodes_list # number of nodes at each level of hierarchy
    hierarchy['num_nodes_per_module'] = num_nodes_per_module # number of nodes per module at each level of hierarchy
    hierarchy['inter_modular_nodes'] = inter_modular_nodes # number of inter-modular nodes at each level of hierarchy
    hierarchy['total_num_nodes'] = total_nodes # total number of nodes in the network         

    return hierarchy


def generate_degree_distribution(out_degree_functional_form = 'power-law', **kwargs):
    
    if 'num_nodes' in kwargs:
        num_nodes = kwargs['num_nodes']
    else:
        raise ValueError('[_functions_network/generate_degree_distribution] You must specify the number of nodes in the network (num_nodes)') 
    
    out_degree_distribution = dict()
    if out_degree_functional_form == 'gaussian':

        if 'center' in kwargs:
            center = kwargs['center']
        else: 
            raise ValueError('[_functions_network/generate_degree_distribution] For a gaussian out-degree distribution, you must specify the mean of the gaussian distribution (center)')
        if 'st_dev' in kwargs:
            st_dev = kwargs['st_dev']
        else: 
            raise ValueError('[_functions_network/generate_degree_distribution] For a gaussian out-degree distribution, you must specify the standard deviation (st_dev)')        
        
        out_degree_distribution['functional_form'] = 'gaussian'
        out_degree_distribution['center'] = center
        out_degree_distribution['st_dev'] = st_dev
        out_degree_distribution['num_nodes'] = int(num_nodes)
        out_degree_distribution['node_degrees'] = np.flipud(np.round(np.sort(np.random.normal(center,st_dev,int(num_nodes))))) # gaussian degree distribution
        
    if out_degree_functional_form == 'power_law':
        
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            raise ValueError('[_functions_network/generate_degree_distribution] For a power-law out-degree distribution, you must specify the exponent (alpha)')
        if 'k_out_min' in kwargs:
            k_out_min = kwargs['k_out_min']
        else:
            raise ValueError('[_functions_network/generate_degree_distribution] For a power-law out-degree distribution, you must specify the minimum out degree (k_out_min)')           
        
        out_degree_distribution['functional_form'] = 'power_law'
        out_degree_distribution['k_out_min'] = k_out_min
        out_degree_distribution['alpha'] = alpha
        out_degree_distribution['num_nodes'] = int(num_nodes)
        out_degree_distribution['node_degrees'] = np.flipud(np.round(np.sort(powerlaw.Power_Law(xmin = k_out_min, parameters = [alpha]).generate_random(int(num_nodes))))) # power-law degree distribution
        
    return out_degree_distribution


def generate_spatial_structure(hierarchy,out_degree_distribution):

    num_row_col = np.sqrt(total_num_nodes).astype(int)
    central_node_index = np.round( (np.sqrt(total_num_nodes)-1)/2 +1 )
    c_coords = [central_node_index-1,central_node_index-1]
    coords_list = []
    for ii in range(num_row_col):
        for jj in range(num_row_col):
            coords_list.append(np.asarray([ii,jj]))
    
    coords_list_full = copy.deepcopy(coords_list) # make backup 
    node_coords = []
    node_x_coords = []
    node_y_coords = []
    for ii in range(num_nodes):
        
        distance_list = np.zeros([len(coords_list)])
        for jj in range(len(coords_list)):        
            distance_list[jj] = ( (coords_list[jj][0]-c_coords[0])**2 + (coords_list[jj][1]-c_coords[1])**2 )**(1/2) # euclidean distance
            
        ind = np.argmin( distance_list )
        node_coords.append( coords_list[ind] )
        node_x_coords.append(node_coords[-1][0])
        node_y_coords.append(node_coords[-1][1])
        coords_list = np.delete(coords_list,ind,0) 
    
    return spatial_information

#%% scraps



# proximity_factor = 1
# A = np.zeros([num_nodes,num_nodes])
# for ii in range(num_nodes):
#     k_out_ii = node_degrees[ii].astype(int)
#     r_out_ii__vec = np.random.exponential(decay_length,k_out_ii) # exponential spatial decay
#     # print('ii = {} of {}, k_out_ii = {}, len(r_out_ii__vec) = {}'.format(ii+1,num_nodes,k_out_ii,len(r_out_ii__vec)))
#     for r_out_ii in r_out_ii__vec:     
#         tracker = 0
#         candidate_nodes = np.where( np.abs( R_mat[ii,:] - r_out_ii ) <= proximity_factor  )[0]
#         # print('here0')
#         if len(candidate_nodes) > 0:
#             while tracker == 0:
                
#                 # print('len(candidate_nodes) = {}'.format(len(candidate_nodes)))
#                 rand_ind = np.random.randint(0,len(candidate_nodes),1)
#                 # print('candidate_nodes = {}, rand_ind[0] = {}, candidate_nodes[rand_ind[0]] = {}'.format(candidate_nodes,rand_ind[0],candidate_nodes[rand_ind[0]]))
    
#                 if A[ii,candidate_nodes[rand_ind[0]]] == 0:
#                     A[ii,candidate_nodes[rand_ind[0]]] = 1
#                     tracker = 1
#                     # print('here1')
#                 elif A[ii,candidate_nodes[rand_ind[0]]] == 1:
#                     candidate_nodes = np.delete(candidate_nodes,rand_ind[0])
#                     # print('here2')
#                     if len(candidate_nodes) == 0:
#                         tracker = 1

# plot_A(A)