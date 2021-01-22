import numpy as np
import pickle
import time

from _util import physical_constants

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