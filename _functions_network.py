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