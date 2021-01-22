import numpy as np
import networkx as nx

from _functions_network import A_random
from _plotting_network import plot_A

from matplotlib import pyplot as plt
import powerlaw

from _util import color_dictionary, physical_constants
colors = color_dictionary()

plt.close('all')


#%% random matrix without networkx                  

num_nodes = 10
num_edges = 20

A1 = A_random(num_nodes,num_edges)
plot_A(A1)
ne1 = np.sum(A1)
print('ne1 = {}'.format(ne1))

G1 = nx.from_numpy_matrix(A1, create_using=nx.DiGraph())
# G = nx.DiGraph(A) # this also works
A1p = nx.to_numpy_matrix(G1)
plot_A(A1p)
ne2 = G1.number_of_edges()
print('ne2 = {}'.format(ne2))

# nx.draw(G)

#%% Erdos-Renyi with networkx
G = nx.erdos_renyi_graph(num_nodes, num_edges/num_nodes**2, directed = True)
# nx.draw(G)
ne = G.number_of_edges()
print('ne = {}'.format(ne))

A = nx.to_numpy_matrix(G)
plot_A(A)

    


#%% k-nearest neighbors

#%% modular via rewiring

#%% growth model

