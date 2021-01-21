import numpy as np
import networkx as nx

from _functions_network import A_random
from _plotting_network import plot_A

from matplotlib import pyplot as plt
import powerlaw

from _util import color_dictionary, physical_constants
colors = color_dictionary()

plt.close('all')

#%% generate R_mat, distance matrix

num_nodes = 32**2
num_row_col = np.sqrt(num_nodes)

R_mat = np.zeros([num_nodes,num_nodes])

for ii in range(num_nodes):
    mm_ii = np.floor(ii/num_row_col)
    nn_ii = ii-num_row_col*mm_ii
    for jj in range(num_nodes):
        mm_jj = np.floor(jj/num_row_col)
        nn_jj = jj-num_row_col*mm_jj
        R_mat[ii,jj] = np.abs(mm_ii-mm_jj)+np.abs(nn_ii-nn_jj)


fig, ax = plt.subplots(1,1)
error = ax.imshow(np.transpose(R_mat[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[0,num_nodes-1,0,num_nodes-1], aspect = 'auto', origin = 'lower')
cbar = fig.colorbar(error, extend='both')
cbar.minorticks_on()     
fig.suptitle('R_mat')
ax.set_xlabel(r'node index 1')
ax.set_ylabel(r'node_index 2')   
plt.show()

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

#%% gaussian degree distribution

center = 10
st_dev = 2
num_samp = num_nodes
num_runs = 4
num_bins = 40

samples = np.random.normal(center,st_dev,[num_samp,num_runs])

fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False)
fig.suptitle('gaussian distribution; center = {:5.2f}, standard deviation = {:5.2f}'.format(center,st_dev))

samp_vec = np.linspace(1,num_samp,num_samp)
color_list = ['blue3','red3','green3','yellow3']
for ii in range(num_runs):
    ax[0].plot(samp_vec,samples[:,ii], '-', color = colors[color_list[ii]], label = 'trial {}'.format(ii+1))
ax[0].set_xlabel(r'Sample #')
ax[0].set_ylabel(r'function value #')
ax[0].legend()

for ii in range(num_runs):
    samp_hist, bin_edges = np.histogram(samples[:,ii],num_bins)
    bin_centers = bin_edges[0:-1]+np.diff(bin_edges)/2
    ax[1].plot(bin_centers,samp_hist, '-o', color = colors[color_list[ii]], label = 'trial {}'.format(ii+1))
ax[1].set_xlabel(r'bin center value')
ax[1].set_ylabel(r'bin occupation')
ax[1].legend()

plt.show()





#%% k-nearest neighbors

#%% modular via rewiring

#%% growth model

