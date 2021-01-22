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

#%% general inputs

num_nodes = 33**2 # best if this is an odd number squared so the x-y layout is a square grid of nodes with a center node that has the highest degree
num_row_col = np.sqrt(num_nodes).astype(int)

#%% generate degree distribution

print('generating degree distribution ... ')

# gaussian degree params
center = 40
st_dev = 5

# exponential decay params
decay_length = 5 # units of lattice constant

num_bins = 20 # for plotting

node_degrees = np.flipud(np.round(np.sort(np.random.normal(center,st_dev,num_nodes)))) # gaussian degree distribution

fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False)
fig.suptitle('gaussian degree distribution; center = {:5.2f}, standard deviation = {:5.2f}'.format(center,st_dev))

degree_vec = np.linspace(1,num_nodes,num_nodes)
color_list = ['blue3','red3','green3','yellow3']
ax[0].plot(degree_vec,node_degrees, '-', color = colors['blue3'])
ax[0].set_xlabel(r'Node Index')
ax[0].set_ylabel(r'Node Degree')
# ax[0].legend()

degree_hist, bin_edges = np.histogram(node_degrees,num_bins)
bin_centers = bin_edges[0:-1]+np.diff(bin_edges)/2
ax[1].plot(bin_centers,degree_hist, '-o', color = colors['blue3'])
ax[1].set_xlabel(r'bin center value')
ax[1].set_ylabel(r'bin frequency')
# ax[1].legend()

plt.show()

#%% assign spatial coordinates

print('assigning spatial coordinates ... ')

central_node_index = np.round( (np.sqrt(num_nodes)-1)/2 +1 )
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
    
fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)
fig.suptitle('x-y positions of nodes')

degree_vec = np.linspace(1,num_nodes,num_nodes)
ax.plot(node_x_coords,node_y_coords, '-o', color = colors['blue3'])
ax.plot(node_x_coords[0],node_y_coords[0], '-o', color = colors['green3'], label = 'first')
ax.plot(node_x_coords[1],node_y_coords[1], '-o', color = colors['yellow3'], label = 'second')
ax.plot(node_x_coords[-1],node_y_coords[-1], '-o', color = colors['red3'], label = 'last')
ax.set_xlabel(r'x coord')
ax.set_ylabel(r'y coord')
ax.set_xlim([-1,num_row_col])
ax.set_ylim([-1,num_row_col])
ax.legend()
    
degree__mn = np.zeros([num_row_col,num_row_col])
mm_ii = node_x_coords
nn_ii = node_y_coords
for ii in range(num_nodes):
    # print('mm_ii = {}, nn_ii = {}'.format(mm_ii,nn_ii))
    degree__mn[mm_ii[ii],nn_ii[ii]] = node_degrees[ii]

fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = False, sharey = False)

degree = ax.imshow(np.transpose(degree__mn[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[0,num_row_col-1,0,num_row_col-1], aspect = 'auto', origin = 'lower')
cbar = fig.colorbar(degree, extend='both')
cbar.minorticks_on()     
fig.suptitle('designed node degrees vs x-y positions')
ax.set_xlabel(r'x coord')
ax.set_ylabel(r'y coord')   
plt.show()

plt.show()


#%% generate distance matrix

print('generating distance matrix ... ')

distance_mat = np.zeros([num_nodes,num_nodes])

for ii in range(num_nodes):
    for jj in range(num_nodes):
        distance_mat[ii,jj] = np.abs(mm_ii[ii]-mm_ii[jj])+np.abs(nn_ii[ii]-nn_ii[jj]) # manhattan distance
        # distance_mat[ii,jj] = ( (mm_ii[ii]-mm_ii[jj])**2 + (nn_ii[ii]-nn_ii[jj])**2 )**(1/2) # euclidean distance

fig, ax = plt.subplots(1,1)
error = ax.imshow(np.transpose(distance_mat[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[0,num_nodes-1,0,num_nodes-1], aspect = 'auto', origin = 'lower')
cbar = fig.colorbar(error, extend='both')
cbar.minorticks_on()     
fig.suptitle('R_mat')
ax.set_xlabel(r'node index 1')
ax.set_ylabel(r'node index 2')   
plt.show()


#%% generate adjacency matrix

print('generating adjacency matrix ... ')

A = np.zeros([num_nodes,num_nodes])

for ii in range(num_nodes):

    # make connections by proximity
    min_indices = distance_mat[ii,:].argsort()[:(node_degrees[ii].astype(int)+1)]
    A[ii,min_indices] = 1
        
for ii in range(num_nodes):
    A[ii,ii] = 0

plot_A(A)


#%% make digraph

print('generating digraph object with networkx ... ')

G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())

out_degree_list = np.zeros([num_nodes])
out_degree_mat = np.zeros([num_row_col,num_row_col])
in_degree_list = np.zeros([num_nodes])
in_degree_mat = np.zeros([num_row_col,num_row_col])

for ii in range(num_nodes):
    out_degree_list[ii] = G.out_degree[ii] # np.sum(A[ii,:]) # 
    out_degree_mat[mm_ii[ii],nn_ii[ii]] = out_degree_list[ii]
    in_degree_list[ii] = G.in_degree[ii] # np.sum(A[:,ii]) # 
    in_degree_mat[mm_ii[ii],nn_ii[ii]] = in_degree_list[ii]


fig, ax = plt.subplots(1,2)
fig.suptitle('Final network degree distributions versus space')

in_degree = ax[0].imshow(np.transpose(in_degree_mat[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[0,num_row_col-1,0,num_row_col-1], aspect = 'auto', origin = 'lower')
cbar = fig.colorbar(in_degree, ax = ax[0], extend='both')
cbar.minorticks_on()     
ax[0].set_xlabel(r'node x coord')
ax[0].set_ylabel(r'node y coord')  
ax[0].set_title('in_degree') 

out_degree = ax[1].imshow(np.transpose(out_degree_mat[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[0,num_row_col-1,0,num_row_col-1], aspect = 'auto', origin = 'lower')
cbar = fig.colorbar(out_degree, ax = ax[1], extend='both')
cbar.minorticks_on()     
ax[1].set_xlabel(r'node x coord')
ax[1].set_ylabel(r'node y coord')  
ax[1].set_title('out_degree') 

plt.show()

#%% plot degree distributions

print('plotting final degree distribution ... ')

# num_bins_degree = 10    

fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False)
fig.suptitle('Total number of nodes: {}; total number of edges: {}\ngaussian degree distribution: center = {:5.2f}, standard deviation = {:5.2f}; exponential spatial decay, decay_length = {:5.2f}'.format(len(G),len(G.edges),center,st_dev,decay_length))

# in_degree_hist, in_degree_bin_edges = np.histogram(in_degree_list,num_bins_degree)
in_degree_hist, in_degree_bin_edges = np.histogram(in_degree_list)
in_degree_bin_centers = in_degree_bin_edges[0:-1]+np.diff(in_degree_bin_edges)/2
ax[0].plot(in_degree_bin_centers,in_degree_hist, '-o', color = colors['blue3'])
ax[0].set_xlabel(r'bin center value')
ax[0].set_ylabel(r'bin frequency')
ax[0].set_title('in-degree')

# out_degree_hist, out_degree_bin_edges = np.histogram(out_degree_list,num_bins_degree)
out_degree_hist, out_degree_bin_edges = np.histogram(out_degree_list)
out_degree_bin_centers = out_degree_bin_edges[0:-1]+np.diff(out_degree_bin_edges)/2
ax[1].plot(out_degree_bin_centers,out_degree_hist, '-o', color = colors['blue3'])
ax[1].set_xlabel(r'bin center value')
ax[1].set_ylabel(r'bin frequency')
ax[1].set_title('out-degree')

# ax[1].legend()

plt.show()
 
# nx.draw(G)

#%% random matrix without networkx                  

# num_nodes = 10
# num_edges = 20

# A1 = A_random(num_nodes,num_edges)
# plot_A(A1)
# ne1 = np.sum(A1)
# print('ne1 = {}'.format(ne1))

# G1 = nx.from_numpy_matrix(A1, create_using=nx.DiGraph())
# # G = nx.DiGraph(A) # this also works
# A1p = nx.to_numpy_matrix(G1)
# plot_A(A1p)
# ne2 = G1.number_of_edges()
# print('ne2 = {}'.format(ne2))

# nx.draw(G)

#%% Erdos-Renyi with networkx
# G = nx.erdos_renyi_graph(num_nodes, num_edges/num_nodes**2, directed = True)
# # nx.draw(G)
# ne = G.number_of_edges()
# print('ne = {}'.format(ne))

# A = nx.to_numpy_matrix(G)
# plot_A(A)

#%% k-nearest neighbors

#%% modular via rewiring

#%% growth model

#%% scratch
# rand_ind = np.random.randint(0,3,1)
# print(rand_ind[0])
        