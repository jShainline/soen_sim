#%%
import numpy as np
from matplotlib import pyplot as plt
import powerlaw

from _util import color_dictionary, physical_constants
colors = color_dictionary()

# plt.close('all')

#%% random

lower_bound = 10 
upper_bound = 1000
num_samp = 1000
num_runs = 4
num_bins = 40

samples = np.random.randint(lower_bound,upper_bound,[num_samp,num_runs])

fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False)
fig.suptitle('random distribution; lower_bound = {:d}, upper_bound = {:d}'.format(lower_bound,upper_bound))

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

#%% exponential

beta = 10 # exp(-x/beta), beta > 0
num_samp = 1000
num_runs = 4
num_bins = 40

samples = np.random.exponential(beta,[num_samp,num_runs])

fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False)
fig.suptitle('exponential distribution; beta = {:5.2f}'.format(beta))

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

#%% gaussian

center = 10
st_dev = 2
num_samp = 1000
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

#%% power law

exponent = 1.5
x_min = 1
num_samp = 10000
num_runs = 4
num_bins = 40

samples = np.zeros([num_samp,num_runs])
samp_vec = np.linspace(1,num_samp,num_samp)

color_list = ['blue3','red3','green3','yellow3']
fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False)
fig.suptitle('power-law distribution; x_min = {:5.2f}, exponent = {:5.2f}'.format(x_min,exponent))

bin_min = 1e9
bin_max = 0
for ii in range(num_runs):
    samples[:,ii] = powerlaw.Power_Law(xmin = x_min, parameters = [exponent]).generate_random(num_samp)

    bin_min_ii, bin_max_ii = np.min(samples[:,ii]), np.max(samples[:,ii])
    bins = 10**(np.linspace(np.log10(bin_min_ii), np.log10(bin_max_ii), num_bins))
    samp_hist, bin_edges = np.histogram(samples[:,ii], bins, density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2.
    
    ax[0].semilogy(samp_vec,samples[:,ii], '-', color = colors[color_list[ii]], label = 'trial {}'.format(ii+1))    
    ax[1].loglog(bin_centers,samp_hist, 'o', color = colors[color_list[ii]], label = 'trial {}'.format(ii+1))
    
    if bin_min_ii < bin_min:
        bin_min = bin_min_ii
    if bin_max_ii > bin_max:
        bin_max = bin_max_ii

bins_dense = np.linspace(bin_min, bin_max, 10000)
ax[1].loglog(bins_dense, [(exponent-1)*x_min**(exponent-1)*x**(-exponent) for x in bins_dense], color = colors['red3'], label = 'fit')

ax[0].set_xlabel(r'Sample #')
ax[0].set_ylabel(r'function value #')
ax[0].legend()

ax[1].set_xlabel(r'bin center value')
ax[1].set_ylabel(r'bin occupation')
ax[1].legend()

plt.show()

#%% log-normal

mean = 3
st_dev = 1
num_samp = 1000
num_runs = 4
num_bins = 100

samples = np.random.lognormal(mean,st_dev,[num_samp,num_runs])

fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False)
fig.suptitle('log-normal distribution; mean = {:5.2f}, standard deviation = {:5.2f}'.format(center,st_dev))

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
    ax[1].loglog(bin_centers,samp_hist, '-o', color = colors[color_list[ii]], label = 'trial {}'.format(ii+1))
    # ax[1].plot(bin_centers,samp_hist, '-o', color = colors[color_list[ii]], label = 'trial {}'.format(ii+1))
ax[1].set_xlabel(r'bin center value')
ax[1].set_ylabel(r'bin occupation')
ax[1].legend()

plt.show()