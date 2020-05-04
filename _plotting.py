import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as gridspec
import pandas as pd

from util import color_dictionary
colors = color_dictionary()


def plot_params():
    
    plot_type = 'large' # 'two_rows' # 'three_rows' # 'four_rows' # 'large' # 'single_frame' # 'four_tiles' #
    
    pp = dict()
    
    if plot_type == 'four_rows':
        
        pp['title_font_size'] = 6
        pp['subtitle_font_size'] = 4
        pp['axes_labels_font_size'] = 8
        pp['axes_labels_pad'] = 0 # 4
        pp['tick_labels_font_size'] = 8
        pp['legend_font_size'] = 8
        pp['nominal_linewidth'] = 1
        pp['fine_linewidth'] = 0.5
        pp['bold_linewidth'] = 1.5
        pp['nominal_markersize'] = 2
        pp['big_markersize'] = 3
        tn = 2*8.6/2.54
        pp['fig_size'] = (tn,1*tn/1.618)
        # pp['fig_size'] = (tn,tn/1.2)
        pp['axes_linewidth'] = 1
        
        pp['major_tick_width'] = 0.75
        pp['major_tick_length'] = 3
        pp['minor_tick_width'] = 0.25
        pp['minor_tick_length'] = 2
        
        pp['xmargin'] = 0 # 0.05 # space between traces and axes
        pp['ymargin'] = 0.05 # 0.05
        
    if plot_type == 'large':
        
        pp['title_font_size'] = 12
        pp['subtitle_font_size'] = 20
        pp['axes_labels_font_size'] = 20
        pp['axes_labels_pad'] = 0 # 4
        pp['tick_labels_font_size'] = 20
        pp['legend_font_size'] = 16
        pp['nominal_linewidth'] = 2
        pp['fine_linewidth'] = 1.5
        pp['bold_linewidth'] = 3
        pp['nominal_markersize'] = 3
        pp['big_markersize'] = 5
        tn = 4*8.6/2.54
        pp['fig_size'] = (tn,tn/1.618)
        # pp['fig_size'] = (tn,tn/1.2)
        pp['axes_linewidth'] = 1
        
        pp['major_tick_width'] = 1.5
        pp['major_tick_length'] = 6
        pp['minor_tick_width'] = 1
        pp['minor_tick_length'] = 4
        
        pp['xmargin'] = 0 # 0.05 # space between traces and axes
        pp['ymargin'] = 0.05 # 0.05
        
    if plot_type == 'two_rows':
        
        pp['title_font_size'] = 10
        pp['subtitle_font_size'] = 8
        pp['axes_labels_font_size'] = 8
        pp['axes_labels_pad'] = 0 # 4
        pp['tick_labels_font_size'] = 8
        pp['legend_font_size'] = 8
        pp['nominal_linewidth'] = 0.75
        pp['fine_linewidth'] = 0.5
        pp['bold_linewidth'] = 2
        pp['nominal_markersize'] = 2
        pp['big_markersize'] = 3
        tn = 1.075*8.6/2.54
        pp['fig_size'] = (tn,2*tn/1.618)
        # pp['fig_size'] = (tn,tn/1.2)
        pp['axes_linewidth'] = 1
        
        pp['major_tick_width'] = 0.75
        pp['major_tick_length'] = 3
        pp['minor_tick_width'] = 0.25
        pp['minor_tick_length'] = 2
        
        pp['xmargin'] = 0 # 0.05 # space between traces and axes
        pp['ymargin'] = 0.05 # 0.05
        
    if plot_type == 'single_frame':
        
        pp['title_font_size'] = 10
        pp['subtitle_font_size'] = 10
        pp['axes_labels_font_size'] = 8
        pp['axes_labels_pad'] = 0 # 4
        pp['tick_labels_font_size'] = 8
        pp['legend_font_size'] = 8
        pp['nominal_linewidth'] = 0.75
        pp['fine_linewidth'] = 0.5
        pp['bold_linewidth'] = 2
        pp['nominal_markersize'] = 2
        pp['big_markersize'] = 3
        tn = 1.075*8.6/2.54
        pp['fig_size'] = (tn,tn/1.618)
        # pp['fig_size'] = (tn,tn/1.2)
        pp['axes_linewidth'] = 1
        
        pp['major_tick_width'] = 0.75
        pp['major_tick_length'] = 3
        pp['minor_tick_width'] = 0.25
        pp['minor_tick_length'] = 2
        
        pp['xmargin'] = 0 # 0.05 # space between traces and axes
        pp['ymargin'] = 0.05 # 0.05
     
    if plot_type == 'three_rows':
        
        pp['title_font_size'] = 14
        pp['subtitle_font_size'] = 10
        pp['axes_labels_font_size'] = 10
        pp['axes_labels_pad'] = 0 # 4
        pp['tick_labels_font_size'] = 10
        pp['legend_font_size'] = 8
        pp['nominal_linewidth'] = 0.75
        pp['fine_linewidth'] = 0.5
        pp['bold_linewidth'] = 2
        pp['nominal_markersize'] = 1
        pp['big_markersize'] = 2
        tn = 1.075*8.6/2.54
        pp['fig_size'] = (tn,3*tn/1.618)
        # pp['fig_size'] = (tn,tn/1.2)
        pp['axes_linewidth'] = 1
        
        pp['major_tick_width'] = 0.75
        pp['major_tick_length'] = 3
        pp['minor_tick_width'] = 0.25
        pp['minor_tick_length'] = 2
        
        pp['xmargin'] = 0.05 # space between traces and axes
        pp['ymargin'] = 0.05   
     
    if plot_type == 'four_tiles':
        
        pp['title_font_size'] = 14
        pp['subtitle_font_size'] = 10
        pp['axes_labels_font_size'] = 10
        pp['axes_labels_pad'] = 0 # 4
        pp['tick_labels_font_size'] = 10
        pp['legend_font_size'] = 8
        pp['nominal_linewidth'] = 1
        pp['fine_linewidth'] = 0.5
        pp['bold_linewidth'] = 2
        pp['nominal_markersize'] = 2
        pp['big_markersize'] = 3
        tn = 17.2/2.54
        pp['fig_size'] = (tn,tn/1.618)
        # pp['fig_size'] = (tn,tn/1.2)
        pp['axes_linewidth'] = 1
        
        pp['major_tick_width'] = 0.75
        pp['major_tick_length'] = 3
        pp['minor_tick_width'] = 0.25
        pp['minor_tick_length'] = 2
        
        pp['xmargin'] = 0.05 # space between traces and axes
        pp['ymargin'] = 0.05
        
    return pp 

pp = plot_params()

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = 'Verdana'#'Computer Modern Sans Serif'

plt.rcParams['figure.figsize'] = pp['fig_size']
plt.rcParams['figure.titlesize'] = pp['title_font_size']
plt.rcParams['figure.autolayout'] = True

plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue3'],colors['red3'],colors['green3'],colors['yellow3']])

# plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
#                                                     colors['red5'],colors['red4'],colors['red3'],colors['red2'],colors['red1'],
#                                                     colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green3'],
#                                                     colors['yellow5'],colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1']])

# plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
#                                                     colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
#                                                     colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
#                                                     colors['red4'],colors['red3'],colors['red2'],colors['red1'],
#                                                     colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
#                                                     colors['green4'],colors['green3'],colors['green2'],colors['green1'],
#                                                     colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
#                                                     colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1']])

# plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],
#                                                     colors['green1'],colors['green2'],colors['green3'],colors['green4'],
#                                                     colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4']])

# plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue1'],colors['blue3'],colors['red1'],colors['red3'],colors['green1'],colors['green3'],colors['yellow1'],colors['yellow3']])
# plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue3'],colors['red3'],colors['green3'],colors['yellow3']])
# plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],colors['blue4'],colors['blue3'],colors['blue2']])
# plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue4']])
plt.rcParams['axes.linewidth'] = pp['axes_linewidth']
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.titlesize'] = pp['subtitle_font_size']
plt.rcParams['axes.labelsize'] = pp['axes_labels_font_size']
plt.rcParams['axes.labelpad'] = pp['axes_labels_pad']
plt.rcParams['axes.xmargin'] = pp['xmargin']
plt.rcParams['axes.ymargin'] = pp['ymargin']
plt.rcParams['axes.titlepad'] = 0

plt.rcParams['legend.fontsize'] = pp['legend_font_size']
plt.rcParams['legend.loc'] = 'best'

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.pad_inches'] = 0

plt.rcParams['xtick.labelsize'] = pp['tick_labels_font_size']
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['xtick.major.bottom'] = True
plt.rcParams['xtick.major.top'] = True
plt.rcParams['xtick.major.size'] = pp['major_tick_length']
plt.rcParams['xtick.major.width'] = pp['major_tick_width']
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.minor.size'] = pp['minor_tick_length']
plt.rcParams['xtick.minor.width'] = pp['minor_tick_width']

plt.rcParams['ytick.labelsize'] = pp['tick_labels_font_size']
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.major.left'] = True
plt.rcParams['ytick.major.right'] = True
plt.rcParams['ytick.major.size'] = pp['major_tick_length']
plt.rcParams['ytick.major.width'] = pp['major_tick_width']
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.minor.size'] = pp['minor_tick_length']
plt.rcParams['ytick.minor.width'] = pp['minor_tick_width']

plt.rcParams['figure.max_open_warning'] = 100


# plt.rcParams[''] = pp['']

def plot_synaptic_integration_loop_current(synapse_instance,time_vec):
    
    input_spike_times = synapse_instance.input_spike_times

    fig, axes = plt.subplots(1,1)
    # axes.plot(time_vec*1e6,input_spike_times, 'o-', linewidth = 1, markersize = 3, label = 'input pulses'.format())
    axes.plot(time_vec*1e6,synapse_instance.I_si*1e6, '-', linewidth = pp['nominal_linewidth'], label = 'synaptic response'.format())
    axes.set_xlabel(r'Time [us]')
    axes.set_ylabel(r'Isi [uA]')

    #ylim((ymin_plot,ymax_plot))
    #xlim((xmin_plot,xmax_plot))

    axes.legend(loc='best')
    # grid(True,which='both')
    title('Synapse: '+synapse_instance.name+' ('+synapse_instance.unique_label+')'+\
          '\nI_sy = '+str(synapse_instance.synaptic_bias_current*1e6)+' uA'+\
          '; tau_si = '+str(synapse_instance.time_constant*1e9)+' ns'+\
          '; L_si = '+str(synapse_instance.integration_loop_total_inductance*1e9)+' nH',fontsize = pp['title_font_size'])
    plt.show()

    return

def plot_receiving_loop_current(neuron_instance,plot_save_string = ''):
        
    tt = time.time()   
    save_str = 'receiving_loop_current__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    time_vec = neuron_instance.time_vec
    #nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('Current in the neuronal receiving loop versus time')
    
    #upper panel, total I_nr
    axs[0].plot(time_vec*1e6,neuron_instance.cell_body_circulating_current*1e6, '-', color = colors['blue_3'], linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = neuron_instance.name+' ('+neuron_instance.unique_label+')')        
    #spike times
    ylim = axs[0].get_ylim()
    for ii in range(len(neuron_instance.spike_times)):
        if ii == len(neuron_instance.spike_times):
            axs[0].plot([neuron_instance.spike_times[ii]*1e6, neuron_instance.spike_times[ii]*1e6], [ylim[0], ylim[1]], ':', color = colors['grey_12'], linewidth = pp['fine_linewidth'], label = 'spike times')
        else:
            axs[0].plot([neuron_instance.spike_times[ii]*1e6, neuron_instance.spike_times[ii]*1e6], [ylim[0], ylim[1]], ':', color = colors['grey_12'], linewidth = pp['fine_linewidth'])
    #threshold
    xlim = axs[0].get_xlim()
    axs[0].plot([xlim[0],xlim[1]], [neuron_instance.thresholding_junction_critical_current*1e6,neuron_instance.thresholding_junction_critical_current*1e6], '-.', color = colors['red_5'], linewidth = pp['fine_linewidth'], label = 'Threshold')
    #obs start
    axs[0].plot([neuron_instance.time_vec[neuron_instance.idx_obs_start]*1e6, neuron_instance.time_vec[neuron_instance.idx_obs_start]*1e6], [ylim[0], ylim[1]], '-.', color = colors['green_5'], linewidth = pp['fine_linewidth'], label = 'Begin observation')
    # axs[0].set_xlabel(r'Time [$\mu$s]', fontsize = pp['axes_labels_font_size'])
    axs[0].set_ylabel(r'$I_{nr}$ [$\mu$A]')
    axs[0].set_title('Total current in NR loop')
    axs[0].legend()
    # axs[0].grid(b = False)
    # axs[0].grid(b = True, which='major', axis='both')

    #lower panel, scaled contributions from each synapse
    color_list = ['red_3','green_3','yellow_3','blue_5','blue_1','red_1','green_1','yellow_1','blue_3','grey_6']
    num_sy = len(neuron_instance.synapses)
    max_vec = np.zeros([num_sy-1,1])
    for ii in range(len(neuron_instance.synapses)-1): 
        max_vec[ii] = max(neuron_instance.synapses[ii].coupling_factor*neuron_instance.synapses[ii].I_si)
    max_response = max(max_vec)
    for ii in range(len(neuron_instance.synapses)):
        jj = num_sy-ii-1
        if jj == num_sy-1:
            if max(neuron_instance.synapses[jj].coupling_factor*neuron_instance.synapses[jj].I_si) > 0:
                tn = max_response/max(neuron_instance.synapses[jj].coupling_factor*neuron_instance.synapses[jj].I_si)
            else:
                tn = 1
        else:
            tn = 1
        axs[1].plot(time_vec*1e6,tn*neuron_instance.synapses[jj].coupling_factor*neuron_instance.synapses[jj].I_si*1e6, '-', color = colors[color_list[jj]], linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = neuron_instance.synapses[jj].unique_label)#'Synapse: '+neuron_instance.synapses[ii].name+' ('+neuron_instance.synapses[ii].unique_label+')'.format()
    axs[1].set_xlabel(r'Time [$\mu$s]')
    axs[1].set_ylabel(r'Contribution to $I_{nr}$ [$\mu$A]')
    axs[1].set_title('Contribution from each synapse')
    axs[1].legend()
    # axs[1].grid(b = False)        
    # axs[1].grid(b = True, which='major', axis='both')
    
    plt.show()       
    fig.savefig('figures/'+save_str+'.png')

    return

def plot_rate_transfer_function(neuron_instance,plot_save_string = ''):
    
    tt = time.time()        
    save_str = 'rate__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('Output rate versus input rate')
    plt.title(plot_save_string)
    
    for qq in range(len(neuron_instance.tau_ref_vec)):
        for jj in range(len(neuron_instance.I_sy_vec)):
            for ii in range(len(neuron_instance.tau_si_vec)):
                # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
                # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
                ax.plot(neuron_instance.rate_vec*1e-6,1e-6*1/neuron_instance.isi_output_avg_mat[:,ii,jj,qq], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'tau_si = {:2.2f}ns'.format(neuron_instance.tau_si_vec[ii]*1e9))
    ylim = ax.get_ylim()
    # print(ylim)
    ax.plot(neuron_instance.rate_vec*1e-6,neuron_instance.rate_vec*1e-6, '-', linewidth = pp['fine_linewidth'], label = 'rate-out equals rate-in')        
    ax.plot(neuron_instance.rate_vec*1e-6,neuron_instance.rate_vec*1e-6/2, '-', linewidth = pp['fine_linewidth'], label = 'rate-out equals rate-in/2')         
    # ylim = ax.get_ylim()
    ax.set_xlabel(r'Input rate [MHz]')
    ax.set_ylabel(r'Output rate [MHz]')
    # ax.set_title('Total current in NR loop')
    ax.legend()
    # ax.grid(b = True, which='major', axis='both')        
    ax.set_ylim([ylim[0],ylim[1]])
    
    plt.show()
    fig.savefig('figures/'+save_str+'.png')
    
    return

def plot_rate_transfer_function__no_lines(neuron_instance,plot_save_string = ''):
    
    tt = time.time()        
    save_str = 'rate__'+plot_save_string+'__no_lines__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('Output rate versus input rate')
    plt.title(plot_save_string)
    
    for jj in range(len(neuron_instance.I_sy_vec)):
        for ii in range(len(neuron_instance.tau_si_vec)):
            # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
            # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
            ax.plot(neuron_instance.rate_vec*1e-6,1e-6*1/neuron_instance.isi_output_avg_mat[:,ii,jj], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'tau_si = {:2.2f}ns'.format(neuron_instance.tau_si_vec[ii]*1e9))        
    ax.set_xlabel(r'Input rate [MHz]')
    ax.set_ylabel(r'Output rate [MHz]')
    ax.legend()
    # ax.grid(b = True, which='major', axis='both')    
    
    plt.show()
    fig.savefig('figures/'+save_str+'.png')  

    return

def plot_num_spikes(neuron_instance,plot_save_string = ''):
    
    tt = time.time()        
    save_str = 'num_spikes_transfer_function__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))

    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('Output rate versus input rate\n'+plot_save_string)
    
    #upper panel, rate_out vs rate_in
    for jj in range(len(neuron_instance.I_sy_vec)):
        for ii in range(len(neuron_instance.tau_si_vec)):
            axs[0].plot(neuron_instance.rate_vec*1e-6,1e-6*1/neuron_instance.isi_output_avg_mat[:,ii,jj], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'mean firing rate; tau_si = {:2.2f}ns, I_sy = {:2.2f}uA'.format(neuron_instance.tau_si_vec[ii]*1e9,neuron_instance.I_sy_vec[jj]*1e6))
    ylim = axs[0].get_ylim()
    axs[0].plot(neuron_instance.rate_vec*1e-6,neuron_instance.rate_vec*1e-6, 'o-', linewidth = pp['fine_linewidth'], markersize = pp['nominal_markersize'], label = 'rate-out equals rate-in')
    axs[0].set_xlabel(r'Input rate [MHz]')
    axs[0].set_ylabel(r'Output rate [MHz]')
    axs[0].legend()
    axs[0].grid(b = True, which='major', axis='both')
    axs[0].set_ylim([ylim[0],ylim[1]])
    
    #lower panel, num_spikes_out vs num_spikes_in
    for jj in range(len(neuron_instance.I_sy_vec)):
        for ii in range(len(neuron_instance.tau_si_vec)):
            # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
            # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
            axs[1].plot(neuron_instance.rate_vec*1e-6,neuron_instance.num_spikes_in_mat[:,ii,jj], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'num_spikes_in')
            axs[1].plot(neuron_instance.rate_vec*1e-6,neuron_instance.num_spikes_out_mat[:,ii,jj], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'num_spikes_out; tau_si = {:2.2f}ns, I_sy = {:2.2f}uA'.format(neuron_instance.tau_si_vec[ii]*1e9,neuron_instance.I_sy_vec[jj]*1e6))        
    axs[1].set_xlabel(r'Input rate [MHz]')
    axs[1].set_ylabel(r'Number of spikes')
    # ax.set_title('Total current in NR loop')
    axs[1].legend()
    # axs[1].grid(b = True, which='major', axis='both')
    
    plt.show()
    fig.savefig('figures/'+save_str+'.png')

    return

def plot_rate_and_isi(neuron_instance,plot_save_string = ''):
    
    pp = plot_params()
    tt = time.time()        
    save_str = 'rate_and_isi__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))

    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
    fig.suptitle('Output versus input transfer function')
    
    #upper panel, rate transfer function
    for ii in range(len(neuron_instance.tau_si_vec)):
        # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
        # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
        axs[0].plot(neuron_instance.rate_vec*1e-6,1e-6*1/neuron_instance.isi_output_avg_mat[:,ii], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'mean firing rate; tau_si = '+str(neuron_instance.tau_si_vec[ii]*1e9)+' ns'.format())            
    axs[0].set_xlabel(r'Input rate [MHz]')
    axs[0].set_ylabel(r'Output rate [MHz]')
    axs[0].legend()
            
    #lower panel, inter-spike interval
    isi_vec = 1/neuron_instance.rate_vec
    for ii in range(len(neuron_instance.tau_si_vec)):
        # ax.plot(rate_vec*1e-6,1e-6*num_spikes_mat[:,ii]/observation_duration, 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'num_spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
        # ax.plot(rate_vec*1e-6,1e-6*1/isi_output_mat[:,ii], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'last two spikes rate; tau_si = '+str(tau_si_vec[ii]*1e9)+' ns'.format())            
        axs[1].plot(isi_vec*1e6,1e6*neuron_instance.isi_output_avg_mat[:,ii], 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'mean inter-spike interval; tau_si = '+str(neuron_instance.tau_si_vec[ii]*1e9)+' ns'.format())            
    axs[1].set_xlabel(r'Input inter-spike interval [$\mu$s]')
    axs[1].set_ylabel(r'Output inter-spike interval [$\mu$s]')
    axs[1].legend()        
            
    plt.show()
    fig.savefig('figures/'+save_str+'.png')

    return 

def plot_spike_train(neuron_instance,plot_save_string):
    
    pp = plot_params()
    tt = time.time()        
    save_str = 'spike_train__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    time_vec = neuron_instance.time_vec
        
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
    fig.suptitle('Input and output spike trains')
    
    #upper panel, total I_nr
    axs[0].plot(time_vec*1e6,neuron_instance.cell_body_circulating_current*1e6, 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'Neuron: '+neuron_instance.name+' ('+neuron_instance.unique_label+')'.format())        

    #threshold
    xlim = axs[0].get_xlim()
    ylim = axs[0].get_ylim()
    axs[0].plot([xlim[0],xlim[1]], [neuron_instance.thresholding_junction_critical_current*1e6,neuron_instance.thresholding_junction_critical_current*1e6], 'g-', linewidth = pp['fine_linewidth'], label = 'Threshold')
    axs[0].plot([neuron_instance.time_vec[neuron_instance.idx_obs_start]*1e6, neuron_instance.time_vec[neuron_instance.idx_obs_start]*1e6], [ylim[0], ylim[1]], 'b-', linewidth = pp['fine_linewidth'], label = 'Begin observation')
    axs[0].set_xlabel(r'Time [$\mu$s]')
    axs[0].set_ylabel(r'$I_{nr}$ [uA]')
    axs[0].set_title('Total current in NR loop')
    axs[0].legend()
            
    #lower panel, synaptic input spikes, neuronal output spikes
    axs[1].plot(time_vec*1e6,neuron_instance.spike_vec,linewidth = pp['nominal_linewidth'], label = 'neuronal spike times'.format())
    axs[1].plot(time_vec*1e6,neuron_instance.synapses[0].spike_vec,linewidth = pp['nominal_linewidth'], label = 'synaptic spike times'.format())        
    axs[1].set_xlabel(r'Time [$\mu$s]')
    axs[1].set_ylabel(r'Spikes [a.u.]')
    axs[1].set_title('Synaptic and neuronal spiking')
    axs[1].legend()       
            
    plt.show()
    fig.savefig('figures/'+save_str+'.png')
    
    return

def plot_fourier_transform(neuron_instance,time_vec):
    
    _sv = neuron_instance.spike_vec
    
    _sv_ft = np.fft.rfft(_sv)
    neuron_instance.spike_vec__ft = _sv_ft
    num_pts = len(_sv_ft)
    temp_vec = np.linspace(0,1,num_pts)

    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
    fig.suptitle('Fourier transform of spike vec')
    
    #upper panel, spike vec
    axs[0].plot(time_vec*1e6,_sv, 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'Neuron: '+neuron_instance.name+' ('+neuron_instance.unique_label+')'.format())        
    axs[0].set_xlabel(r'Time [$\mu$s]')
    axs[0].set_ylabel(r'Spikes [binary]')
    axs[0].set_title('Spike train')
    axs[0].legend()
    
    #lower panel, fourier transform
    axs[1].plot(temp_vec,_sv_ft, 'o-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'Neuron: '+neuron_instance.name+' ('+neuron_instance.unique_label+')'.format())        
    axs[1].set_xlabel(r'frequency')
    axs[1].set_ylabel(r'amplitude')
    axs[1].set_title('fft')
    axs[1].legend()
            
    plt.show()
    save_str = 'spike_vec_fourier_transform__'+neuron_instance.name        
    fig.savefig('figures/'+save_str+'.png')

    return
    
def plot_rate_vs_num_active_synapses(neuron_instance,plot_save_string = ''):
    
    tt = time.time()        
    save_str = 'rate_vs_num_active_synapses__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))

    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('Output rate versus input rate')
    plt.title(plot_save_string)
    
    color_list_1 = ['blue_3','red_3','green_3','yellow_3']
    color_list_2 = ['blue_1','red_1','green_1','yellow_1']
    for ii in range(len(neuron_instance.I_sy_vec)):
        ax.plot(neuron_instance.num_active_synapses_vec,1e-6*1/neuron_instance.isi_output_avg_mat[ii,:], 'o-', color = colors[color_list_1[ii]], linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'I_sy = {:2.2f}uA (ISI averaged)'.format(neuron_instance.I_sy_vec[ii]*1e6))
        ax.plot(neuron_instance.num_active_synapses_vec,1e-6*1/neuron_instance.isi_output_mat[ii,:], 'o-', color = colors[color_list_2[ii]], linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'I_sy = {:2.2f}uA (ISI last two)'.format(neuron_instance.I_sy_vec[ii]*1e6))

    ax.set_xlabel(r'Number of active synapses')
    ax.set_ylabel(r'Output rate [MHz]')
    ax.legend()     
    
    plt.show()
    fig.savefig('figures/'+save_str+'.png')
    
    return

def plot_neuronal_response__single_synaptic_pulse(neuron_instance,plot_save_string = ''):
    
    tt = time.time()   
    save_str = 'bursting__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    time_vec = neuron_instance.time_vec
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('Current in the neuronal receiving loop versus time')
    
    #upper panel, total I_nr
    axs[0].plot(time_vec*1e6,neuron_instance.cell_body_circulating_current*1e6, '-', color = colors['blue_3'], linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = neuron_instance.name+' ('+neuron_instance.unique_label+')')        
    #spike times
    ylim = axs[0].get_ylim()
    for ii in range(len(neuron_instance.spike_times)):
        if ii == len(neuron_instance.spike_times):
            axs[0].plot([neuron_instance.spike_times[ii]*1e6, neuron_instance.spike_times[ii]*1e6], [ylim[0], ylim[1]], ':', color = colors['grey_12'], linewidth = pp['fine_linewidth'], label = 'spike times')
        else:
            axs[0].plot([neuron_instance.spike_times[ii]*1e6, neuron_instance.spike_times[ii]*1e6], [ylim[0], ylim[1]], ':', color = colors['grey_12'], linewidth = pp['fine_linewidth'])
    #threshold
    xlim = axs[0].get_xlim()
    axs[0].plot([xlim[0],xlim[1]], [neuron_instance.thresholding_junction_critical_current*1e6,neuron_instance.thresholding_junction_critical_current*1e6], '-.', color = colors['red_5'], linewidth = pp['fine_linewidth'], label = 'Threshold')
    axs[0].set_ylabel(r'$I_{nr}$ [$\mu$A]')
    axs[0].set_title('Total current in NR loop')
    axs[0].legend()

    #lower panel, scaled contributions from each synapse    
    max1 = max(neuron_instance.synapses[0].coupling_factor*neuron_instance.synapses[0].I_si)
    max2 = max(neuron_instance.synapses[1].coupling_factor*neuron_instance.synapses[1].I_si)
    if max2 > 0:
        tn = max1/max2
    else:
        tn = 1
    axs[1].plot(time_vec*1e6,neuron_instance.synapses[0].coupling_factor*neuron_instance.synapses[0].I_si*1e6, '-', color = colors['green_3'], linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = neuron_instance.synapses[0].unique_label+' ($tau_{si}$ = '+'{:3.0f}ns)'.format(neuron_instance.synapses[0].integration_loop_time_constant*1e9))
    axs[1].plot(time_vec*1e6,tn*neuron_instance.synapses[1].coupling_factor*neuron_instance.synapses[1].I_si*1e6, '-', color = colors['yellow_3'], linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = neuron_instance.synapses[1].unique_label+' ($tau_{ref}$ = '+'{:4.0f}ns)'.format(neuron_instance.synapses[1].integration_loop_time_constant*1e9))
    axs[1].set_xlabel(r'Time [$\mu$s]')
    axs[1].set_ylabel(r'Contribution to $I_{nr}$ [$\mu$A]')
    axs[1].set_title('Contribution from each synapse')
    axs[1].legend()
    # axs[1].grid(b = False)        
    # axs[1].grid(b = True, which='major', axis='both')
    
    plt.show()       
    fig.savefig('figures/'+save_str+'.png')
    
    return

def plot_burst_size_vs_num_active_synapses(neuron_instance,plot_save_string = ''):
    
    tt = time.time()        
    save_str = 'burst_vs_num_active__'+plot_save_string+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('Number of spikes in output burst')
    plt.title(plot_save_string)

    color_list = ['blue_3','red_3','green_3','yellow_3']
    for ii in range(len(neuron_instance.I_sy_vec)):
        ax.plot(np.arange(1,neuron_instance.num_synapses_tot+1,1),neuron_instance.num_spikes_out_mat[ii,:], 'o-', color = colors[color_list[ii]], linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'tau_ref = {:2.2f}ns, tau_si = {:2.2f}ns, I_sy = {:2.2f}uA'.format(neuron_instance.tau_ref*1e9,neuron_instance.tau_si*1e9,neuron_instance.I_sy_vec[ii]*1e6))        
    ax.set_xlabel(r'Number of active pixels')
    ax.set_ylabel(r'Number of spikes in output burst')
    ax.legend()    
    
    plt.show()
    fig.savefig('figures/'+save_str+'.png')  

    return

def plot_dendritic_drive(time_vec, input_signal__dd):
        
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('Dendritic drive signal')
    # plt.title(plot_save_string)
    
    ax.plot(time_vec*1e6,input_signal__dd*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'])        
    ax.set_xlabel(r'Time [$\mu$s]')
    ax.set_ylabel(r'Dendritic drive [$\mu$A]')    
    
    plt.show() 

    return

def plot_dendritic_integration_loop_current(dendrite_instance):
        
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('Current in dendritic integration loop')
    # plt.title(plot_save_string)
    
    ax.plot(dendrite_instance.time_vec*1e6,dendrite_instance.I_di*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'])        
    ax.set_xlabel(r'Time [$\mu$s]')
    ax.set_ylabel(r'$I_{di}$ [$\mu$A]')    
    
    plt.show() 

    return

def plot_wr_data(data_dict,data_to_plot,plot_save_string):
    
    tt = time.time()  
    if plot_save_string != False:
        save_str = 'wr__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('WRSpice data')
    if 'file_name' in data_dict.keys():
        plt.title(data_dict['file_name'])

    for ii in range(len(data_to_plot)):
        ax.plot(data_dict['time']*1e9,data_dict[data_to_plot[ii]]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = data_to_plot[ii])        
    ax.set_xlabel(r'Time [ns]')
    ax.set_ylabel(r'Current [$\mu$A]')
    ax.legend()    
    
    plt.show()
    if plot_save_string != False:
        fig.savefig('figures/'+save_str+'.png') 
        
    return

def plot_wr_data__currents_and_voltages(data_dict,data_to_plot,plot_save_string):
    
    tt = time.time()  
    if plot_save_string != False:
        save_str = 'wr__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('WRSpice data')
    if 'file_name' in data_dict.keys():
        plt.title(data_dict['file_name'])

    for ii in range(len(data_to_plot)):
        if data_to_plot[ii][0] == 'v':
            ax[0].plot(data_dict['time']*1e9,data_dict[data_to_plot[ii]]*1e3, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = data_to_plot[ii]) 
        if data_to_plot[ii][0] == 'L' or data_to_plot[ii][0] == '@':
            ax[1].plot(data_dict['time']*1e9,data_dict[data_to_plot[ii]]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = data_to_plot[ii]) 
        
    ax[0].set_ylabel(r'Voltage [mV]')
    ax[0].legend()
    ax[0].set_title('Voltages in the circuit')
        
    ax[1].set_xlabel(r'Time [ns]')
    ax[1].set_ylabel(r'Current [$\mu$A]')
    ax[1].legend()
    ax[1].set_title('Currents in the circuit')
    
    plt.show()
    if plot_save_string != False:
        fig.savefig('figures/'+save_str+'.png') 
        
    return

def plot_wr_comparison(target_data,actual_data,main_title,sub_title,y_axis_label):
    
    tt = time.time()    
    save_str = sub_title+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle(main_title)
    plt.title(sub_title)
    
    ax.plot(actual_data[0,:]*1e6,actual_data[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'soen_sim')   
    ax.plot(target_data[0,:]*1e6,target_data[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'WRSpice')             
    ax.set_xlabel(r'Time [$\mu$s]')
    ax.set_ylabel(r'{}'.format(y_axis_label))
    ax.legend()    
    
    plt.show()
    fig.savefig('figures/'+save_str+'.png') 
    
    return

def plot_wr_comparison__dend_drive_and_response(main_title,target_data__drive,actual_data__drive,target_data,actual_data,wr_data_file_name,error__drive,error__signal):
    
    tt = time.time()    
    save_str = 'soen_sim_wr_cmpr__dend__'+wr_data_file_name+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle(main_title)
    
    tf_ind = (np.abs(np.asarray(target_data__drive[0,:])-actual_data__drive[0,-1])).argmin()
    
    axs[0].plot(actual_data__drive[0,:]*1e6,actual_data__drive[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'soen_sim')   
    axs[0].plot(target_data__drive[0,0:tf_ind]*1e6,target_data__drive[1,0:tf_ind]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'WRSpice')             
    axs[0].set_xlabel(r'Time [$\mu$s]')
    axs[0].set_ylabel(r'$I_{flux}$ [$\mu$A]')
    axs[0].legend()
    axs[0].set_title('Drive signal input to DR loop (error = {:1.5f}%)'.format(error__drive*100))
     
    axs[1].plot(actual_data[0,:]*1e6,actual_data[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'soen_sim')   
    axs[1].plot(target_data[0,0:tf_ind]*1e6,target_data[1,0:tf_ind]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'WRSpice')             
    axs[1].set_xlabel(r'Time [$\mu$s]')
    axs[1].set_ylabel(r'$I_{di}$ [$\mu$A]')
    axs[1].legend()
    axs[1].set_title('Output signal in the DI loop (error = {:1.5f}%)'.format(error__signal*100))
    
    plt.show()
    fig.savefig('figures/'+save_str+'.png') 

    return


def plot_wr_comparison__synapse(main_title,spike_times,wr_drive,target_data,actual_data,wr_data_file_name,error__si):
    
    tt = time.time()    
    save_str = 'soen_sim_wr_cmpr__sy__'+wr_data_file_name+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle(main_title)
        
    axs[0].plot(wr_drive[0,:]*1e6,wr_drive[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'wr_drive') 
    tn1 = np.min(wr_drive[1,:])
    tn2 = np.max(wr_drive[1,:])
    for ii in range(len(spike_times)):
        if ii == 0:
            axs[0].plot([spike_times[ii]*1e6,spike_times[ii]*1e6],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'], label = 'Spike times')             
        else:
            axs[0].plot([spike_times[ii]*1e6,spike_times[ii]*1e6],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'])             
    axs[0].set_xlabel(r'Time [$\mu$s]')
    axs[0].set_ylabel(r'$I_{drive}$ [$\mu$A]')
    axs[0].legend()
    axs[0].set_title('Drive signal input SPD to $J_{sf}$')
     
    axs[1].plot(actual_data[0,:]*1e6,actual_data[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'soen_sim')   
    axs[1].plot(target_data[0,:]*1e6,target_data[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'WRSpice')             
    axs[1].set_xlabel(r'Time [$\mu$s]')
    axs[1].set_ylabel(r'$I_{si}$ [$\mu$A]')
    axs[1].legend()
    axs[1].set_title('Output signal in the SI loop (error = {:1.5e})'.format(error__si))
    
    plt.show()
    fig.savefig('figures/'+save_str+'.png') 

    return


def plot_wr_comparison__synapse__tiles(target_data_array,actual_data_array,spike_times,error_array):
    
    tt = time.time()    
   
    # 
        
    # axs[0].plot(wr_drive[0,:]*1e6,wr_drive[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'wr_drive') 
    # tn1 = np.min(wr_drive[1,:])
    # tn2 = np.max(wr_drive[1,:])
    # color_list_actual = ['blue3','red3','green3','yellow3']
    # color_list_target = ['blue2','red2','green2','yellow2']
    title_strings = ['Vary $I_{sy}$','Vary $L_{si}$','Vary $tau_{si}$']
    save_strings = ['vary_Isy','vary_Lsi','vary_tausi']
    legend_strings = [ ['$I_{sy} = 23\mu A$','$I_{sy} = 28\mu A$','$I_{sy} = 33\mu A$','$I_{sy} = 38\mu A$'],
                       ['$L_{si} = 7.75nH$','$L_{si} = 77.5nH$','$L_{si} = 775nH$','$L_{si} = 7.75\mu H$'],
                       ['$tau_{si} = 10ns$','$tau_{si} = 50ns$','$tau_{si} = 250ns$','$tau_{si} = 1.25\mu s$',] ]
    i1 = [0,1,0,1]
    i2 = [0,0,1,1]
    for ii in range(3):
        
        save_str = 'soen_sim_wr_cmpr__sy__3tiles__'+save_strings[ii]+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))    
        fig, axs = plt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = False)
        fig.suptitle(title_strings[ii])
        # print(size(axs))     
        
        for jj in range(4):
            
            axs[i1[jj],i2[jj]].plot(target_data_array[ii*4+jj][0,:]*1e6,target_data_array[ii*4+jj][1,:]*1e6, '-', color = colors['yellow3'], linewidth = pp['bold_linewidth'], label = 'WRSpice')
            axs[i1[jj],i2[jj]].plot(actual_data_array[ii*4+jj][0,:]*1e6,actual_data_array[ii*4+jj][1,:]*1e6, '-', color = colors['blue3'], linewidth = pp['nominal_linewidth'], label = 'soen_sim')   
        
    
        # for kk in range(len(spike_times)):
        #     if ii == 0:
        #         axs[ii].plot([spike_times[kk]*1e6,spike_times[kk]*1e6],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'], label = 'Spike times')             
        #     else:
        #         axs[ii].plot([spike_times[kk]*1e6,spike_times[kk]*1e6],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'])  
                
            if jj == 3:
                axs[i1[jj],i2[jj]].legend() 
            axs[i1[jj],i2[jj]].set_xlabel(r'Time [$\mu$s]')
            axs[i1[jj],i2[jj]].set_ylabel(r'$I_{si}$ [$\mu$A]')
            axs[i1[jj],i2[jj]].set_title(legend_strings[ii][jj]+'; error = {:7.5e}'.format(error_array[ii*4+jj]))
    
        plt.show()
        fig.savefig('figures/'+save_str+'.png') 

    return

def plot_error_mat(error_mat,vec1,vec2,x_label,y_label,title_string,plot_string):
    
    tt = time.time()    
    save_str = 'wr_err__'+plot_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    # fig, ax = plt.subplots(1,1)
    # error = ax.imshow(np.transpose(error_mat[:,:]), cmap = plt.cm.viridis, interpolation='none', extent=[vec1[0],vec1[-1],vec2[0],vec2[-1]], aspect = 'auto', origin = 'lower')
    # cbar = fig.colorbar(error, extend='both')
    # cbar.minorticks_on()
    # fig.suptitle('Error versus {} and {}, {}'.format(x_label,y_label,extra_title_str))
    # plt.title(title_string)
    # ax.set_xlabel(r'{}'.format(x_label))
    # ax.set_ylabel(r'{}'.format(y_label)) 
    # plt.show()       
    # fig.savefig('figures/'+save_str+'__lin.png')  
     
    fig, ax = plt.subplots(1,1)
    error = ax.imshow(np.log10(np.transpose(error_mat[:,:])), cmap = plt.cm.viridis, interpolation='none', extent=[vec1[0],vec1[-1],vec2[0],vec2[-1]], aspect = 'auto', origin = 'lower')
    cbar = fig.colorbar(error, extend='both')
    cbar.minorticks_on()     
    fig.suptitle('log10(Error) versus {} and {}'.format(x_label,y_label))
    plt.title(title_string)
    ax.set_xlabel(r'{}'.format(x_label))
    ax.set_ylabel(r'{}'.format(y_label))   
    plt.show()      
    fig.savefig('figures/'+save_str+'__log.png') 
        
    return


def plot_fq_peaks__three_jjs(time_vec,t_lims,j_sf,j_sf_peaks,j_jtl,j_jtl_peaks,j_si,j_si_peaks,I_si,file_name):
    
    
    # fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)
    # fig.suptitle(file_name)   
    # ax[0].plot(time_vec*1e9,j_sf*1e3, '-', label = '$J_{sf}$')             
    # ax[0].plot(time_vec[j_sf_peaks]*1e9,j_sf[j_sf_peaks]*1e3, 'x')
    # ax[0].set_xlabel(r'Time [ns]')
    # ax[0].set_ylabel(r'Voltage [mV]')
    # ax[0].legend()
    # ax[1].plot(time_vec*1e9,j_jtl*1e3, '-', label = '$J_{jtl}$')             
    # ax[1].plot(time_vec[j_jtl_peaks]*1e9,j_jtl[j_jtl_peaks]*1e3, 'x')
    # ax[1].set_xlabel(r'Time [ns]')
    # ax[1].set_ylabel(r'Voltage [mV]')
    # ax[1].legend()
    # ax[2].plot(time_vec*1e9,j_si*1e3, '-', label = '$J_{si}$')             
    # ax[2].plot(time_vec[j_si_peaks]*1e9,j_si[j_si_peaks]*1e3, 'x')
    # ax[2].set_xlabel(r'Time [ns]')
    # ax[2].set_ylabel(r'Voltage [mV]')
    # ax[2].legend()
    # ax[2].set_xlim(t_lims)
    # plt.show()
    
    
    # fig = plt.figure()
    # fig.suptitle(file_name)   
    # ax = fig.gca()
    
    # ax.plot(time_vec*1e9,j_sf*1e3, '-', color = colors['blue3'], label = '$J_{sf}$')             
    # ax.plot(time_vec[j_sf_peaks]*1e9,j_sf[j_sf_peaks]*1e3, 'x', color = colors['blue5'])
    
    # ax.plot(time_vec*1e9,j_jtl*1e3, '-', color = colors['green3'], label = '$J_{jtl}$')             
    # ax.plot(time_vec[j_jtl_peaks]*1e9,j_jtl[j_jtl_peaks]*1e3, 'x', color = colors['green5'])
    
    # ax.plot(time_vec*1e9,j_si*1e3, '-', color = colors['yellow3'], label = '$J_{si}$')             
    # ax.plot(time_vec[j_si_peaks]*1e9,j_si[j_si_peaks]*1e3, 'x', color = colors['yellow5'])
    
    # ax.set_xlabel(r'Time [ns]')
    # ax.set_ylabel(r'Voltage [mV]')
    # ax.legend()
    
    # ax.set_xlim(t_lims)
    # plt.show()
 
    
     
    fig = fig, ax = plt.subplots(nrows = 4, ncols = 1, sharex = False, sharey = False)
    fig.suptitle(file_name)   
    
    ax[0].plot(time_vec*1e9,I_si*1e6, '-', linewidth = pp['nominal_linewidth'], color = colors['blue3']) 
    ax[0].set_ylabel(r'$I_{si}$ [$\mu$A]')
    for ii in range(3):
        ax[ii+1].plot(time_vec*1e9,j_sf*1e3, '-', linewidth = pp['nominal_linewidth'], color = colors['yellow3'], label = '$J_{sf}$')             
        # ax[ii].plot(time_vec[j_sf_peaks]*1e9,j_sf[j_sf_peaks]*1e3, 'x', color = colors['blue5'])
        ax[ii+1].plot(time_vec*1e9,j_jtl*1e3, '-', linewidth = pp['nominal_linewidth'], color = colors['green3'], label = '$J_{jtl}$')             
        # ax[ii].plot(time_vec[j_jtl_peaks]*1e9,j_jtl[j_jtl_peaks]*1e3, 'x', color = colors['green5'])
        ax[ii+1].plot(time_vec*1e9,j_si*1e3, '-', linewidth = pp['nominal_linewidth'], color = colors['blue3'], label = '$J_{si}$')             
        # ax[ii].plot(time_vec[j_si_peaks]*1e9,j_si[j_si_peaks]*1e3, 'x', color = colors['yellow5'])           
        ax[ii+1].set_xlim(t_lims[ii])
        if ii == 1:
            ax[ii+1].set_ylabel(r'Voltage [mV]')
        if ii == 2:
            ax[ii+1].set_xlabel(r'Time [ns]')
            ax[ii+1].legend()
    
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout(pad = 0, w_pad = 0, h_pad = 0)
    plt.show()
        
    return


def plot_fq_rate_and_delay__three_jjs(time_vec,I_si,j_si_ifi,j_si_rate,j_si_peaks,j_sf_peaks,j_jtl_peaks,dt_sf_jtl,dt_jtl_si,dt_si_sf,file_name):
    
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)
    # fig.suptitle(file_name) 
    
    ax[0].plot(time_vec[j_si_peaks[0:-1]]*1e9,dt_sf_jtl*1e12, '-', color = colors['blue3'], label = '$J_{jtl}-J_{sf}$')     
    ax[0].plot(time_vec[j_si_peaks[0:-1]]*1e9,dt_jtl_si*1e12, '-', color = colors['green3'], label = '$J_{si}-J_{jtl}$')     
    ax[0].plot(time_vec[j_si_peaks[0:-1]]*1e9,dt_si_sf*1e12, '-', color = colors['yellow3'], label = '$J_{sf}-J_{si}$') 
    
    ax[0].set_xlabel(r'Time [ns]')
    ax[0].set_ylabel(r'Fluxon generation delay [ps]')
    ax[0].legend()
    
    ax[1].plot(I_si[j_si_peaks[0:-1]]*1e6,j_si_ifi*1e12, '-', color = colors['green3'])  
    ax[1].set_xlabel(r'$I_{si}$ [$\mu$A]')
    ax[1].set_ylabel(r'IFI [ps]', color = colors['green3'])
    ax[1].tick_params(axis = 'y', labelcolor = colors['green3'])
    
    ax2 = ax[1].twinx()
    ax2.plot(I_si[j_si_peaks[0:-1]]*1e6,j_si_rate*1e-9, '-', color = colors['yellow3']) 
    ax2.set_ylabel(r'Rate [kFQ per $\mu$s]', color = colors['yellow3'])
    ax2.tick_params(axis = 'y', labelcolor = colors['yellow3'])    
    
    plt.show()    
    
    return


def plot_fq_peaks(data_x,data_y,peak_indices):
    
    fig, ax = plt.subplots(1,1)
    ax.plot(data_x[:]*1e9,data_y[:], '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'data')   
    ax.plot(data_x[peak_indices]*1e9,data_y[peak_indices], 'x', markersize = pp['big_markersize'], label = 'peaks') 
    ax.set_xlabel(r'Time [ns]')
    ax.set_ylabel(r'Amplitude')
    plt.show()
    
    return

def plot_fq_peaks__dt_vs_bias(bias_current,dt_fq_peaks,Ic):
    
    fig, ax = plt.subplots(1,1)
    ax.plot(bias_current[:]*1e6,dt_fq_peaks[:]*1e9, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize']) #, label = 'data'  
    ax.plot([Ic*1e6,Ic*1e6],[np.min(dt_fq_peaks)*1e9,np.max(dt_fq_peaks)*1e9], '-.', linewidth = pp['fine_linewidth'])  
    ax.set_xlabel(r'Current bias [$\mu$A]')
    ax.set_ylabel(r'Time between flux quanta [ns]')
    plt.show()
    
    return


def plot_syn_rate_array(I_si_array__scaled,master_rate_array,I_drive_list):
    
    # I_sy = 40
    #fig, ax = plt
    # fig.suptitle('master _ sy _ rates') 
    # plt.title('$Isy =$ {} $\mu$A'.format(I_sy))
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    num_drives = len(I_drive_list)
    for ii in range(num_drives):
        ax.plot(I_si_array__scaled[num_drives-ii-1][:],master_rate_array[num_drives-ii-1][:]*1e-3, '-', linewidth = pp['nominal_linewidth'], label = 'I_drive = {}'.format(I_drive_list[num_drives-ii-1]))    
    ax.set_xlabel(r'$I_{si}$ [$\mu$A]')
    ax.set_ylabel(r'$r_{j_{si}}$ [kilofluxons per $\mu$s]')
    # ax.legend()
    plt.show()
    
    return


def plot_syn_rate_array__waterfall(I_si_array__scaled,master_rate_array,I_drive_list):

    color_list = [colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
                  colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
                  colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
                  colors['red4'],colors['red3'],colors['red2'],colors['red1'],
                  colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
                  colors['green4'],colors['green3'],colors['green2'],colors['green1'],
                  colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
                  colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
                  colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
                  colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
                  colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
                  colors['red4'],colors['red3'],colors['red2'],colors['red1'],
                  colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
                  colors['green4'],colors['green3'],colors['green2'],colors['green1'],
                  colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
                  colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
                  colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
                  colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
                  colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
                  colors['red4'],colors['red3'],colors['red2'],colors['red1'],
                  colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
                  colors['green4'],colors['green3'],colors['green2'],colors['green1'],
                  colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
                  colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
                  colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
                  colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
                  colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
                  colors['red4'],colors['red3'],colors['red2'],colors['red1'],
                  colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
                  colors['green4'],colors['green3'],colors['green2'],colors['green1'],
                  colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
                  colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
                  colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
                  colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
                  colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
                  colors['red4'],colors['red3'],colors['red2'],colors['red1'],
                  colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
                  colors['green4'],colors['green3'],colors['green2'],colors['green1'],
                  colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
                  colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
                  colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
                  colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
                  colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
                  colors['red4'],colors['red3'],colors['red2'],colors['red1'],
                  colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
                  colors['green4'],colors['green3'],colors['green2'],colors['green1'],
                  colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
                  colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
                  ]
    color_ind = np.arange(0,len(color_list),1)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    verts = []
    ys = np.asarray(I_drive_list)
    for ii in range(len(ys)):
        xs = I_si_array__scaled[ii][:]
        zs = [xx*1e-3 for xx in master_rate_array[ii][:]]
        verts.append(list(zip(xs,zs)))

    poly = PolyCollection(verts, edgecolors = color_list, facecolors = [color_list[aa] for aa in color_ind])#, facecolors = [color_list[aa] for aa in range(len(color_list))]
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs = ys, zdir = 'z')
    
    ax.set_zlabel('Idrive')
    ax.set_zlim3d(0, 20)
    ax.set_xlabel('Isi')
    ax.set_xlim3d(0,20)
    ax.set_ylabel('rate')
    ax.set_ylim3d(0, 50)
    
    plt.show()
    
    
    # cmap = pl.cm.get_cmap('viridis')
    # verts = []
    # zs = alpha_list
    # for i, z in enumerate(zs):
    #     ys = B_l2[:, i]
    #     verts.append(list(zip(x, ys)))
    
    # ax = pl.gcf().gca(projection='3d')
    
    # poly = PolyCollection(verts, facecolors=[cmap(a) for a in alpha_list])
    # poly.set_alpha(0.7)
    # ax.add_collection3d(poly, zs=zs, zdir='y')
    

    # import matplotlib.pyplot as plt
    # from matplotlib import colors as mcolors
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    
    
    # def cc(arg):
    #     return mcolors.to_rgba(arg, alpha=0.6)
    
    # xs = np.arange(0, 10, 0.4)
    # verts = []
    # zs = [0.0, 1.0, 2.0, 3.0]
    # for z in zs:
    #     ys = np.random.rand(len(xs))
    #     ys[0], ys[-1] = 0, 0
    #     verts.append(list(zip(xs, ys)))
    
    # poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),
    #                                          cc('y')])
    # poly.set_alpha(0.7)
    # ax.add_collection3d(poly, zs=zs, zdir='y')
    
    # ax.set_xlabel('X')
    # ax.set_xlim3d(0, 10)
    # ax.set_ylabel('Y')
    # ax.set_ylim3d(-1, 4)
    # ax.set_zlabel('Z')
    # ax.set_zlim3d(0, 1)
    
    # plt.show()

    return


def plot__syn__error_vs_dt(dt_vec,error_array):
    
    num_cases = 3
    title_list = ['Vary $I_{sy}$','Vary $L_{si}$','Vary $tau_{si}$']
    legend_list = [['$I_{sy}$ = 23uA','$I_{sy}$ = 28uA','$I_{sy}$ = 33uA','$I_{sy}$ = 38uA'],
                   ['$L_{si}$ = 7.75nH','$L_{si}$ = 77.5nH','$L_{si}$ = 775nH','$L_{si}$ = 7.75$\mu$H'],
                   ['$tau_{si}$ = 10ns','$tau_{si}$ = 50ns','$tau_{si}$ = 250ns','$tau_{si}$ = 1.25$\mu$s']]
    fig, ax = plt.subplots(nrows = num_cases, ncols = 1)
    for ii in range(num_cases):
        for jj in range(4):
            ax[ii].loglog(dt_vec*1e9,error_array[ii*4+jj,:], '-o', markersize = pp['nominal_markersize'], label = legend_list[ii][jj] )    
        ax[ii].set_xlabel(r'dt [ns]')
        ax[ii].set_ylabel(r'$Chi^2$ error')
        ax[ii].set_title(title_list[ii])
        ax[ii].legend()
    # grid(True,which='both')
    plt.show()
    
    return


def plot_spd_response(time_vec,time_vec_reduced,I_sy_list,I_spd_array,I_spd_array_reduced):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
    # fig.suptitle('spd response') 
    plot_list = [23,28,33,38]
    num_plot = len(plot_list)
    for ii in range(num_plot):  
        ind = (np.abs(I_sy_list[:]-plot_list[ii])).argmin()
        ax.plot([xx*1e3 for xx in time_vec],I_spd_array[ind])    
        ax.plot([xx*1e3 for xx in time_vec_reduced],I_spd_array_reduced[ind], label = 'I_sy = {}uA'.format(I_sy_list[ind]))    
    ax.set_xlabel(r'Time [ns]')
    ax.set_ylabel(r'$I_{spd}$ [$\mu$A]')
    ax.set_xlim([0,150])
    ax.legend()
    plt.show()

    return
