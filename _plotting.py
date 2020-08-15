import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib as mp
import matplotlib.gridspec as gridspec
import pandas as pd
import pickle

from util import color_dictionary, physical_constants
colors = color_dictionary()

# from _functions import syn_1jj_rate_fit

def plot_params():
    
    plot_type = '17.2__for_pubs' # 'large' # 'two_rows' # 'three_rows' # 'four_rows' # 'large' # 'single_frame' # 'four_tiles' #
    
    pp = dict()
        
    if plot_type == '17.2__for_pubs':
        
        pp['title_font_size'] = 8
        pp['subtitle_font_size'] = 8
        pp['axes_labels_font_size'] = 8
        pp['axes_labels_pad'] = 0 # 4
        pp['tick_labels_font_size'] = 8
        pp['legend_font_size'] = 8
        pp['nominal_linewidth'] = 1
        pp['fine_linewidth'] = 0.5
        pp['bold_linewidth'] = 1.5
        pp['small_markersize'] = 3
        pp['nominal_markersize'] = 4
        pp['big_markersize'] = 5
        tn = 6.9 # 4*8.6/2.54
        pp['fig_size'] = (tn,tn/1.618)
        # pp['fig_size'] = (tn,tn/1.2)
        pp['axes_linewidth'] = 1
        
        pp['major_tick_width'] = 0.75
        pp['major_tick_length'] = 2
        pp['minor_tick_width'] = 0.5
        pp['minor_tick_length'] = 1
        
        pp['xmargin'] = 0 # 0.05 # space between traces and axes
        pp['ymargin'] = 0.05 # 0.05
        
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
        
        pp['title_font_size'] = 10
        pp['subtitle_font_size'] = 10
        pp['axes_labels_font_size'] = 14
        pp['axes_labels_pad'] = 0 # 4
        pp['tick_labels_font_size'] = 14
        pp['legend_font_size'] = 10
        pp['nominal_linewidth'] = 2
        pp['fine_linewidth'] = 1.5
        pp['bold_linewidth'] = 3
        pp['small_markersize'] = 3
        pp['nominal_markersize'] = 4
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
        pp['axes_labels_font_size'] = 10
        pp['axes_labels_pad'] = 0 # 4
        pp['tick_labels_font_size'] = 10
        pp['legend_font_size'] = 10
        pp['nominal_linewidth'] = 0.75
        pp['fine_linewidth'] = 0.5
        pp['bold_linewidth'] = 2
        pp['nominal_markersize'] = 2
        pp['big_markersize'] = 3
        tn = 1.075*8.6/2.54
        pp['fig_size'] = (tn,tn/1.618)
        # pp['fig_size'] = (tn,tn/1.2)
        pp['axes_linewidth'] = 0.75
        
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

# plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue3'],colors['red3'],colors['green3'],colors['yellow3']])
# plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue1'],colors['blue3'],colors['red1'],colors['red3'],colors['green1'],colors['green3'],colors['yellow1'],colors['yellow3']])

plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
                                                    colors['red5'],colors['red4'],colors['red3'],colors['red2'],colors['red1'],
                                                    colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green3'],
                                                    colors['yellow5'],colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1']])

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

def plot_synaptic_integration_loop_current(synapse_instance):
    
    input_spike_times = synapse_instance.input_spike_times
    time_vec = synapse_instance.time_vec

    if synapse_instance.integration_loop_time_constant < 1e-3:
        
        main_title = 'Synapse: {} ({})\nI_sy = {:7.4f} uA; tau_si = {:7.4f} ns; L_si = {:7.4f} nH'.format(synapse_instance.name,
                                                                                                   synapse_instance.unique_label,
                                                                                                   synapse_instance.I_sy*1e6,
                                                                                                   synapse_instance.integration_loop_time_constant*1e9,
                                                                                                   synapse_instance.integration_loop_total_inductance*1e9)
    else:
        
                
        main_title = 'Synapse: {} ({})\nI_sy = {:7.4f} uA; tau_si = inf; L_si = {:7.4f} nH'.format(synapse_instance.name,
                                                                                                   synapse_instance.unique_label,
                                                                                                   synapse_instance.I_sy*1e6,
                                                                                                   synapse_instance.integration_loop_total_inductance*1e9)

    fig = plt.figure()
    fig.suptitle(main_title)    
    ax = fig.gca()
    
    ax.plot(time_vec*1e6,synapse_instance.I_si*1e6, '-', linewidth = pp['nominal_linewidth'], color = colors['blue3'], label = 'synaptic response')
    ylim = ax.get_ylim()
    for ii in range(len(input_spike_times)):
        if ii == 0:
            ax.plot([input_spike_times[ii]*1e6, input_spike_times[ii]*1e6], [ylim[0], ylim[1]], ':', color = colors['bluegrey1'], linewidth = pp['fine_linewidth'], label = 'spike times')
        else:
            ax.plot([input_spike_times[ii]*1e6, input_spike_times[ii]*1e6], [ylim[0], ylim[1]], ':', color = colors['bluegrey1'], linewidth = pp['fine_linewidth'])
    ax.set_xlabel(r'Time [us]')
    ax.set_ylabel(r'Isi [uA]')

    ax.legend()
    # grid(True,which='both')
    
    plt.show()
    
    return


def plot_synaptic_integration_loop_current__multiple_synapses(synapse_list):


    fig = plt.figure()
    # fig.suptitle('Synapse saturation vs Isy; tau_si = inf; L_si = {:7.4f} nH'.format(synapse_list[0].integration_loop_total_inductance*1e9))    
    ax = fig.gca()
    
    for jj in range(len(synapse_list)):  
        
        synapse_instance = synapse_list[jj]
        time_vec = synapse_instance.time_vec
        
        legend_text = 'I_sy = {:7.4f} uA'.format(synapse_instance.I_sy*1e6)                                                                                            
            
        ax.plot(time_vec*1e6,synapse_instance.I_si*1e6, '-', linewidth = pp['nominal_linewidth'], label = legend_text)
        
        if jj == len(synapse_list)-1:
            
            ylim = ax.get_ylim()            
            input_spike_times = synapse_instance.input_spike_times
            for ii in range(len(input_spike_times)):
                if ii == 0:
                    ax.plot([input_spike_times[ii]*1e6, input_spike_times[ii]*1e6], [ylim[0], ylim[1]], ':', color = colors['bluegrey1'], linewidth = pp['fine_linewidth'], label = 'spike times')
                else:
                    ax.plot([input_spike_times[ii]*1e6, input_spike_times[ii]*1e6], [ylim[0], ylim[1]], ':', color = colors['bluegrey1'], linewidth = pp['fine_linewidth'])
                    
        ax.set_xlabel(r'Time [$\mu$s]')
        ax.set_ylabel(r'$I_{si}$ [$\mu$A]')
    
        # ax.legend()
    
    plt.show()
    
    return


def plot_Isisat_vs_Isy(synapse_list):
    
    I_sy_vec = np.zeros([len(synapse_list)])
    I_si_sat_vec = np.zeros([len(synapse_list)])
    for ii in range(len(synapse_list)):
        I_sy_vec[ii] = 1e6*synapse_list[ii].I_sy
        I_si_sat_vec[ii] = 1e6*synapse_list[ii].I_si[-1]
    
    fig = plt.figure()
    # fig.suptitle('Synapse saturation vs Isy; tau_si = inf; L_si = {:7.4f} nH'.format(synapse_list[0].integration_loop_total_inductance*1e9)) 
    ax = fig.gca()
    
    ax.plot(I_sy_vec,I_si_sat_vec, '-', linewidth = pp['nominal_linewidth'], color = colors['blue3'])
    ax.set_xlabel(r'$I_{sy}$ [$\mu$A]')
    ax.set_ylabel(r'$I_{si}^{sat}$ [$\mu$A]')
    
    plt.show()
    
    return


def plot_Isi_vs_Isy(synapse_list):
    
    I_sy_vec = np.zeros([len(synapse_list)])
    I_si_sat_vec = np.zeros([len(synapse_list)])
    for ii in range(len(synapse_list)):
        I_sy_vec[ii] = 1e6*synapse_list[ii].I_sy
        I_si_sat_vec[ii] = 1e6*synapse_list[ii].I_si[-1]
    
    fig = plt.figure()
    # fig.suptitle('Isi vs Isy; tau_si = inf; L_si = {:7.4f} nH'.format(synapse_list[0].integration_loop_total_inductance*1e9)) 
    ax = fig.gca()
    
    ax.plot(I_sy_vec,I_si_sat_vec, '-', linewidth = pp['nominal_linewidth'], color = colors['blue3'])
    ax.set_xlabel(r'$I_{sy}$ [$\mu$A]')
    ax.set_ylabel(r'$I_{si}$ [$\mu$A]')
    
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


def plot_neuronal_response(neuron_instance):
    
    # tt = time.time()   
    # save_str = 'bursting__'+plot_save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    # p = physical_constants()
    Phi_0 = 1 # 1e18*p['Phi0']    
    
    time_vec = neuron_instance.time_vec
    fig, axs = plt.subplots(nrows = 6, ncols = 1, sharex = True, sharey = False)   
    # fig.suptitle('Current in the neuronal receiving loop versus time')
    
    # input synapses     
    for name_1 in neuron_instance.input_synaptic_connections:
        
        axs[0].plot(time_vec*1e6,neuron_instance.synapses[name_1].I_spd_vec, 
                    '-', color = colors['green3'], markersize = pp['nominal_markersize'], label = name_1+' ($I_{sy}$ = '+'{:5.2f}$\mu$A)'.format(neuron_instance.synapses[name_1].I_sy))         
        
        axs[1].plot(time_vec*1e6,neuron_instance.synapses[name_1].I_si_vec, '-', color = colors['green3'], markersize = pp['nominal_markersize'], 
                    label = name_1+' ($L_{si}$ = '+'{:4.0f}nH'.format(neuron_instance.synapses[name_1].integration_loop_total_inductance*1e9)+
                            '; $tau_{si}$ = '+'{:3.0f}ns)'.format(neuron_instance.synapses[name_1].integration_loop_time_constant*1e9))
        
        for _time in neuron_instance.synapses[name_1].input_spike_times:
            _time_ind = (np.abs(neuron_instance.time_vec-_time)).argmin()
            axs[0].plot(time_vec[_time_ind]*1e6,neuron_instance.synapses[name_1].I_spd_vec[_time_ind], 'o', color = colors['green5'], markersize = pp['nominal_markersize'])  
            axs[1].plot(time_vec[_time_ind]*1e6,neuron_instance.synapses[name_1].I_si_vec[_time_ind], 'o', color = colors['green5'], markersize = pp['nominal_markersize'])  
            axs[2].plot(time_vec[_time_ind]*1e6,Phi_0*neuron_instance.influx_vec[_time_ind], 'o', color = colors['blue5'], markersize = pp['nominal_markersize'])
               
         
    axs[1].set_ylabel(r'$I^{si}$ [$\mu$A]')
    # axs[4].set_title('Contribution from each synapse')
    axs[1].legend() 
    
    axs[0].set_ylabel(r'$I^{sy}_{spd}$ [$\mu$A]')
    # axs[5].set_title('Contribution from each synapse')
    axs[0].legend()
    
    # neuron
    axs[2].plot(time_vec*1e6,Phi_0*neuron_instance.influx_vec, '-', color = colors['blue3'], label = neuron_instance.name)
    axs[3].plot(time_vec*1e6,neuron_instance.I_ni_vec, '-', color = colors['blue3'], label = neuron_instance.name+' ($tau_{ni}$ = '+'{:3.0f}ns)'.format(neuron_instance.integration_loop_time_constant*1e9))
    # for _st in range(len(neuron_instance.spike_times)):
        # axs[2].plot([neuron_instance.spike_times[_st]*1e6,neuron_instance.spike_times[_st]*1e6], [np.min(Phi_0*neuron_instance.influx_vec),np.max(Phi_0*neuron_instance.influx_vec)],':', color = colors['greengrey1'])
        # axs[3].plot([neuron_instance.spike_times[_st]*1e6,neuron_instance.spike_times[_st]*1e6], [np.min(neuron_instance.I_ni_vec),np.max(neuron_instance.I_ni_vec)],':', color = colors['greengrey1'])
    axs[3].plot(neuron_instance.spike_times*1e6,neuron_instance.output_voltage[neuron_instance.voltage_peaks], 'x', color = colors['blue5'])
    
    axs[2].set_ylabel(r'$\Phi^{nr}_{a}$ [$\mu$A pH]')
    axs[2].legend()
    
    axs[3].set_ylabel(r'$I^{ni}$ [$\mu$A]')
    axs[3].legend()
    
    # axs[4].plot(time_vec[0:-1]*1e6,neuron_instance.output_voltage*1e9, '-', color = colors['yellow3'], label = '$V^{out}$')
    # axs[4].plot(neuron_instance.spike_times*1e6,neuron_instance.output_voltage[neuron_instance.voltage_peaks]*1e9, 'x', color = colors['yellow5'])
    # axs[4].set_ylabel(r'$V^{out}$ [$nV$]')
    # axs[4].legend()
    
    # refractory dendrite
    name = '{}__r'.format(neuron_instance.name)
    axs[4].plot(time_vec*1e6,Phi_0*neuron_instance.dendrites[name].influx_vec, '-', color = colors['red3'], markersize = pp['nominal_markersize'], label = name)
    axs[4].plot(time_vec[neuron_instance.voltage_peaks]*1e6,Phi_0*neuron_instance.dendrites[name].influx_vec[neuron_instance.voltage_peaks], 'x', color = colors['red5'])
    axs[5].plot(time_vec*1e6,neuron_instance.dendrites[name].I_di_vec, '-', color = colors['red3'], markersize = pp['nominal_markersize'], label = name+' ($tau_{ri}$ = '+'{:3.0f}ns)'.format(neuron_instance.dendrites[name].integration_loop_time_constant*1e9))        
    axs[5].plot(time_vec[neuron_instance.voltage_peaks]*1e6,neuron_instance.dendrites[name].I_di_vec[neuron_instance.voltage_peaks], 'x', color = colors['red5'])
    
    # for name in neuron_instance.input_dendritic_connections:
        # axs[4].plot(time_vec*1e6,Phi_0*neuron_instance.dendrites[name].influx_vec, '-', color = colors['red3'], markersize = pp['nominal_markersize'], label = name)
        # axs[5].plot(time_vec*1e6,neuron_instance.dendrites[name].I_di_vec, '-', color = colors['red3'], markersize = pp['nominal_markersize'], label = name+' ($tau_{ri}$ = '+'{:3.0f}ns)'.format(neuron_instance.dendrites[name].integration_loop_time_constant*1e9))        
    
    axs[4].set_ylabel(r'$\Phi^{dr}_{ref}$ [$\mu$A pH]')
    axs[4].legend()
         
    axs[5].set_ylabel(r'$I^{di}_{ref}$ [$\mu$A]')
    axs[5].set_xlabel(r'Time [$\mu$s]')
    axs[5].legend()     
    
    plt.show()       
    # fig.savefig('figures/'+save_str+'.png')
    
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
    # fig.savefig('figures/'+save_str+'.png') 
    
    return

def plot_wr_comparison__dend_drive_and_response(main_title,target_data__drive,actual_data__drive,target_data,actual_data,wr_data_file_name,error__drive,error__signal):
    
    tt = time.time()    
    # save_str = 'soen_sim_wr_cmpr__dend__'+wr_data_file_name+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
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
    # fig.savefig('figures/'+save_str+'.png') 

    return


def plot_wr_comparison__synapse(main_title,spike_times,target_drive,actual_drive,target_data,actual_data,wr_data_file_name,error_drive,error__si):
    
    tt = time.time()    
    save_str = 'soen_sim_wr_cmpr__sy__'+wr_data_file_name+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)   
    fig.suptitle('file: {}; error_drive = {:7.5e}, error_signal = {:7.5e}'.format(main_title,error_drive,error__si))
        
    axs[0].plot(actual_drive[0,:]*1e6,actual_drive[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['green3'], label = 'soen_drive')
    axs[0].plot(target_drive[0,:]*1e6,target_drive[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['yellow3'], label = 'wr_drive')  
    tn1 = np.min(target_drive[1,:])
    tn2 = np.max(target_drive[1,:])
    for ii in range(len(spike_times)):
        if ii == 0:
            axs[0].plot([spike_times[ii]*1e6,spike_times[ii]*1e6],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'], label = 'Spike times')             
        else:
            axs[0].plot([spike_times[ii]*1e6,spike_times[ii]*1e6],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'])             
    axs[0].set_xlabel(r'Time [$\mu$s]')
    axs[0].set_ylabel(r'$I_{drive}$ [$\mu$A]')
    axs[0].legend()
    # axs[0].set_title('Drive signal input SPD to Jsf (error = {:7.5e})'.format(error_drive))
     
    axs[1].plot(actual_data[0,:]*1e6,actual_data[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['green3'], label = 'soen_sim')   
    axs[1].plot(target_data[0,:]*1e6,target_data[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['yellow3'], label = 'WRSpice')             
    axs[1].set_xlabel(r'Time [$\mu$s]')
    axs[1].set_ylabel(r'$I_{si}$ [$\mu$A]')
    axs[1].legend()
    # axs[1].set_title('Output signal in the SI loop (error = {:7.5e})'.format(error__si))
    
    plt.show()
    # fig.savefig('figures/'+save_str+'.png') 

    return fig


def plot_wr_comparison__synapse__spd_jj_test(main_title,spike_times,target_drive,actual_drive,target_data,actual_data,wr_data_file_name,V_fq,j_peaks,V_avg,time_avg,V_sf):
    
    tt = time.time()    
    save_str = 'soen_sim_wr_cmpr__sy__'+wr_data_file_name+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)   
    # fig.suptitle('file: {}; error_drive = {:7.5e}, error_signal = {:7.5e}'.format(main_title,error_drive,error__si))
        
    axs[0].plot(actual_drive[0,:]*1e9,actual_drive[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['red3'], label = 'soen_drive')
    axs[0].plot(target_drive[0,:]*1e9,target_drive[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['blue3'], label = 'wr_drive')  
    tn1 = np.min(target_drive[1,:])
    tn2 = np.max(target_drive[1,:])
    for ii in range(len(spike_times)):
        if ii == 0:
            axs[0].plot([spike_times[ii]*1e9,spike_times[ii]*1e9],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'], label = 'Spike times')             
        else:
            axs[0].plot([spike_times[ii]*1e9,spike_times[ii]*1e9],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'])             
    axs[0].set_xlabel(r'Time [ns]')
    axs[0].set_ylabel(r'$I_{spd2}$ [$\mu$A]')
    # axs[0].set_ylim([np.min(target_drive[1,:])*1e6,np.max(target_drive[1,:])*1e6])
    axs[0].legend()
    # axs[0].set_title('Drive signal input SPD to Jsf (error = {:7.5e})'.format(error_drive))
     
    axs[1].plot(actual_data[0,:]*1e9,actual_data[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['red3'], label = 'soen_sim')   
    axs[1].plot(target_data[0,:]*1e9,target_data[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['blue3'], label = 'WRSpice')             
    axs[1].set_xlabel(r'Time [ns]')
    axs[1].set_ylabel(r'$I_{sf}$ [$\mu$A]')
    # axs[1].set_ylim([np.min(target_data[1,:])*1e6,np.max(target_data[1,:])*1e6])
    axs[1].legend()
    # axs[1].set_title('Output signal in the SI loop (error = {:7.5e})'.format(error__si))
      
    axs[2].plot(target_drive[0,:]*1e9,V_fq*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['blue3'], label = 'WRSpice')   
    axs[2].plot(target_drive[0,j_peaks[:]]*1e9,V_fq[j_peaks[:]]*1e6, 'x', markersize = pp['big_markersize'], color = colors['green3'], label = 'WR peaks')   
    axs[2].plot(time_avg*1e9,V_avg*1e6, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['small_markersize'], color = colors['yellow3'], label = 'WRSpice')             
    axs[2].plot(actual_drive[0,:]*1e9,V_sf*1e6, '-o', markersize = pp['small_markersize'], linewidth = pp['nominal_linewidth'], color = colors['red3'], label = 'soen_sim')             
    axs[2].set_xlabel(r'Time [ns]')
    axs[2].set_ylabel(r'$V_{sf}$ [$\mu$V]')
    # axs[2].set_ylim([np.min(V_fq[:])*1e6,np.max(V_fq[:])*1e6])
    axs[2].legend()
    
    plt.show()
    # fig.savefig('figures/'+save_str+'.png') 

    return


def plot_wr_comparison__synapse__vary_Isy(I_sy_vec,target_data_array,actual_data_array):
    
    fig = plt.figure()
    # fig.suptitle(main_title)   
    ax = fig.gca()
    
    color_list_1 = ['blue3','red3','green3','yellow3','blue3','red3','green3','yellow3','blue3','red3','green3','yellow3','blue3','red3','green3','yellow3','blue3','red3','green3','yellow3']
    color_list_2 = ['blue2','red2','green2','yellow2','blue2','red2','green2','yellow2','blue2','red2','green2','yellow2','blue2','red2','green2','yellow2','blue2','red2','green2','yellow2']
    for ii in range(len(I_sy_vec)):
        ax.plot(target_data_array[ii][0,:]*1e9,target_data_array[ii][1,:]*1e6, '-', color = colors[color_list_1[ii]], label = '$Isy$ = {:2.0f}$\mu$A'.format(I_sy_vec[ii]))             
        ax.plot(actual_data_array[ii][0,:]*1e9,actual_data_array[ii][1,:]*1e6, '-', color = colors[color_list_2[ii]])             
    
    ax.set_xlabel(r'Time [ns]')
    ax.set_ylabel(r'$I_{si}$ [$\mu$V]')
    ax.legend()
    
    # dt1 = time_vec[j_sf_peaks[1]]-time_vec[j_sf_peaks[0]]
    # dt2 = time_vec[j_sf_peaks[-1]]-time_vec[j_sf_peaks[-2]]
    # t_lims = [1e9*(time_vec[j_sf_peaks[0]]-2*dt1),1e9*(time_vec[j_sf_peaks[-1]]+2*dt2)]
    # ax.set_xlim(t_lims)
    plt.show()
    
    return



def plot_rate_vs_Isf(j_si_rate_array,Isf_array,V_sf_fit):
    
    Ic = 40
    Ir = 1.1768
    Phi0 = 2.06783375e-15
    
    fig = plt.figure()
    # fig.suptitle('mu1 = {:8.6f}, mu2  = {:8.6f}, V0 = {:7.3f}'.format(mu1,mu2,V0))   
    ax = fig.gca()
    
    # color_list_1 = ['blue3','red3','green3','yellow3','blue3','red3','green3','yellow3','blue3','red3','green3','yellow3','blue3','red3','green3','yellow3','blue3','red3','green3','yellow3']
    # color_list_2 = ['blue2','red2','green2','yellow2','blue2','red2','green2','yellow2','blue2','red2','green2','yellow2','blue2','red2','green2','yellow2','blue2','red2','green2','yellow2']
    for ii in range(len(j_si_rate_array)):
        ax.plot(Isf_array[ii][:]/(Ic-Ir),1e-9*j_si_rate_array[ii][:], linewidth = pp['nominal_linewidth']) 
    ax.plot(Isf_array[-1][:]/(Ic-Ir),1e-9*1e-6*V_sf_fit/Phi0, color = colors['red3'], linewidth = pp['fine_linewidth'], label = 'fit (V/$\Phi_0$)') # label = 'fit; IcRn = {:6.2f}'.format(40*4.125)
    V_sf_fit                 
    
    locs, labels = xticks()  # Get the current locations and labels.
    xticks(np.arange(1.0, 1.6, 0.1))  # Set label locations.

    ax.set_xlabel(r'$I_{sf}/(I_c-I_r)$')
    ax.set_ylabel(r'$r_{sf}$ [kilofluxons per $\mu$s]')
    ax.legend()  
    
    return


def plot_Vsf_vs_Isf(Vsf_array,Isf_array,V_sf_fit,mu1,mu2,V0):
    
    Ic = 40
    Ir = 1.1768
    
    fig = plt.figure()
    fig.suptitle('mu1 = {:8.6f}, mu2  = {:8.6f}, V0 = {:7.3f}'.format(mu1,mu2,V0))   
    ax = fig.gca()
    
    # color_list_1 = ['blue3','red3','green3','yellow3','blue3','red3','green3','yellow3','blue3','red3','green3','yellow3','blue3','red3','green3','yellow3','blue3','red3','green3','yellow3']
    # color_list_2 = ['blue2','red2','green2','yellow2','blue2','red2','green2','yellow2','blue2','red2','green2','yellow2','blue2','red2','green2','yellow2','blue2','red2','green2','yellow2']
    for ii in range(len(Vsf_array)):
        ax.plot(Isf_array[ii][:]/(Ic-Ir),Vsf_array[ii][:], linewidth = pp['nominal_linewidth']) 
    ax.plot(Isf_array[-1][:]/(Ic-Ir),V_sf_fit, color = colors['red3'], linewidth = pp['fine_linewidth'], label = 'fit') # label = 'fit; IcRn = {:6.2f}'.format(40*4.125)
    V_sf_fit                 
    
    locs, labels = xticks()  # Get the current locations and labels.
    xticks(np.arange(1.0, 1.6, 0.1))  # Set label locations.

    ax.set_xlabel(r'$I_{sf}/(I_c-I_r)$')
    ax.set_ylabel(r'$V_{sf}$ [$\mu$V]')
    ax.legend()  
    
    return


def plot_fq_peaks_and_average_voltage_vs_time__1jj(time_vec,V_sf,j_sf_peaks,time_vec_avg,V_sf_avg,main_title):
    
    fig = plt.figure()
    fig.suptitle(main_title)   
    ax = fig.gca()
    
    ax.plot(time_vec*1e9,V_sf*1e6, '-', color = colors['blue3'], label = 'Voltage trace of $J_{sf}$')             
    ax.plot(time_vec[j_sf_peaks]*1e9,V_sf[j_sf_peaks]*1e6, 'x', markersize = pp['big_markersize'], color = colors['red3'], label = 'Fluxon peaks of $J_{sf}$')      
    ax.plot(time_vec_avg,V_sf_avg, '-o', color = colors['green3'], markerfacecolor = colors['green5'], markeredgecolor = colors['green5'], markersize = pp['nominal_markersize'], label = 'Time-averaged voltage')
    
    ax.set_xlabel(r'Time [ns]')
    ax.set_ylabel(r'Voltage [$\mu$V]')
    ax.legend()
    
    dt1 = time_vec[j_sf_peaks[1]]-time_vec[j_sf_peaks[0]]
    dt2 = time_vec[j_sf_peaks[-1]]-time_vec[j_sf_peaks[-2]]
    t_lims = [1e9*(time_vec[j_sf_peaks[0]]-2*dt1),1e9*(time_vec[j_sf_peaks[-1]]+2*dt2)]
    ax.set_xlim(t_lims)
    plt.show()
    
    return


def plot_fq_peaks_and_average_voltage_vs_Isf__1jj(I_sf,V_sf,j_sf_peaks,I_sf_avg,V_sf_avg,main_title):
    
    fig = plt.figure()
    fig.suptitle(main_title)   
    ax = fig.gca()
    
    ax.plot(I_sf,V_sf*1e6, '-', color = colors['blue3'], label = 'Voltage trace of $J_{sf}$')             
    ax.plot(I_sf[j_sf_peaks],V_sf[j_sf_peaks]*1e6, 'x', markersize = pp['big_markersize'], color = colors['red3'], label = 'Fluxon peaks of $J_{sf}$')      
    ax.plot(I_sf_avg,V_sf_avg, '-o', color = colors['green3'], markerfacecolor = colors['green5'], markeredgecolor = colors['green5'], markersize = pp['nominal_markersize'], label = 'Time-averaged voltage')
    
    ax.set_xlabel(r'$I_{sf}$ [$\mu$A]')
    ax.set_ylabel(r'Voltage [$\mu$V]')
    ax.legend()
    
    dI1 = I_sf[j_sf_peaks[1]]-I_sf[j_sf_peaks[0]]
    dI2 = I_sf[j_sf_peaks[-1]]-I_sf[j_sf_peaks[-2]]
    I_lims = [I_sf[j_sf_peaks[0]]-2*dI1,I_sf[j_sf_peaks[-1]]+2*dI2]
    ax.set_xlim(I_lims)
    plt.show()
    
    return


def plot_fq_peaks_and_average_voltage__isolated_JJ(time_vec,I_bias,V_fq,j_peaks,time_avg,I_avg,V_avg,V_fit,V0,mu1,mu2,Ir):
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
    # fig.suptitle(main_title)
        
    axs[0].plot(time_vec*1e9,V_fq*1e6, '-', linewidth = pp['nominal_linewidth'], color = colors['blue3'], label = 'JJ response')
    axs[0].plot(time_vec[j_peaks]*1e9,V_fq[j_peaks]*1e6, 'x', linewidth = pp['nominal_linewidth'], markersize = pp['big_markersize'], color = colors['red3'], label = 'FQ peaks')
    axs[0].plot(time_avg,V_avg, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['green3'], label = 'Average voltage')     
    axs[0].set_xlabel(r'Time [ns]')
    axs[0].set_ylabel(r'Voltage [$\mu$V]')
    axs[0].set_xlim([time_vec[j_peaks[0]]*1e9,time_vec[j_peaks[-1]]*1e9])
    axs[0].legend()
    axs[0].set_title('JJ voltage versus time as bias current is ramped up and down')
     
    axs[1].plot(I_bias*1e6,V_fq*1e6, '-', linewidth = pp['nominal_linewidth'], color = colors['blue3'], label = 'JJ response')
    axs[1].plot(I_bias[j_peaks]*1e6,V_fq[j_peaks]*1e6, 'x', linewidth = pp['nominal_linewidth'], markersize = pp['big_markersize'], color = colors['red3'], label = 'FQ peaks')
    axs[1].plot(I_avg,V_avg, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['green3'], label = 'Average voltage')     
    axs[1].plot(I_avg,V_fit, '-', linewidth = pp['nominal_linewidth'], color = colors['yellow3'], label = '\nVoltage fit:\nV0 = {:10.6f}uV\nmu1 = {:8.6f}\nmu2 = {:8.6f}\nIr = {:9.6f}'.format(V0,mu1,mu2,Ir))     
    axs[1].set_xlabel(r'$I_{bias}$ [$\mu$A]')
    axs[1].set_ylabel(r'Voltage [$\mu$V]')
    axs[1].set_xlim([38,61])
    axs[1].legend()
    axs[1].set_title('JJ voltage versus bias current as bias current is ramped up and down')
    
    return


# def plot_fq_peaks_and_average_voltage__isolated_JJ(time_vec,I_bias,V_fq,j_peaks,time_avg,I_avg,V_avg,V_fit,V0,mu,Ir):
    
#     fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False)   
#     # fig.suptitle(main_title)
        
#     axs[0].plot(time_vec*1e9,V_fq*1e6, '-', linewidth = pp['nominal_linewidth'], color = colors['blue3'], label = 'JJ response')
#     axs[0].plot(time_vec[j_peaks]*1e9,V_fq[j_peaks]*1e6, 'x', linewidth = pp['nominal_linewidth'], markersize = pp['big_markersize'], color = colors['red3'], label = 'FQ peaks')
#     axs[0].plot(time_avg,V_avg, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['green3'], label = 'Average voltage')     
#     axs[0].set_xlabel(r'Time [ns]')
#     axs[0].set_ylabel(r'Voltage [$\mu$V]')
#     axs[0].set_xlim([time_vec[j_peaks[0]]*1e9,time_vec[j_peaks[-1]]*1e9])
#     axs[0].legend()
#     axs[0].set_title('JJ voltage versus time as bias current is ramped up and down')
     
#     axs[1].plot(I_bias*1e6,V_fq*1e6, '-', linewidth = pp['nominal_linewidth'], color = colors['blue3'], label = 'JJ response')
#     axs[1].plot(I_bias[j_peaks]*1e6,V_fq[j_peaks]*1e6, 'x', linewidth = pp['nominal_linewidth'], markersize = pp['big_markersize'], color = colors['red3'], label = 'FQ peaks')
#     axs[1].plot(I_avg,V_avg, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['green3'], label = 'Average voltage')     
#     axs[1].plot(I_avg,V_fit, '-', linewidth = pp['nominal_linewidth'], color = colors['yellow3'], label = '\nVoltage fit:\nV0 = {:10.6f}uV\nmu = {:8.6f}\nIr = {:9.6f}'.format(V0,mu,Ir))     
#     axs[1].set_xlabel(r'$I_{bias}$ [$\mu$A]')
#     axs[1].set_ylabel(r'Voltage [$\mu$V]')
#     axs[1].set_xlim([38,61])
#     axs[1].legend()
#     axs[1].set_title('JJ voltage versus bias current as bias current is ramped up and down')
    
#     return

def plot_Vsf_vs_Isf_fit(I_sf,V_sf,j_sf_peaks,I_sf_avg,V_sf_avg,V_sf_fit,V_sf_fixed,main_title,mu1,mu2,V0):
        
    fig = plt.figure()
    fig.suptitle(main_title)   
    ax = fig.gca()
    
    ax.plot(I_sf,V_sf*1e6, '-', color = colors['blue3'], label = 'Voltage trace of $J_{sf}$')             
    ax.plot(I_sf[j_sf_peaks],V_sf[j_sf_peaks]*1e6, 'x', markersize = pp['big_markersize'], color = colors['red3'], label = 'Fluxon peaks of $J_{sf}$')      
    ax.plot(I_sf_avg,V_sf_avg, '-o', color = colors['green3'], markerfacecolor = colors['green5'], markeredgecolor = colors['green5'], markersize = pp['nominal_markersize'], label = 'Time-averaged voltage')
    ax.plot(I_sf_avg,V_sf_fit, '-o', color = colors['yellow3'], markerfacecolor = colors['yellow5'], markeredgecolor = colors['yellow5'], markersize = pp['nominal_markersize'], label = 'Fit; mu1 = {:6.4f}, mu2 = {:6.4f}, V0 = {:6.2f}'.format(mu1,mu2,V0))
    # ax.plot(I_sf_avg,V_sf_fixed, '-o', color = colors['red3'], markerfacecolor = colors['red5'], markeredgecolor = colors['red5'], markersize = pp['nominal_markersize'], label = 'Fixed; mu1 = {:6.4f}, mu2 = {:6.4f}, V0 = {:6.2f}'.format(2,0.5,40*4.125))
    
    ax.set_xlabel(r'$I_{sf}$ [$\mu$A]')
    ax.set_ylabel(r'Voltage [$\mu$V]')
    ax.legend()
    
    dI1 = I_sf[j_sf_peaks[1]]-I_sf[j_sf_peaks[0]]
    dI2 = I_sf[j_sf_peaks[-1]]-I_sf[j_sf_peaks[-2]]
    I_lims = [I_sf[j_sf_peaks[0]]-2*dI1,I_sf[j_sf_peaks[-1]]+2*dI2]
    ax.set_xlim(I_lims)
    plt.show()
    
    return

def plot_wr_comparison__synapse__Isi_and_Isf(main_title,spike_times,target_drive,actual_drive,target_data,actual_data,sf_data,I_c,I_reset,wr_data_file_name,error_drive,error__si):
    
    tt = time.time()    
    save_str = 'soen_sim_wr_cmpr__sy__'+wr_data_file_name+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))
    
    fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = False)   
    # fig.suptitle(main_title)
        
    axs[0].plot(actual_drive[0,:]*1e6,actual_drive[1,:]*1e6, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['blue3'], label = 'soen_drive')
    axs[0].plot(target_drive[0,:]*1e6,target_drive[1,:]*1e6, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['red3'], label = 'wr_drive')  
    tn1 = np.min(target_drive[1,:])
    tn2 = np.max(target_drive[1,:])
    for ii in range(len(spike_times)):
        if ii == 0:
            axs[0].plot([spike_times[ii]*1e6,spike_times[ii]*1e6],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'], label = 'Spike times')             
        else:
            axs[0].plot([spike_times[ii]*1e6,spike_times[ii]*1e6],[tn1*1e6,tn2*1e6], '-.', color = colors['black'], linewidth = pp['fine_linewidth'])             
    axs[0].set_xlabel(r'Time [$\mu$s]')
    axs[0].set_ylabel(r'$I_{drive}$ [$\mu$A]')
    axs[0].legend()
    axs[0].set_title('{}; Drive signal input SPD to Jsf (error = {:7.5e})'.format(main_title,error_drive))
     
    axs[1].plot(actual_data[0,:]*1e6,actual_data[1,:]*1e6, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['blue3'], label = 'soen_sim')   
    axs[1].plot(target_data[0,:]*1e6,target_data[1,:]*1e6, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['red3'], label = 'WRSpice')             
    axs[1].set_xlabel(r'Time [$\mu$s]')
    axs[1].set_ylabel(r'$I_{si}$ [$\mu$A]')
    axs[1].legend()
    axs[1].set_title('Output signal in the SI loop (error = {:7.5e})'.format(error__si))
      
    axs[2].plot(sf_data[0,:]*1e6,sf_data[1,:]*1e6, '-o', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], color = colors['blue3'])  
    axs[2].plot([actual_drive[0,0]*1e6,actual_drive[0,-1]*1e6],[I_c*1e6,I_c*1e6], '-.', color = colors['greyblue1'], linewidth = pp['fine_linewidth'])                        
    axs[2].plot([actual_drive[0,0]*1e6,actual_drive[0,-1]*1e6],[I_reset*1e6,I_reset*1e6], '-.', color = colors['greyblue5'], linewidth = pp['fine_linewidth'])                        
    axs[2].set_xlabel(r'Time [$\mu$s]')
    axs[2].set_ylabel(r'$I_{sf}$ [$\mu$A]')
    axs[2].set_title('$I_{sf}$')
    
    plt.show()
    fig.savefig('figures/'+save_str+'.png') 

    return


def plot_wr_comparison__synapse__tiles(target_data_array,actual_data_array,spike_times,error_array_drive,error_array_signal,legend_strings):
    
    tt = time.time()    
   
    # 
        
    # axs[0].plot(wr_drive[0,:]*1e6,wr_drive[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'wr_drive') 
    # tn1 = np.min(wr_drive[1,:])
    # tn2 = np.max(wr_drive[1,:])
    # color_list_actual = ['blue3','red3','green3','yellow3']
    # color_list_target = ['blue2','red2','green2','yellow2']
    title_strings = ['Vary $I_{sy}$','Vary $L_{si}$','Vary $tau_{si}$']
    save_strings = ['vary_Isy','vary_Lsi','vary_tausi']
    
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
            axs[i1[jj],i2[jj]].set_title(legend_strings[ii][jj]+'; error_drive = {:7.5e}, error_signal = {:7.5e}'.format(error_array_drive[ii*4+jj],error_array_signal[ii*4+jj]))
    
        plt.show()
        fig.savefig('figures/'+save_str+'.png') 

    return


def plot_wr_comparison__synapse__tiles__with_drive(target_drive_array,actual_drive_array,target_data_array,actual_data_array,spike_times,error_array_drive,error_array_signal,legend_strings):
    
    tt = time.time()    
   
    # 
        
    # axs[0].plot(wr_drive[0,:]*1e6,wr_drive[1,:]*1e6, '-', linewidth = pp['nominal_linewidth'], markersize = pp['nominal_markersize'], label = 'wr_drive') 
    # tn1 = np.min(wr_drive[1,:])
    # tn2 = np.max(wr_drive[1,:])
    # color_list_actual = ['blue3','red3','green3','yellow3']
    # color_list_target = ['blue2','red2','green2','yellow2']
    title_strings = ['Vary $I_{sy}$','Vary $L_{si}$','Vary $tau_{si}$']
    save_strings = ['vary_Isy','vary_Lsi','vary_tausi']
    
    i0 = [0,2,0,2]
    i1 = [1,3,1,3]
    i2 = [0,0,1,1]
    
    # color_list_wr_drive = ['yellow3','red']
    for ii in range(3):
        
        save_str = 'soen_sim_wr_cmpr__sy__3tiles__'+save_strings[ii]+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))    
        fig, axs = plt.subplots(nrows = 4, ncols = 2, sharex = True, sharey = False)
        fig.suptitle(title_strings[ii])
        # print(size(axs))     
        
        for jj in range(4):
            
            axs[i0[jj],i2[jj]].plot(target_drive_array[ii*4+jj][0,:]*1e6,target_drive_array[ii*4+jj][1,:]*1e6, '-', color = colors['yellow3'], linewidth = pp['nominal_linewidth'], label = 'WRSpice drive')
            axs[i0[jj],i2[jj]].plot(actual_drive_array[ii*4+jj][0,:]*1e6,actual_drive_array[ii*4+jj][1,:]*1e6, '-.', color = colors['green3'], linewidth = pp['nominal_linewidth'], label = 'soen_sim drive')   
            
            axs[i1[jj],i2[jj]].plot(target_data_array[ii*4+jj][0,:]*1e6,target_data_array[ii*4+jj][1,:]*1e6, '-', color = colors['red3'], linewidth = pp['nominal_linewidth'], label = 'WRSpice signal')
            axs[i1[jj],i2[jj]].plot(actual_data_array[ii*4+jj][0,:]*1e6,actual_data_array[ii*4+jj][1,:]*1e6, '-.', color = colors['blue3'], linewidth = pp['nominal_linewidth'], label = 'soen_sim signal')   
  
            if jj == 3:
                axs[i0[jj],i2[jj]].legend() 
                axs[i1[jj],i2[jj]].legend() 
            
            axs[i1[jj],i2[jj]].set_xlabel(r'Time [$\mu$s]')
            axs[i0[jj],i2[jj]].set_ylabel(r'$I_{spd}$ [$\mu$A]')
            axs[i1[jj],i2[jj]].set_ylabel(r'$I_{si}$ [$\mu$A]')
            
            axs[i0[jj],i2[jj]].set_title(legend_strings[ii][jj])
            axs[i1[jj],i2[jj]].set_title('e_drv = {:4.2e}, e_sig = {:4.2e}'.format(error_array_drive[ii*4+jj],error_array_signal[ii*4+jj]))
    
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
    
    I_sy = 40
    I_c = 40
    Ir = 1.1768
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

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    num_drives = len(I_drive_list)
    for ii in range(num_drives):
        jj = ii
        I_sf = I_sy+I_drive_list[jj]-I_si_array__scaled[jj][:]
        ax.plot(I_sf/(I_c-Ir),master_rate_array[jj][:]*1e-3, '-', linewidth = pp['nominal_linewidth'], label = 'I_drive = {}'.format(I_drive_list[jj]))    
    ax.set_xlabel(r'$I_{sf}/(I_{c}-I_{r})$')
    ax.set_ylabel(r'$r_{j_{si}}$ [kilofluxons per $\mu$s]')
    # ax.legend()
    plt.show()
    
    return


def plot_syn_rate_array__fit_cmpr(I_si_array,rate_array,I_drive_list,mu1,mu2,V0):
    
    I_sy = 40
    #fig, ax = plt
    # fig.suptitle('master _ sy _ rates') 
    # plt.title('$Isy =$ {} $\mu$A'.format(I_sy))
    
    Ic = 40
    rn = 4.125
    Phi0 = 1e12*2.06783375e-15 
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    num_drives = len(I_drive_list)
    for ii in range(num_drives):
        I_si = I_si_array[num_drives-ii-1][:]
        ax.plot(I_si,rate_array[num_drives-ii-1][:]*1e-3, '-', linewidth = pp['nominal_linewidth'], label = 'I_drive = {}'.format(I_drive_list[num_drives-ii-1])) 
        I_sf = I_sy+I_drive_list[ii]-I_si
        rate = np.real( ( Ic*rn*( ( (I_sf/Ic)**mu1 - 1 )**mu2 ) + V0 )/Phi0 )
        ax.plot(I_si,rate*1e-3)
    ax.set_xlabel(r'$I_{si}$ [$\mu$A]')
    ax.set_ylabel(r'$r_{j_{si}}$ [kilofluxons per $\mu$s]')
    # ax.legend()
    plt.show()
    
    return


def plot_syn_rate_array(**kwargs):

    if 'file_name' in kwargs:
        with open('../_circuit_data/'+kwargs['file_name'], 'rb') as data_file:         
            data_array = pickle.load(data_file)
        # data_array = load_session_data(kwargs['file_name'])
        master_rate_array = data_array['rate_array']
        I_drive_list = data_array['I_drive_list']      
        I_si_array = data_array['I_si_array']
                
    elif 'I_si_array' in kwargs:
        I_si_array = kwargs['I_si_array']
        I_drive_list = kwargs['I_drive_list']
        master_rate_array = kwargs['master_rate_array']
        
    cmap = mp.cm.get_cmap('gist_earth') # 'cividis' 'summer'
    fig = plt.figure()
    if 'file_name' in kwargs:
        fig.suptitle(kwargs['file_name'])
    ax = fig.add_subplot(111, projection='3d')
    
    I_si_min = 1000
    I_si_max = -1000
    I_drive_min = 1e9
    I_drive_max = -1e9
    rate_min = 1000
    rate_max = -1000
    if 'I_drive_reduction_factor' in kwargs:
        I_drive_reduction_factor = kwargs['I_drive_reduction_factor']
    else:
        I_drive_reduction_factor = 1
        
    I_drive_list__reduced = I_drive_list[0::I_drive_reduction_factor]
    num_drives = len(I_drive_list__reduced)
    for ii in range(num_drives):
        
            _ind = (np.abs(np.asarray(I_drive_list[:])-np.asarray(I_drive_list__reduced[ii]))).argmin()
            X3 = np.insert(I_si_array[_ind][:],0,0)            
            Z3 = I_drive_list[_ind]
            Y3 = np.insert(master_rate_array[_ind][:]*1e-3,0,0)
            verts = [(X3[jj],Y3[jj]-0.5) for jj in range(len(X3))]
            ax.add_collection3d(PolyCollection([verts], color = cmap(1-ii/num_drives),alpha=0.3), zs = Z3, zdir='y')
            ax.plot(X3,Y3,Z3,linewidth=4, color = cmap(1-ii/num_drives), zdir='y',alpha=1)
            
            if np.min(I_si_array[_ind][:]) < I_si_min:
                I_si_min = np.min(I_si_array[_ind][:])
            if np.max(I_si_array[_ind][:]) > I_si_max:
                I_si_max = np.max(I_si_array[_ind][:])
                            
            if np.min(I_drive_list[_ind]) < I_drive_min:
                I_drive_min = np.min(I_drive_list[_ind])
            if np.max(I_drive_list[_ind]) > I_drive_max:
                I_drive_max = np.max(I_drive_list[_ind])
                
            if np.min(master_rate_array[_ind][:]*1e-3) < rate_min:
                rate_min = np.min(master_rate_array[_ind][:]*1e-3)
            if np.max(master_rate_array[_ind][:]*1e-3) > rate_max:
                rate_max = np.max(master_rate_array[_ind][:]*1e-3)
           
    ax.set_xticks([0, 10, 20])
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(24)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(24)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(24)
    
    ax.set_xlabel(r'$I_{si}$ [$\mu$A]',fontsize=24, fontweight='bold', labelpad=30) ; ax.set_xlim3d(I_si_min-1,I_si_max+1)    
    ax.set_ylabel('$I_{drive}$ [$\mu$A]',fontsize=24, fontweight='bold', labelpad=30) ; ax.set_ylim3d(I_drive_min-1,I_drive_max+1)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r'$r_{j_{di}}$ [kilofluxons per $\mu$s]',fontsize=24, fontweight='bold', rotation=96,labelpad=10) ; ax.set_zlim3d(rate_min-1,rate_max+1)
    ax.yaxis._axinfo['label']['space_factor'] = 30.0
    
    ax.view_init(45,-30)
    
    plt.show()
    
    return


def plot__syn__error_vs_dt(dt_vec,error_array,error_drive_array):
    
    num_cases = 3
    title_list = ['Vary $I_{sy}$','Vary $L_{si}$','Vary $tau_{si}$']
    legend_list = [['$I_{sy}$ = 21uA','$I_{sy}$ = 27uA','$I_{sy}$ = 33uA','$I_{sy}$ = 39uA'],
                   ['$L_{si}$ = 7.75nH','$L_{si}$ = 77.5nH','$L_{si}$ = 775nH','$L_{si}$ = 7.75$\mu$H'],
                   ['$tau_{si}$ = 10ns','$tau_{si}$ = 50ns','$tau_{si}$ = 250ns','$tau_{si}$ = 1.25$\mu$s']]
    
    color_list = ['blue3','red3','green3','yellow3']
    fig, ax = plt.subplots(nrows = num_cases, ncols = 2)
    for ii in range(num_cases):
        
        for jj in range(4):
            ax[ii,0].loglog(dt_vec*1e9,error_drive_array[ii*4+jj,:], '-o', color = colors[color_list[jj]], markersize = pp['nominal_markersize'], label = legend_list[ii][jj] )    
            ax[ii,1].loglog(dt_vec*1e9,error_array[ii*4+jj,:], '-o', color = colors[color_list[jj]], markersize = pp['nominal_markersize'], label = legend_list[ii][jj] )    
        
        ax[ii,0].set_xlabel(r'dt [ns]')
        ax[ii,0].set_ylabel(r'Drive $Chi^2$ error')
        ax[ii,0].set_title(title_list[ii])
        ax[ii,0].legend()        
        plt.sca(ax[ii,0])
        plt.yticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
        
        ax[ii,1].set_xlabel(r'dt [ns]')
        ax[ii,1].set_ylabel(r'Signal $Chi^2$ error')
        ax[ii,1].set_title(title_list[ii])
        ax[ii,1].legend()       
        plt.sca(ax[ii,1])
        plt.yticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
        
    # grid(True,which='both')
    plt.show()
    
    return


def plot_spd_response(**kwargs): # (time_vec,time_vec_reduced,I_sy_list,I_spd_array,I_spd_array_reduced):

    if 'file_name' in kwargs:
        with open('../_circuit_data/'+kwargs['file_name'], 'rb') as data_file:         
            data_array = pickle.load(data_file)
        spd_response_array = data_array['spd_response_array']
        I_sy_list = data_array['I_sy_list']      
        time_vec = 1e3*data_array['time_vec']
                
    elif 'I_sy_list' in kwargs:
        time_vec = 1e3*kwargs['time_vec']
        I_sy_list = kwargs['I_sy_list']
        spd_response_array = kwargs['spd_response_array']
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
    # fig.suptitle('spd response') 
    if 'I_sy_reduction_factor' in kwargs:
        I_sy_reduction_factor = kwargs['I_sy_reduction_factor']
    else:
        I_sy_reduction_factor = 1
        
    I_sy_list__reduced = I_sy_list[0::I_sy_reduction_factor]
    # print('len(I_sy_list__reduced) = {}; size(spd_response_array) = {}'.format(len(I_sy_list__reduced),np.shape(spd_response_array)))
    for ii in range(len(I_sy_list__reduced)):  
        _ind = (np.abs(I_sy_list[:]-I_sy_list__reduced[ii])).argmin()
        # ax.plot([xx*1e3 for xx in time_vec],I_spd_array[ind])  
        if spd_response_array[_ind,0] < 0:
            spd_response_array[_ind,0]  = 0
        ax.plot(np.insert(time_vec,0,-5),np.insert(spd_response_array[_ind,:],0,0), linewidth = pp['fine_linewidth'], label = 'I_sy = {}uA'.format(I_sy_list__reduced[ii]))    
        # ax.plot(time_vec+5,spd_response_array[_ind], linewidth = pp['fine_linewidth'], label = 'I_sy = {}uA'.format(I_sy_list[_ind]))    
    ax.set_xlabel(r'Time [ns]')
    ax.set_ylabel(r'$I_{spd}$ [$\mu$A]')
    ax.set_xlim([-1,25])
    # ax.set_xlim([-5,200])
    # ax.legend()
    plt.show()

    return


def plot_spd_response__waterfall(**kwargs): # (time_vec,time_vec_reduced,I_sy_list,I_spd_array,I_spd_array_reduced):

    if 'file_name' in kwargs:
        with open('../_circuit_data/'+kwargs['file_name'], 'rb') as data_file:         
            data_array = pickle.load(data_file)
        spd_response_array = data_array['spd_response_array']
        I_sy_list = data_array['I_sy_list']      
        time_vec = 1e3*data_array['time_vec']
                
    elif 'I_sy_list' in kwargs:
        time_vec = 1e3*kwargs['time_vec']
        I_sy_list = kwargs['I_sy_list']
        spd_response_array = kwargs['spd_response_array']
        
    cmap = mp.cm.get_cmap('gist_earth') # 'cividis' 'summer'
    fig = plt.figure()
    if 'file_name' in kwargs:
        fig.suptitle(kwargs['file_name'])
    ax = fig.add_subplot(111, projection='3d')
    
    time_min = 1e9
    time_max = -1e9
    I_sy_min = 1e9
    I_sy_max = -1e9
    spd_min = 1e9
    spd_max = -1e9
    if 'I_sy_reduction_factor' in kwargs:
        I_sy_reduction_factor = kwargs['I_drive_reduction_factor']
    else:
        I_sy_reduction_factor = 1
        
    I_sy_list__reduced = I_sy_list[0::I_sy_reduction_factor]
    num_drives = len(I_sy_list__reduced)
    for ii in range(num_drives):
        
        _ind = (np.abs(np.asarray(I_sy_list[:])-np.asarray(I_sy_list__reduced[ii]))).argmin()
        X3 = np.insert(time_vec[:],0,0)            
        Z3 = I_sy_list[_ind]
        Y3 = np.insert(spd_response_array[_ind,:],0,0)
        verts = [(X3[jj],Y3[jj]-0.5) for jj in range(len(X3))]
        ax.add_collection3d(PolyCollection([verts], color = cmap(1-ii/num_drives),alpha=0.3), zs = Z3, zdir='y')
        ax.plot(X3,Y3,Z3,linewidth=4, color = cmap(1-ii/num_drives), zdir='y',alpha=1)
        
        if np.min(time_vec[:]) < time_min:
            time_min = np.min(time_vec[:])
        if np.max(time_vec[:]) > time_max:
            time_max = np.max(time_vec[:])
                        
        if np.min(I_sy_list[_ind]) < I_sy_min:
            I_sy_min = np.min(I_sy_list[_ind])
        if np.max(I_sy_list[_ind]) > I_sy_max:
            I_sy_max = np.max(I_sy_list[_ind])
            
        if np.min(spd_response_array[_ind,:]) < spd_min:
            spd_min = np.min(spd_response_array[_ind,:])
        if np.max(spd_response_array[_ind,:]) > spd_max:
            spd_max = np.max(spd_response_array[_ind,:])
           
    # ax.set_xticks([0, 10, 20])
    # print('time_min = {}; time_max = {}'.format(time_min,time_max))
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(24)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(24)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(24)
    
    Dt = time_max-time_min
    ax.set_xlabel(r'Time [ns]',fontsize=24, fontweight='bold', labelpad=30) ; ax.set_xlim3d(time_min-0.1*Dt,time_max+0.1*Dt)    
    ax.set_ylabel('$I_{sy}$ [$\mu$A]',fontsize=24, fontweight='bold', labelpad=30) ; ax.set_ylim3d(I_sy_min-1,I_sy_max+1)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r'$I_{spd}$ [$\mu$A]',fontsize=24, fontweight='bold', rotation=96,labelpad=10) ; ax.set_zlim3d(spd_min-1,spd_max+1)
    ax.yaxis._axinfo['label']['space_factor'] = 30.0
    
    ax.view_init(45,-30)
    
    plt.show()
    
    # fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = False)
    # # fig.suptitle('spd response') 
    # plot_list = I_sy_list # [23,28,33,38]
    # num_plot = len(plot_list)
    # for ii in range(num_plot):  
    #     ind = (np.abs(I_sy_list[:]-plot_list[ii])).argmin()
    #     # ax.plot([xx*1e3 for xx in time_vec],I_spd_array[ind])    
    #     ax.plot([xx*1e3 for xx in time_vec_reduced],I_spd_array_reduced[ind], linewidth = pp['fine_linewidth'], label = 'I_sy = {}uA'.format(I_sy_list[ind]))    
    # ax.set_xlabel(r'Time [ns]')
    # ax.set_ylabel(r'$I_{spd}$ [$\mu$A]')
    # ax.set_xlim([0,150])
    # ax.legend()
    # plt.show()

    return

def plot_dend_time_traces(time_vec,j_di,j_di_peaks,min_peak_height,I_di,file_name):
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)
    fig.suptitle(file_name)  
    ax[0].plot(time_vec*1e9,j_di*1e3, '-', color = colors['blue3'], label = '$J_{di}$')             
    ax[0].plot(time_vec[j_di_peaks]*1e9,j_di[j_di_peaks]*1e3, 'x', color = colors['red3'])
    ax[0].plot([time_vec[0]*1e9,time_vec[-1]*1e9],[min_peak_height*1e3,min_peak_height*1e3], ':', color = colors['black'], label = 'peak cutoff')
    ax[0].set_xlabel(r'Time [ns]')
    ax[0].set_ylabel(r'Voltage [mV]')
    ax[0].legend() 
    ax[1].plot(time_vec*1e9,I_di*1e6, '-', color = colors['blue3'], label = '$I_{di}$')             
    ax[1].plot(time_vec[j_di_peaks]*1e9,I_di[j_di_peaks]*1e6, 'x', color = colors['red3'])
    ax[1].set_xlabel(r'Time [ns]')
    ax[1].set_ylabel(r'Current [$\mu$V]')
    ax[1].legend()
    plt.show()
    
    return


def plot_dend_rate_array(**kwargs):
        
    # plt.close('all')
    # Make data.
    
    if 'file_name' in kwargs:
        with open('../_circuit_data/'+kwargs['file_name'], 'rb') as data_file:         
            data_array = pickle.load(data_file)
        # data_array = load_session_data(kwargs['file_name'])
        master_rate_array = data_array['rate_array']
        I_drive_list = data_array['I_drive_list']
        influx_list = data_array['influx_list']        
        I_di_array = data_array['I_di_array']
                
    elif 'I_di_array' in kwargs:
        I_di_array = kwargs['I_di_array']
        I_drive_list = kwargs['I_drive_list']
        influx_list = kwargs['influx_list']
        master_rate_array = kwargs['master_rate_array']
    
    # num_drives = len(I_drive_list)
    num_drives = len(influx_list)
    cmap = mp.cm.get_cmap('gist_earth') # 'cividis' 'summer'
    fig = plt.figure()
    if 'file_name' in kwargs:
        str0 = kwargs['file_name']
        fig.suptitle(str0)
    else:        
        if 'L_left' in kwargs:
            str1 = 'L_left = {:5.2f}pH'.format(kwargs['L_left'])
        if 'I_de' in kwargs:
            str2 = 'I_de = {:5.2f}uA'.format(kwargs['I_de'])
        fig.suptitle('{}; {}'.format(str1,str2))
    
    ax = fig.add_subplot(111, projection='3d')
    
    I_di_min = 1000
    I_di_max = -1000
    # I_drive_min = 1000
    # I_drive_max = -1000
    influx_min = 1e9
    influx_max = -1e9
    rate_min = 1000
    rate_max = -1000
    for ii in range(num_drives):
    #    ax.plot(I_di_array__scaled[ii][:],master_rate_array[ii][:]*1e-3, '-', label = 'I_drive = {}'.format(I_drive_list[ii]))  
            X3 = np.insert(I_di_array[ii][:],0,0)
            # Z3 = I_drive_list[ii]
            Z3 = influx_list[ii]
            Y3 = np.insert(master_rate_array[ii][:]*1e-3,0,0)
            verts = [(X3[jj],Y3[jj]-0.5) for jj in range(len(X3))]
            ax.add_collection3d(PolyCollection([verts],color=cmap(1-ii/num_drives),alpha=0.3),zs=Z3, zdir='y')
            ax.plot(X3,Y3,Z3,linewidth=4, color=cmap(1-ii/num_drives), zdir='y',alpha=1)
            
            if np.min(I_di_array[ii][:]) < I_di_min:
                I_di_min = np.min(I_di_array[ii][:])
            if np.max(I_di_array[ii][:]) > I_di_max:
                I_di_max = np.max(I_di_array[ii][:])
                
            # if np.min(I_drive_list[ii]) < I_drive_min:
            #     I_drive_min = np.min(I_drive_list[ii])
            # if np.max(I_drive_list[ii]) > I_drive_max:
            #     I_drive_max = np.max(I_drive_list[ii])
                            
            if np.min(influx_list[ii]) < influx_min:
                influx_min = np.min(influx_list[ii])
            if np.max(influx_list[ii]) > influx_max:
                influx_max = np.max(influx_list[ii])
                
            if np.min(master_rate_array[ii][:]*1e-3) < rate_min:
                rate_min = np.min(master_rate_array[ii][:]*1e-3)
            if np.max(master_rate_array[ii][:]*1e-3) > rate_max:
                rate_max = np.max(master_rate_array[ii][:]*1e-3)
           
    ax.set_xticks([0, 10, 20])
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(24)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(24)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(24)
    
    ax.set_xlabel(r'$I_{di}$ [$\mu$A]',fontsize=24, fontweight='bold', labelpad=30) ; ax.set_xlim3d(I_di_min-1,I_di_max+1)
    
    # ax.set_ylabel('Idrive [$\mu$A]',fontsize=24, fontweight='bold', labelpad=30) ; ax.set_ylim3d(I_drive_min-1,I_drive_max+1)
    ax.set_ylabel('$\Phi_{in}$ [$\mu$A pH]',fontsize=24, fontweight='bold', labelpad=30) ; ax.set_ylim3d(influx_min-1,influx_max+1)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(r'$r_{j_{di}}$ [kilofluxons per $\mu$s]',fontsize=24, fontweight='bold', rotation=96,labelpad=10) ; ax.set_zlim3d(rate_min-1,rate_max+1)
    ax.yaxis._axinfo['label']['space_factor'] = 30.0
    
    ax.view_init(45,-30)
    
    plt.show()
    
    return

def plot_num_in_burst(I_sy_vec,L_si_vec,tau_si_vec,num_in_burst):
    
    fig = plt.figure()
    # fig.suptitle('Isi vs Isy; tau_si = inf; L_si = {:7.4f} nH'.format(synapse_list[0].integration_loop_total_inductance*1e9)) 
    ax = fig.gca()
    
    num_I_sy = len(I_sy_vec)
    num_L_si = len(L_si_vec)
    num_tau_si = len(tau_si_vec)
    for ii in range(num_L_si):
        for jj in range(num_I_sy):
            ax.plot(tau_si_vec*1e6,num_in_burst[ii,jj,:], '-o', linewidth = pp['nominal_linewidth'], label = 'I_sy = {:5.2f}uA, L_si = {:7.2f}nH'.format(I_sy_vec[jj]*1e6,L_si_vec[ii]*1e9))
    
    ax.set_xlabel(r'$\tau_{si}$ [$\mu$s]')
    ax.set_ylabel(r'Num spikes')
    ax.legend()
    
    plt.show()
    
    return

def plot_phase_portrait(neuron_instance):
    
    drive_vec = neuron_instance.influx_vec__no_refraction
    ref_dend_name = '{}__r'.format(neuron_instance.name)
    refraction_vec = neuron_instance.dendrites[ref_dend_name].M*neuron_instance.dendrites[ref_dend_name].I_di_vec
    
    fig = plt.figure()
    # fig.suptitle('tau_si = {:5.2f} ns; tau_ri = {:5.2f}'.format(synapse_list[0].integration_loop_total_inductance*1e9)) 
    ax = fig.gca()
    
    ax.plot(drive_vec,refraction_vec, color = colors['blue3']) # , '-o', linewidth = pp['nominal_linewidth'], label = 'I_sy = {:5.2f}uA, L_si = {:7.2f}nH'.format(I_sy_vec[jj]*1e6,L_si_vec[ii]*1e9)
    ax.plot(drive_vec[neuron_instance.voltage_peaks],refraction_vec[neuron_instance.voltage_peaks], 'x', color = colors['red3'])
    for syn_name in neuron_instance.input_synaptic_connections:
        for _time in neuron_instance.synapses[syn_name].input_spike_times:
            _time_ind = (np.abs(neuron_instance.time_vec-_time)).argmin()
            ax.plot(drive_vec[_time_ind],refraction_vec[_time_ind], 'o', color = colors['green3'])
        
    
    ax.set_xlabel(r'$\Phi_{+}^{nr}$ [$\mu$A pH]')
    ax.set_ylabel(r'$\Phi_{ref}^{nr}$ [$\mu$A pH]')
    # ax.legend()
    
    x_min = np.min(drive_vec)
    x_max = np.max(drive_vec)
    x_range = x_max-x_min
    ax.set_xlim([x_min-0.05*x_range,x_max+0.05*x_range])
    
    plt.show()
    
    return

    # color_list = [colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
    #               colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
    #               colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
    #               colors['red4'],colors['red3'],colors['red2'],colors['red1'],
    #               colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
    #               colors['green4'],colors['green3'],colors['green2'],colors['green1'],
    #               colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
    #               colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
    #               colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
    #               colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
    #               colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
    #               colors['red4'],colors['red3'],colors['red2'],colors['red1'],
    #               colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
    #               colors['green4'],colors['green3'],colors['green2'],colors['green1'],
    #               colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
    #               colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
    #               colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
    #               colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
    #               colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
    #               colors['red4'],colors['red3'],colors['red2'],colors['red1'],
    #               colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
    #               colors['green4'],colors['green3'],colors['green2'],colors['green1'],
    #               colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
    #               colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
    #               colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
    #               colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
    #               colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
    #               colors['red4'],colors['red3'],colors['red2'],colors['red1'],
    #               colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
    #               colors['green4'],colors['green3'],colors['green2'],colors['green1'],
    #               colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
    #               colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
    #               colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
    #               colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
    #               colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
    #               colors['red4'],colors['red3'],colors['red2'],colors['red1'],
    #               colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
    #               colors['green4'],colors['green3'],colors['green2'],colors['green1'],
    #               colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
    #               colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
    #               colors['blue1'],colors['blue2'],colors['blue3'],colors['blue4'],colors['blue5'],
    #               colors['blue4'],colors['blue3'],colors['blue2'],colors['blue1'],
    #               colors['red1'],colors['red2'],colors['red3'],colors['red4'],colors['red5'],
    #               colors['red4'],colors['red3'],colors['red2'],colors['red1'],
    #               colors['green1'],colors['green2'],colors['green3'],colors['green4'],colors['green5'],
    #               colors['green4'],colors['green3'],colors['green2'],colors['green1'],
    #               colors['yellow1'],colors['yellow2'],colors['yellow3'],colors['yellow4'],colors['yellow5'],
    #               colors['yellow4'],colors['yellow3'],colors['yellow2'],colors['yellow1'],
    #               ]
