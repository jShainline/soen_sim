import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import time

from util import color_dictionary
colors = color_dictionary()


def plot_params():
    
    pp = dict()
    pp['title_font_size'] = 20
    pp['subtitle_font_size'] = 10
    pp['axes_labels_font_size'] = 16
    pp['tick_labels_font_size'] = 14
    pp['legend_font_size'] = 14
    pp['nominal_linewidth'] = 2
    pp['fine_linewidth'] = 0.75
    pp['bold_linewidth'] = 3
    pp['nominal_markersize'] = 3
    tn = 4*8.6/2.54
    pp['fig_size'] = (tn,tn/1.618)
    pp['axes_linewidth'] = 1.5
    
    pp['major_tick_width'] = 1
    pp['major_tick_length'] = 7
    pp['minor_tick_width'] = 0.5
    pp['minor_tick_length'] = 3
    
    return pp 

pp = plot_params()

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = 'Verdana'#'Computer Modern Sans Serif'

plt.rcParams['figure.figsize'] = pp['fig_size']
plt.rcParams['figure.titlesize'] = pp['title_font_size']

plt.rcParams['axes.prop_cycle'] = cycler('color', [colors['blue_3'],colors['red_3'],colors['green_3'],colors['yellow_3']])
plt.rcParams['axes.linewidth'] = pp['axes_linewidth']
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.titlesize'] = pp['subtitle_font_size']
plt.rcParams['axes.labelsize'] = pp['axes_labels_font_size']

plt.rcParams['legend.fontsize'] = pp['legend_font_size']
plt.rcParams['legend.loc'] = 'best'

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'png'

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
    
    plt.rcParams['figure.figsize'] = pp['fig_size']
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