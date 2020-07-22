#%%
import numpy as np
from matplotlib import pyplot as plt
import time

### Alex extra imports
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'soen_sim'))
try:
    import util  # this will fail unless you are on Jeff's computer
except ImportError:
    import util_safe  # Local fallback
    sys.modules['util'] = util_safe  # any further "import util" will receive the fallback version

# this only works if nengo is installed: "pip install nengo"
from nengo import Simulator, Network, Connection, Probe, Node
# this only works if nengo_soen is a sister directory of soen_sim
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'nengo_soen'))
from nengo_soen.processes import Photons, SPD, SI_Loop, synapse__1jj_ode
from nengo_soen.SPD_models import Lspd_typ
# what I used for saving
from matplotlib.backends.backend_pdf import PdfPages
###

# from soen_sim import input_signal, synapse, dendrite, neuron
from _plotting import plot_wr_comparison__synapse__Isi_and_Isf, plot_wr_comparison__synapse, plot_wr_comparison__synapse__tiles, plot_wr_comparison__synapse__vary_Isy, plot_wr_comparison__synapse__tiles__with_drive
from _functions import read_wr_data, chi_squared_error
from util import physical_constants
from soen_sim import input_signal, synapse
p = physical_constants()

plt.close('all')

#%%
I_spd = 20e-6

spike_times = [5e-9,55e-9,105e-9,155e-9,205e-9,255e-9,305e-9,355e-9,505e-9,555e-9,605e-9,655e-9,705e-9,755e-9,805e-9,855e-9]
# Using multiple lines so that commenting is easier --atait
I_sy_vec =  [
             21e-6,     27e-6,  33e-6,  39e-6,
             # 28e-6,     28e-6,  28e-6,  28e-6,
             # 32e-6,     32e-6,  32e-6,  32e-6
             ]
L_si_vec =  [
             77.5e-9,   77.5e-9,77.5e-9, 77.5e-9,
             # 7.75e-9,   77.5e-9,775e-9, 7.75e-6,
             # 775e-9,    775e-9, 775e-9, 775e-9
             ]
tau_si_vec =[
             250e-9,    250e-9, 250e-9, 250e-9,
             # 250e-9,    250e-9, 250e-9, 250e-9,
             # 10e-9,     50e-9,  250e-9, 1.25e-6
             ]

dt = 0.01e-9
tf = 1e-6

# create sim_params dictionary
sim_params = dict()
sim_params['dt'] = dt
sim_params['tf'] = tf
sim_params['synapse_model'] = 'ode' # __spd_delta

target_data_array = []
target_drive_array = []

actual_data_array = []
actual_drive_array = []

error_array_drive = []
error_array_signal = []

calculate_chi_squared = False
plot_each = True
save_figures = False

#%%

num_files = len(I_sy_vec)
t_tot = time.time()
for ii in range(num_files): # range(1): #

    print('\nii = {} of {}\n'.format(ii+1,num_files))

    #load WR data.      NO! --atait
    # file_name = 'syn_1jj_Ispd20.00uA_trep50ns_Isy{:04.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii]*1e6,L_si_vec[ii]*1e9,tau_si_vec[ii]*1e9)
    # data_dict = read_wr_data('wrspice_data/test_data/1jj/'+file_name)
    # target_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
    # target_drive_array.append(target_drive)
    # target_data = np.vstack((data_dict['time'],data_dict['L1#branch']))
    # target_data_array.append(target_data)

    # initialize input signal
    input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)

    # initialize synapse
    synapse_1 = synapse('sy', num_jjs = 1, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si_vec[ii],
                        integration_loop_self_inductance = L_si_vec[ii], integration_loop_output_inductance = 0e-12,
                        synaptic_bias_currents = [I_spd,I_sy_vec[ii]],
                        input_signal_name = 'in', synapse_model_params = sim_params)

    # run simulation
    t0 = time.time()
    synapse_1.run_sim()
    print('Python time {:.3f}s'.format(time.time() - t0))

    # get the data out
    actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
    actual_drive_array.append(actual_drive)
    actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:]))
    sf_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_sf[:]))
    actual_data_array.append(actual_data)

    ### mirror ###
    new_style = True
    nengo_bells_and_whistles = True and not new_style

    file_name = 'nengo_Isy{:.0f}u-L{:.0f}n-tau{:.0f}n'.format(I_sy_vec[ii]*1e6, L_si_vec[ii]*1e9, tau_si_vec[ii]*1e9)

    # initialize input signal
    input_1 = Photons([(t, 10) for t in spike_times])

    # initialize synapse
    r1 = 8.25  # I got this from line 328 of soen_sim
    if new_style:
        # some logic pulled from synapse class
        L_spd = 247.5e-9
        L_list = [(L_spd),(L_si_vec[ii])]
        tau_fall = tau_si_vec[ii]
        r_list = [r1, L_list[1]/tau_fall]

        syn_process = synapse__1jj_ode(L_list=L_list, r_list=r_list, I_bias_list=[I_spd,I_sy_vec[ii]],
                                       refraction=0, stochastic=False, give_Ispd=True)

    else:
        spd_process = SPD(tau=L_spd_typ/r1)
        sil_process = SI_Loop(tausi=tau_si_vec[ii], Lsi=L_si_vec[ii])
        if nengo_bells_and_whistles:
            with Network('Single JJ synapse') as SIL_network:
                pho_node = Node(input_1)
                spd_node = Node(spd_process)
                sil_node = Node(sil_process)
                Connection(pho_node, spd_node[0])
                Connection(Node(I_spd), spd_node[1])
                Connection(spd_node, sil_node)
                Connection(Node(I_sy_vec[ii]), sil_node)
                spd_probe = Probe(spd_node)
                sil_probe = Probe(sil_node)

    # do simulation
    t0 = time.time()
    if nengo_bells_and_whistles:
        with Simulator(SIL_network, dt=dt) as sim:
            sim.run(tf)
        tvec = sim.trange()
    else:
        pho_sig = input_1.run(tf, dt=dt)
        if new_style:
            both_sig = syn_process.apply(pho_sig, dt=dt)
            sil_sig = both_sig[:, 0, :]
            spd_sig = both_sig[:, 1, :]
        else:
            spd_sig = spd_process.apply(np.hstack((pho_sig, I_spd * np.ones_like(pho_sig))), dt=dt)
            sil_sig = sil_process.apply(I_sy_vec[ii] + spd_sig, dt=dt)
        tvec = input_1.trange(tf, dt=dt)
    print('Nengo time {:.3f}s'.format(time.time() - t0))

    # get the data out
    if nengo_bells_and_whistles:
        target_drive = np.vstack((tvec, sim.data[spd_probe][:,0]))
        target_data = np.vstack((tvec, sim.data[sil_probe][:,0]))
    else:
        target_drive = np.vstack((tvec, np.squeeze(spd_sig)))
        target_data = np.vstack((tvec, np.squeeze(sil_sig)))


    if calculate_chi_squared == True:
        error_drive = chi_squared_error(target_drive,actual_drive)
        error_array_drive.append(error_drive)
        error_signal = chi_squared_error(target_data,actual_data)
        error_array_signal.append(error_signal)
        if plot_each == True:
            plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,error_drive,error_signal)
    else:
        if plot_each == True:
            fig = plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,1,1)
            if save_figures:
                pp = PdfPages('nengo_verification/{}.pdf'.format(file_name))
                pp.savefig(fig)
                pp.close()

elapsed = time.time() - t_tot
print('soen_sim duration = '+str(elapsed)+'s')

#%%

if calculate_chi_squared == True:
    legend_strings = [ ['Isy = {:5.2f}uA'.format(I_sy_vec[0]*1e6),
                        'Isy = {:5.2f}uA'.format(I_sy_vec[1]*1e6),
                        'Isy = {:5.2f}uA'.format(I_sy_vec[2]*1e6),
                        'Isy = {:5.2f}uA'.format(I_sy_vec[3]*1e6)],
                       ['Lsi = {:7.2f}nH'.format(L_si_vec[4]*1e9),
                        'Lsi = {:7.2f}nH'.format(L_si_vec[5]*1e9),
                        'Lsi = {:7.2f}nH'.format(L_si_vec[6]*1e9),
                        'Lsi = {:7.2f}nH'.format(L_si_vec[7]*1e9)],
                       ['tau_si = {:7.2f}ns'.format(tau_si_vec[8]*1e9),
                        'tau_si = {:7.2f}ns'.format(tau_si_vec[9]*1e9),
                        'tau_si = {:7.2f}ns'.format(tau_si_vec[10]*1e9),
                        'tau_si = {:7.2f}ns'.format(tau_si_vec[11]*1e9)] ]
    plot_wr_comparison__synapse__tiles__with_drive(target_drive_array,actual_drive_array,target_data_array,actual_data_array,spike_times,error_array_drive,error_array_signal,legend_strings)


#%% vary Isy more

if 1 == 2:

    dI = 1
    I_sy_vec = np.arange(21,39+dI,dI)
    L_si = 77.5e-9
    tau_si = 250e-9

    num_files = len(I_sy_vec)
    t_tot = time.time()

    target_data_array = []
    actual_data_array = []
    for ii in range(num_files): # range(1): #

        print('\nvary Isy, ii = {} of {}\n'.format(ii+1,num_files))

        #load WR data
        file_name = 'syn_1jj_Ispd20.00uA_trep50ns_Isy{:04.2f}uA_Lsi{:07.2f}nH_tausi{:04.0f}ns_dt10.0ps_tsim1000ns.dat'.format(I_sy_vec[ii],L_si*1e9,tau_si*1e9)
        data_dict = read_wr_data('wrspice_data/test_data/1jj/'+file_name)
        target_drive = np.vstack((data_dict['time'],data_dict['L0#branch']))
        target_data = np.vstack((data_dict['time'],data_dict['L1#branch']))
        target_data_array.append(target_data)

        # initialize input signal
        input_1 = input_signal('in', input_temporal_form = 'arbitrary_spike_train', spike_times = spike_times)

        # initialize synapse
        synapse_1 = synapse('sy', num_jjs = 1, integration_loop_temporal_form = 'exponential', integration_loop_time_constant = tau_si,
                            integration_loop_self_inductance = L_si, integration_loop_output_inductance = 0e-12,
                            synaptic_bias_currents = [I_spd,1e-6*I_sy_vec[ii]],
                            input_signal_name = 'in', synapse_model_params = sim_params)

        synapse_1.run_sim()

        actual_drive = np.vstack((synapse_1.time_vec[:],synapse_1.I_spd[:]))
        actual_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_si[:]))
        sf_data = np.vstack((synapse_1.time_vec[:],synapse_1.I_sf[:]))
        actual_data_array.append(actual_data)

    if calculate_chi_squared == True:
        error_drive = chi_squared_error(target_drive,actual_drive)
        error_array_drive.append(error_drive)
        error_signal = chi_squared_error(target_data,actual_data)
        error_array_signal.append(error_signal)
        if plot_each == True:
            plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,error_drive,error_signal)
    else:
        if plot_each == True:
            plot_wr_comparison__synapse(file_name,spike_times,target_drive,actual_drive,target_data,actual_data,file_name,1,1)

    elapsed = time.time() - t_tot
    print('soen_sim duration = '+str(elapsed)+' s for vary I_sy')

    plot_wr_comparison__synapse__vary_Isy(I_sy_vec,target_data_array,actual_data_array)
