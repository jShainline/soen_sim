import numpy as np
import pickle
import time
from matplotlib import pyplot as plt
from pylab import *

from util import physical_constants
from _plotting import plot_dendritic_drive, plot_wr_comparison, plot_error_mat

def synapse_time_stepper(time_vec,present_time_index,input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall):
    
    # print('gamma1 = {}'.format(gamma1))
    _pti = present_time_index
    _pt = time_vec[_pti] #present time
    I_si = 0
    if len(input_spike_times) > 0:
        num_tau_retain = 15
        idx_start = (np.abs(input_spike_times-(_pt-num_tau_retain*tau_fall))).argmin()
        # idx_start = 0
        for ii in range(len(input_spike_times[idx_start:])):
            qq = ii+idx_start
            if _pt > input_spike_times[qq]:
                if _pt-input_spike_times[qq] <= tau_rise:
                    I_si += synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si,tau_rise,tau_fall)*\
                        ( (1/tau_rise**gamma3)*(_pt - input_spike_times[qq])**gamma3 )*\
                            np.exp(tau_rise/tau_fall)*np.exp(-(_pt-input_spike_times[qq])/tau_fall)
                else:
                    I_si += synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si,tau_rise,tau_fall)*\
                        np.exp(tau_rise/tau_fall)*np.exp(-(_pt-input_spike_times[qq])/tau_fall)
                    
    return I_si

def synaptic_response_function(time_vec,input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall):

    I_si_mat = np.zeros([len(time_vec),len(input_spike_times)])

    for ii in range(len(input_spike_times)):
        ind_vec = np.argwhere( time_vec > input_spike_times[ii] )
        I_si_vec_temp = np.sum(I_si_mat, axis = 1)
        # I_si_mat(ind_vec(1:end),ii) = f__synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si_vec_temp(ind_vec(ii)-1))*(1-exp(-(time_vec(ind_vec(1:end))-input_spike_times(ii))/tau_rise)).*exp(-(time_vec(ind_vec(1:end))-input_spike_times(ii))/tau_fall);
        for jj in range(len(ind_vec)):
            if time_vec[ind_vec[jj]]-input_spike_times[ii] <= tau_rise:
                I_si_mat[ind_vec[jj],ii] = synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si_vec_temp[ind_vec[jj]-1],tau_rise,tau_fall)*\
                ( (1/tau_rise**gamma3)*( time_vec[ind_vec[jj]] - input_spike_times[ii] )**gamma3 )*\
                np.exp(tau_rise/tau_fall)*\
                np.exp(-(time_vec[ind_vec[jj]]-input_spike_times[ii])/tau_fall)
            else:
                I_si_mat[ind_vec[jj],ii] = synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si_vec_temp[ind_vec[jj]-1],tau_rise,tau_fall)*\
                np.exp(tau_rise/tau_fall)*\
                np.exp(-(time_vec[ind_vec[jj]]-input_spike_times[ii])/tau_fall)
        I_si_vec = np.sum(I_si_mat, axis = 1);

    return I_si_vec

def synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si,tau_rise,tau_fall):

    if I_si >= 0 and I_si < I_si_sat:
        A = I_0
        I_prefactor = min([A*(1-(I_si/I_si_sat)**gamma1)**gamma2, (I_si_sat-I_si)*np.exp(tau_rise/tau_fall)]);
        # I_prefactor = A*(1-log(I_si/I_si_sat)/log(gamma1))^gamma2;
        # I_prefactor = I_0*(I_si_sat-I_si)/I_si_sat
        # I_prefactor = I_0*(1-exp((I_si_sat-I_si)/I_si_sat))
    else:
        I_prefactor = 0

    #print('\n\nI_si = %g',I_si)
    #print('\n\nI_prefactor = %g',I_prefactor)

    return I_prefactor

def dendrite_time_stepper(time_vec,I_drive,L3,tau_di):
    
    with open('../master_rate_matrix.soen', 'rb') as data_file:         
        data_array_imported = pickle.load(data_file)
    
    I_di_list__imported = data_array_imported['I_di_list']
    I_drive_vec__imported = data_array_imported['I_drive_vec']
    master_rate_matrix__imported = data_array_imported['master_rate_matrix']
        
    p = physical_constants()
    Phi0 = p['Phi0']
    I_fq = Phi0/L3
    
    I_di_vec = np.zeros([len(time_vec),1])
    for ii in range(len(time_vec)-1):
        dt = time_vec[ii+1]-time_vec[ii]
                               
        if I_drive[ii] > 18.6e-6:
            ind1 = (np.abs(np.asarray(I_drive_vec__imported)-I_drive[ii])).argmin()
            ind2 = (np.abs(np.asarray(I_di_list__imported[ind1])-I_di_vec[ii])).argmin()
            rate = master_rate_matrix__imported[ind1,ind2]
            # linear interpolation
            # rate = np.interp(I_drive[ii],I_drive_vec__imported,master_rate_matrix__imported[:,ind2])            
        else:
            rate = 0

        I_di_vec[ii+1] = rate*I_fq*dt + (1-dt/tau_di)*I_di_vec[ii]        
    
    return I_di_vec

def dendritic_time_stepper(time_vec,R,I_drive,I_b,Ic,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,tau_di,mu_1,mu_2):
    
    p = physical_constants()
    Phi0 = p['Phi0']
    prefactor = Ic*R/Phi0
    I_fq = Phi0/L3
        
    #initial approximations
    Lj0 = Ljj(Ic,0)
    Iflux = 0
    Idr2_prev = ((Lm2+Ldr1+Lj0)*I_b[0]+M_direct*Iflux)/( Lm2+Ldr1+Ldr2+2*Lj0 + (Lm2+Ldr1+Lj0)*(Ldr2+Lj0)/L1 )
    Idr1_prev = I_b[0]-( 1 + (Ldr2+Lj0)/L1 )*Idr2_prev
    Ij2_prev = I_b[1]
    Ij3_prev = I_b[2]
    
    I_di_vec = np.zeros([len(time_vec),1])
    # Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,0,I_b[0],I_b[1],I_b[2],M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
    # print('Idr1 = {}uA, Idr2 = {}uA, Ij2 = {}uA, Ij3 = {}uA'.format(Idr1_next*1e6,Idr2_next*1e6,Ij2_next*1e6,Ij3_next*1e6))
    for ii in range(len(time_vec)-1):
        dt = time_vec[ii+1]-time_vec[ii]
                               
                                                              #dendrite_current_splitting(Ic,  Iflux,        Ib1,   Ib2,   Ib3,   M,       Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
        Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,I_drive[ii+1],I_b,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
        # I_di_sat = I_di_sat_of_I_dr_2(Idr2_next)
        
        Ljdr2 = Ljj(Ic,Idr2_next)
        Lj2 = Ljj(Ic,Ij2_next)
        Lj3 = Ljj(Ic,Ij3_next)        
        # Ljdr2 = Ljj(Ic,0)
        # Lj2 = Ljj(Ic,0)
        # Lj3 = Ljj(Ic,0)
        
        Idr1_prev = Idr1_next
        Idr2_prev = Idr2_next
        Ij2_prev = Ij2_next#(Lj3/(L2+Lj2))*I_di_vec[ii]
        Ij3_prev = Ij3_next
        
        # I_j_df_fluxon_soen = Phi0/(L1+Ldr2+Ljdr2+Lj2)
        # I_j_2_fluxon_soen = Phi0/(Lj2+L_pp)
        # I_j_3_fluxon_soen = Phi0/(L3+Lj3)
        
        I_loop2_from_di = (Lj3/(L2+Lj2))*I_di_vec[ii]
        I_loop1_from_loop2 = (Lj2/(L1+Ljdr2+Ldr2))*I_loop2_from_di
        # print('I_loop2_from_di = {}uA, I_loop1_from_loop2 = {}uA'.format(I_loop2_from_di*1e6,I_loop1_from_loop2*1e6))
        
        Idr2_next -= I_loop1_from_loop2
        Ij2_next -= I_loop2_from_di
        Ij3_next -= I_di_vec[ii] - I_loop2_from_di        
                
        L_ppp = Lj3*L3/(Lj3+L3)
        L_pp = L2+L_ppp
        L_p = Lj2*L_pp/(Lj2+L_pp)
        # print('L_p = {}pH, L_pp = {}pH, L_ppp = {}pH'.format(L_p*1e12,L_pp*1e12,L_ppp*1e12))
        
        large_number = 1e9
        
        I_flux_1 = 6e-6
        I_flux_2 = 20e-6
        
        Ij2_next += I_flux_1 # (Phi0/(L1+L_p))*(L_pp)/(Lj2+L_pp)#(Lj3/(L2+Lj2))*I_di_vec[ii]
        # print('Ij2_next += {}uA'.format(1e6*(Phi0/(L1+L_p))*(L_pp)/(Lj2+L_pp)))
        # Ij3_next += (Phi0/L_pp)*(L3/(L3+Lj3)) + (Phi0/(L1+L_p))*L3/(Lj3+L3) - I_di_vec[ii]
        Ij3_next += I_flux_2 # (Phi0/L_pp)*(L3/(L3+Lj3))
        # print('Ij3_next += {}uA'.format(1e6*(Phi0/L_pp)*(L3/(L3+Lj3))))
        # print('term_1 = {}; term_2 = {}'.format( (Phi0/L_pp)*(L3/(L3+Lj3)) , (Phi0/(L1+L_p))*L3/(Lj3+L3) ) )
        if Idr2_next > Ic:
            factor_1 = inter_fluxon_interval(Idr2_next) # ( (Idr2_next/Ic)**mu_1 - 1 )**(-mu_2)            
        else:
            factor_1 = large_number
        if Ij2_next > Ic:
            factor_2 = inter_fluxon_interval(Ij2_next) # ( (Ij2_next/Ic)**mu_1 - 1 )**(-mu_2)  
        else:
            factor_2 = large_number
        if Ij3_next > Ic:
            factor_3 = inter_fluxon_interval(Ij3_next) # ( (Ij3_next/Ic)**mu_1 - 1 )**(-mu_2)  
        else:
            factor_3 = large_number

        # print('factor_1 = {}, factor_2 = {}, factor_3 = {}'.format(factor_1,factor_2,factor_3))
        r_tot = (factor_1+factor_2+factor_3)**(-1)
        I_di_vec[ii+1] = r_tot*I_fq*dt + (1-dt/tau_di)*I_di_vec[ii]        
    
    return I_di_vec

def dendritic_time_stepper_old2(time_vec,A_prefactor,I_drive,I_b,I_th,M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,tau_di,mu_1,mu_2,mu_3,mu_4):
        
    #initial approximations
    Lj0 = Ljj(I_th,0)
    Iflux = 0
    Idr2_prev = ((Lm2+Ldr1+Lj0)*I_b[0]+M_direct*Iflux)/( Lm2+Ldr1+Ldr2+2*Lj0 + (Lm2+Ldr1+Lj0)*(Ldr2+Lj0)/L1 )
    Idr1_prev = I_b[0]-( 1 + (Ldr2+Lj0)/L1 )*Idr2_prev
    Ij2_prev = I_b[1]
    Ij3_prev = I_b[2]
    
    I_di_vec = np.zeros([len(time_vec),1])
    for ii in range(len(time_vec)-1):
        dt = time_vec[ii+1]-time_vec[ii]
                                                                                            
        Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(I_th,I_drive[ii+1],I_b[0],I_b[1],I_b[2],M_direct,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
        I_di_sat = I_di_sat_of_I_dr_2(Idr2_next)
        if Idr2_next > I_th:
            factor_1 = ( (Idr2_next/I_th)**mu_1 - 1 )**mu_2            
        else:
            factor_1 = 0
        if I_di_vec[ii] <= I_di_sat:
            factor_2 = ( 1-(I_di_vec[ii]/I_di_sat)**mu_3 )**mu_4
        else:
            factor_2 = 0
        I_di_vec[ii+1] = dt * A_prefactor * factor_1 * factor_2 + (1-dt/tau_di)*I_di_vec[ii]
        Idr1_prev = Idr1_next
        Idr2_prev = Idr2_next
        Ij2_prev = Ij2_next
        Ij3_prev = Ij3_next
    
    return I_di_vec

def dendritic_drive__piecewise_linear(time_vec,pwl):
    
    input_signal__dd = np.zeros([len(time_vec),1])
    for ii in range(len(pwl)-1):
        t1_ind = (np.abs(np.asarray(time_vec)-pwl[ii][0])).argmin()
        t2_ind = (np.abs(np.asarray(time_vec)-pwl[ii+1][0])).argmin()
        slope = (pwl[ii+1][1]-pwl[ii][1])/(pwl[ii+1][0]-pwl[ii][0])
        # print('t1_ind = {}'.format(t1_ind))
        # print('t2_ind = {}'.format(t2_ind))
        # print('slope = {}'.format(slope))
        partial_time_vec = time_vec[t1_ind:t2_ind+1]
        input_signal__dd[t1_ind] = pwl[ii][1]
        for jj in range(len(partial_time_vec)-1):
            input_signal__dd[t1_ind+jj+1] = input_signal__dd[t1_ind+jj]+(partial_time_vec[jj+1]-partial_time_vec[jj])*slope
    input_signal__dd[t2_ind:] = input_signal__dd[t2_ind]*np.ones([len(time_vec)-t2_ind,1])
    
    return input_signal__dd

def dendritic_drive__exp_pls_train__LR(time_vec,exp_pls_trn_params):
        
    t_r1_start = exp_pls_trn_params['t_r1_start']
    t_r1_rise = exp_pls_trn_params['t_r1_rise']
    t_r1_pulse = exp_pls_trn_params['t_r1_pulse']
    t_r1_fall = exp_pls_trn_params['t_r1_fall']
    t_r1_period = exp_pls_trn_params['t_r1_period']
    value_r1_off = exp_pls_trn_params['value_r1_off']
    value_r1_on = exp_pls_trn_params['value_r1_on']
    r2 = exp_pls_trn_params['r2']
    L1 = exp_pls_trn_params['L1']
    L2 = exp_pls_trn_params['L2']
    Ib = exp_pls_trn_params['Ib']
    
    # make vector of r1(t)
    sq_pls_trn_params = dict()
    sq_pls_trn_params['t_start'] = t_r1_start
    sq_pls_trn_params['t_rise'] = t_r1_rise
    sq_pls_trn_params['t_pulse'] = t_r1_pulse
    sq_pls_trn_params['t_fall'] = t_r1_fall
    sq_pls_trn_params['t_period'] = t_r1_period
    sq_pls_trn_params['value_off'] = value_r1_off
    sq_pls_trn_params['value_on'] = value_r1_on
    # print('making resistance vec ...')
    r1_vec = dendritic_drive__square_pulse_train(time_vec,sq_pls_trn_params)
    
    input_signal__dd = np.zeros([len(time_vec),1])
    # print('time stepping ...')
    for ii in range(len(time_vec)-1):
        # print('ii = {} of {}'.format(ii+1,len(time_vec)-1))
        dt = time_vec[ii+1]-time_vec[ii]
        input_signal__dd[ii+1] = input_signal__dd[ii]*( 1 - dt*(r1_vec[ii]+r2)/(L1+L2) ) + dt*Ib*r1_vec[ii]/(L1+L2)
    
    return input_signal__dd

def dendritic_drive__exponential(time_vec,exp_params):
        
    t_rise = exp_params['t_rise']
    t_fall = exp_params['t_fall']
    tau_rise = exp_params['tau_rise']
    tau_fall = exp_params['tau_fall']
    value_on = exp_params['value_on']
    value_off = exp_params['value_off']
    
    input_signal__dd = np.zeros([len(time_vec),1])
    for ii in range(len(time_vec)):
        time = time_vec[ii]
        if time < t_rise:
            input_signal__dd[ii] = value_off
        if time >= t_rise and time < t_fall:
            input_signal__dd[ii] = value_off+(value_on-value_off)*(1-np.exp(-(time-t_rise)/tau_rise))
        if time >= t_fall:
            input_signal__dd[ii] = value_off+(value_on-value_off)*(1-np.exp(-(time-t_rise)/tau_rise))*np.exp(-(time-t_fall)/tau_fall)
    
    return input_signal__dd

def dendritic_drive__square_pulse_train(time_vec,sq_pls_trn_params):
    
    input_signal__dd = np.zeros([len(time_vec),1])
    dt = time_vec[1]-time_vec[0]
    t_start = sq_pls_trn_params['t_start']
    t_rise = sq_pls_trn_params['t_rise']
    t_pulse = sq_pls_trn_params['t_pulse']
    t_fall = sq_pls_trn_params['t_fall']
    t_period = sq_pls_trn_params['t_period']
    value_off = sq_pls_trn_params['value_off']
    value_on = sq_pls_trn_params['value_on']
    
    tf_sub = t_rise+t_pulse+t_fall
    time_vec_sub = np.arange(0,tf_sub+dt,dt)
    pwl = [[0,value_off],[t_rise,value_on],[t_rise+t_pulse,value_on],[t_rise+t_pulse+t_fall,value_off]]
    
    pulse = dendritic_drive__piecewise_linear(time_vec_sub,pwl)    
    num_pulses = np.floor((time_vec[-1]-t_start)/t_period).astype(int)        
    ind_start = (np.abs(np.asarray(time_vec)-t_start)).argmin()
    ind_pulse_end = (np.abs(np.asarray(time_vec)-t_start-t_rise-t_pulse-t_fall)).argmin()
    ind_per_end = (np.abs(np.asarray(time_vec)-t_start-t_period)).argmin()
    num_ind_pulse = len(pulse) # ind_pulse_end-ind_start
    num_ind_per = ind_per_end-ind_start
    for ii in range(num_pulses):
        input_signal__dd[ind_start+ii*num_ind_per:ind_start+ii*num_ind_per+num_ind_pulse] = pulse[:]
        
    if t_start+num_pulses*t_period <= time_vec[-1] and t_start+(num_pulses+1)*t_period >= time_vec[-1]:
        ind_final = (np.abs(np.asarray(time_vec)-t_start-num_pulses*t_period)).argmin()
        ind_end = (np.abs(np.asarray(time_vec)-t_start-num_pulses*t_period-t_rise-t_pulse-t_fall)).argmin()
        num_ind = ind_end-ind_final
        input_signal__dd[ind_final:ind_end] = pulse[0:num_ind]
        
    return input_signal__dd

# def dendritic_drive__linear_ramp(time_vec, time_on = 5e-9, slope = 1e-6/1e-9):
    
#     t_on_ind = (np.abs(np.asarray(time_vec)-time_on)).argmin()
#     input_signal__dd = np.zeros([len(time_vec),1])
#     partial_time_vec = time_vec[t_on_ind:]
#     for ii in range(len(partial_time_vec)):
#         time = partial_time_vec[ii]
#         input_signal__dd[t_on_ind+ii] = (time-time_on)*slope
    
#     return input_signal__dd

def dendrite_current_splitting(Ic,Iflux,Ib,M,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev):
    # print('Ic = {}'.format(Ic))
    # print('Iflux = {}'.format(Iflux))
    # print('Ib1 = {}'.format(Ib1))
    # print('Ib2 = {}'.format(Ib2))
    # print('Ib3 = {}'.format(Ib3))
    # print('M = {}'.format(M))
    # print('Lm2 = {}'.format(Lm2))
    # print('Ldr1 = {}'.format(Ldr1))
    # print('Ldr2 = {}'.format(Ldr2))
    # print('L1 = {}'.format(L1))
    # print('L2 = {}'.format(L2))
    # print('L3 = {}'.format(L3))
    # pause(10)
    #see pgs 74, 75 in green lab notebook from 2020_04_01
    
    Ib1 = Ib[0]
    Ib2 = Ib[1]
    Ib3 = Ib[2]
    
    # Lj0 = Ljj(Ic,0)
    Lj2 = Ljj(Ic,Ij2_prev)
    Lj3 = Ljj(Ic,Ij3_prev)
    Ljdr1 = Ljj(Ic,Idr1_prev)
    Ljdr2 = Ljj(Ic,Idr2_prev)
    
    Idr1_next = ( -((-Lj2*(-Ib3*L3*Lj3-Ib2*(L3*Lj3+L2*(L3+Lj3)))
                    +Ib1*(-Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    +L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))))
                    *(-Ldr2-Ljdr2)-Iflux*(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))*M)
                    /((Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3)))
                    *(-Ldr2-Ljdr2)-(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))
                    *(Ldr1+Ljdr1+Lm2)) )
                 
    Idr2_next = ( (Iflux*M)/(Ldr2+Ljdr2)+((Ldr1+Ljdr1+Lm2)
                    *((-Lj2*(-Ib3*L3*Lj3-Ib2*(L3*Lj3+L2*(L3+Lj3)))
                    +Ib1*(-Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    +L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))))
                    *(-Ldr2-Ljdr2)-Iflux*(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    *(Ldr2+Ljdr2))*M))/((-Ldr2-Ljdr2)
                    *((Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3)))
                    *(-Ldr2-Ljdr2)-(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                    -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                    -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))
                    *(Ldr1+Ljdr1+Lm2))) )
                                        
    Ij2_next = ( (1/Lj2)*(-Ib1*L1+(Iflux*(L1+Ldr2+Ljdr2)*M)
                     /(Ldr2+Ljdr2)-(((Ldr2+Ljdr2)*(Ldr1+Ljdr1+Lm2)
                    +L1*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2))
                    *(-(Lj2*(Ib2*L2*L3+Ib3*L3*Lj3+Ib2*(L2+L3)*Lj3)
                    +Ib1*(L1*L3*(L2+Lj2)+L3*Lj2*Lj3
                    +L1*(L2+L3+Lj2)*Lj3+L2*Lj2*(L3+Lj3)))
                    *(Ldr2+Ljdr2)-Iflux*(-L1*(L3*(L2+Lj2)
                    +(L2+L3+Lj2)*Lj3)-Lj2*(L3*Lj3+L2*(L3+Lj3))
                    -(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr2+Ljdr2))*M))
                    /((Ldr2+Ljdr2)*((L1*L3*(L2+Lj2)+L3*Lj2*Lj3
                    +L1*(L2+L3+Lj2)*Lj3+L2*Lj2*(L3+Lj3))
                    *(Ldr2+Ljdr2)-(-L1*(L3*(L2+Lj2)
                    +(L2+L3+Lj2)*Lj3)-Lj2*(L3*Lj3+L2*(L3+Lj3))
                    -(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    *(Ldr2+Ljdr2))*(Ldr1+Ljdr1+Lm2)))) )                                       
                                                                                                 
    Ij3_next = ( (L3*(Ib3*(Lj2*(Ldr2+Ljdr2)*(Ldr1+Ljdr1+Lm2)
                    +L1*(L2+Lj2)*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    +L2*(Lj2*Ljdr1+Lj2*Ljdr2+Ljdr1*Ljdr2
                    +Ldr1*(Ldr2+Lj2+Ljdr2)+(Lj2+Ljdr2)*Lm2
                    +Ldr2*(Lj2+Ljdr1+Lm2)))+Lj2*(Ib2*((Ldr2+Ljdr2)
                    *(Ldr1+Ljdr1+Lm2)+L1*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2))
                    +(Ldr2+Ljdr2)*(Ib1*(Ldr1+Ljdr1+Lm2)+Iflux*M))))
                    /(L3*Ldr1*Ldr2*Lj2+L3*Ldr1*Ldr2*Lj3
                    +L3*Ldr1*Lj2*Lj3+L3*Ldr2*Lj2*Lj3+Ldr1*Ldr2*Lj2*Lj3
                    +L3*Ldr2*Lj2*Ljdr1+L3*Ldr2*Lj3*Ljdr1+L3*Lj2*Lj3*Ljdr1
                    +Ldr2*Lj2*Lj3*Ljdr1+L3*Ldr1*Lj2*Ljdr2+L3*Ldr1*Lj3*Ljdr2
                    +L3*Lj2*Lj3*Ljdr2+Ldr1*Lj2*Lj3*Ljdr2+L3*Lj2*Ljdr1*Ljdr2
                    +L3*Lj3*Ljdr1*Ljdr2+Lj2*Lj3*Ljdr1*Ljdr2+(Lj2*Lj3*(Ldr2+Ljdr2)
                    +L3*(Lj2*Lj3+Ldr2*(Lj2+Lj3)+(Lj2+Lj3)*Ljdr2))*Lm2
                    +L1*(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    +L2*(L3+Lj3)*(Lj2*Ljdr1+Lj2*Ljdr2+Ljdr1*Ljdr2
                    +Ldr1*(Ldr2+Lj2+Ljdr2)+(Lj2+Ljdr2)*Lm2+Ldr2*(Lj2+Ljdr1+Lm2))) )
    
    I1_next = ( Ib1-(Iflux*M)/(Ldr2+Ljdr2)+((Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    *(-(Lj2*(Ib2*L2*L3+Ib3*L3*Lj3+Ib2*(L2+L3)*Lj3)
                    +Ib1*(L1*L3*(L2+Lj2)+L3*Lj2*Lj3+L1*(L2+L3+Lj2)
                    *Lj3+L2*Lj2*(L3+Lj3)))*(Ldr2+Ljdr2)
                    -Iflux*(-L1*(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    -Lj2*(L3*Lj3+L2*(L3+Lj3))-(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    *(Ldr2+Ljdr2))*M))/((Ldr2+Ljdr2)*((L1*L3*(L2+Lj2)
                    +L3*Lj2*Lj3+L1*(L2+L3+Lj2)*Lj3+L2*Lj2*(L3+Lj3))
                    *(Ldr2+Ljdr2)-(-L1*(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    -Lj2*(L3*Lj3+L2*(L3+Lj3))-(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)
                    *(Ldr2+Ljdr2))*(Ldr1+Ljdr1+Lm2))) )
                                                       
    I2_next = ( Ib1+Ib2+(Ib1*L1)/Lj2-(Iflux*(L1+Ldr2+Lj2+Ljdr2)*M)
                    /(Lj2*(Ldr2+Ljdr2))+((L1+Lj2+((L1+Ldr2+Lj2+Ljdr2)*
                    (Ldr1 + Ljdr1 + Lm2))/(Ldr2+Ljdr2))
                    *(-(Lj2*(Ib2*L2*L3+Ib3*L3*Lj3+Ib2*(L2+L3)*Lj3)
                    +Ib1*(L1*L3*(L2+Lj2)+L3*Lj2*Lj3
                    +L1*(L2+L3+Lj2)*Lj3+L2*Lj2*(L3+Lj3)))
                    *(Ldr2+Ljdr2)-Iflux*(-L1*(L3*(L2+Lj2)
                    +(L2+L3+Lj2)*Lj3)-Lj2*(L3*Lj3+L2*(L3+Lj3))
                    -(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr2+Ljdr2))*M))
                    /(Lj2*((L1*L3*(L2+Lj2)+L3*Lj2*Lj3+L1*(L2+L3+Lj2)*Lj3
                    +L2*Lj2*(L3+Lj3))*(Ldr2+Ljdr2)-(-L1*(L3*(L2+Lj2)
                    +(L2+L3+Lj2)*Lj3)-Lj2*(L3*Lj3+L2*(L3+Lj3))
                    -(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr2+Ljdr2))*(Ldr1+Ljdr1+Lm2))) )
                                                         
    I3_next = ( (Lj3*(Ib3*(Lj2*(Ldr2+Ljdr2)*(Ldr1+Ljdr1+Lm2)
                    +L1*(L2+Lj2)*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    +L2*(Lj2*Ljdr1+Lj2*Ljdr2+Ljdr1*Ljdr2
                    +Ldr1*(Ldr2+Lj2+Ljdr2)+(Lj2+Ljdr2)*Lm2
                    +Ldr2*(Lj2+Ljdr1+Lm2)))+Lj2*(Ib2*((Ldr2+Ljdr2)*(Ldr1+Ljdr1+Lm2)
                    +L1*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2))+(Ldr2+Ljdr2)*(Ib1*(Ldr1+Ljdr1+Lm2)
                    +Iflux*M))))/(L3*Ldr1*Ldr2*Lj2+L3*Ldr1*Ldr2*Lj3+L3*Ldr1*Lj2*Lj3
                    +L3*Ldr2*Lj2*Lj3+Ldr1*Ldr2*Lj2*Lj3
                    +L3*Ldr2*Lj2*Ljdr1+L3*Ldr2*Lj3*Ljdr1
                    +L3*Lj2*Lj3*Ljdr1+Ldr2*Lj2*Lj3*Ljdr1
                    +L3*Ldr1*Lj2*Ljdr2+L3*Ldr1*Lj3*Ljdr2
                    +L3*Lj2*Lj3*Ljdr2+Ldr1*Lj2*Lj3*Ljdr2
                    +L3*Lj2*Ljdr1*Ljdr2+L3*Lj3*Ljdr1*Ljdr2
                    +Lj2*Lj3*Ljdr1*Ljdr2+(Lj2*Lj3*(Ldr2+Ljdr2)
                    +L3*(Lj2*Lj3+Ldr2*(Lj2+Lj3)+(Lj2+Lj3)*Ljdr2))*Lm2
                    +L1*(L3*(L2+Lj2)+(L2+L3+Lj2)*Lj3)*(Ldr1+Ldr2+Ljdr1+Ljdr2+Lm2)
                    +L2*(L3+Lj3)*(Lj2*Ljdr1+Lj2*Ljdr2+Ljdr1*Ljdr2
                    +Ldr1*(Ldr2+Lj2+Ljdr2)+(Lj2+Ljdr2)*Lm2
                    +Ldr2*(Lj2+Ljdr1+Lm2))) )
                                                
    return Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1_next, I2_next, I3_next

def dendrite_current_splitting__old(Ic,Iflux,Ib1,Ib2,Ib3,M,Lm2,Ldr1,Ldr2,L1,L2,L3):
    # print('Ic = {}'.format(Ic))
    # print('Iflux = {}'.format(Iflux))
    # print('Ib1 = {}'.format(Ib1))
    # print('Ib2 = {}'.format(Ib2))
    # print('Ib3 = {}'.format(Ib3))
    # print('M = {}'.format(M))
    # print('Lm2 = {}'.format(Lm2))
    # print('Ldr1 = {}'.format(Ldr1))
    # print('Ldr2 = {}'.format(Ldr2))
    # print('L1 = {}'.format(L1))
    # print('L2 = {}'.format(L2))
    # print('L3 = {}'.format(L3))
    # pause(10)
    #see pgs 74, 75 in green lab notebook from 2020_04_01
    
    Lj0 = Ljj(Ic,0)
    Lj2 = Ljj(Ic,Ib2)#approximation; current passing through jj2 is not exactly Ib2
    Lj3 = Ljj(Ic,Ib3)#approximation; current passing through jj3 is not exactly Ib3
    
    #initial approximations
    Idr2_prev = ((Lm2+Ldr1+Lj0)*Ib1+M*Iflux)/( Lm2+Ldr1+Ldr2+2*Lj0 + (Lm2+Ldr1+Lj0)*(Ldr2+Lj0)/L1 )
    Idr1_prev = Ib1-( 1 + (Ldr2+Lj0)/L1 )*Idr2_prev
    # I1_prev = Ib1-Idr1_prev-Idr2_prev
    
    Idr1_next = Ib1/2
    Idr2_next = Ib1/2
    num_it = 1
    while abs((Idr2_next-Idr2_prev)/Idr2_next) > 1e-4:
        
        # print('num_it = {:d}'.format(num_it))
        num_it += 1
        
        Idr1_prev = Idr1_next
        Idr2_prev = Idr2_next
        
        Ljdr1 = Ljj(Ic,Idr1_prev)
        Ljdr2 = Ljj(Ic,Idr2_prev)
        
        Idr1_next = ( -((-Lj2*(-Ib3*L3*Lj3-Ib2*(L3*Lj3+L2*(L3+Lj3)))
                        +Ib1*(-Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        +L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))))
                        *(-Ldr2-Ljdr2)-Iflux*(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))*M)
                        /((Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3)))
                        *(-Ldr2-Ljdr2)-(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))
                        *(Ldr1+Ljdr1+Lm2)) )
                     
        Idr2_next = ( (Iflux*M)/(Ldr2+Ljdr2)+((Ldr1+Ljdr1+Lm2)
                        *((-Lj2*(-Ib3*L3*Lj3-Ib2*(L3*Lj3+L2*(L3+Lj3)))
                        +Ib1*(-Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        +L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))))
                        *(-Ldr2-Ljdr2)-Iflux*(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        *(Ldr2+Ljdr2))*M))/((-Ldr2-Ljdr2)
                        *((Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3)))
                        *(-Ldr2-Ljdr2)-(Lj2*(-L3*Lj3-L2*(L3+Lj3))
                        -L1*(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))
                        -(L3*Lj3+L2*(L3+Lj3)+Lj2*(L3+Lj3))*(Ldr2+Ljdr2))
                        *(Ldr1+Ljdr1+Lm2))) )
                                            
        if num_it > 10:
            # print('dendrite_current_splitting _ num_it > 10 _ convergence unlikely\nIdr2_prev = {}, Idr2_next = {}\n\n'.format(Idr2_prev,Idr2_next))
            break
                                            
    Idr = Idr2_next
    
    return Idr

def Ljj(critical_current,current):
    
    norm_current = np.max([np.min([current/critical_current,1]),1e-9])
    L = (3.2910596281416393e-16/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

def I_di_sat_of_I_dr_2(Idr2):
    
    if Idr2 < 40e-6:
        I_di_sat = 10e-6
    if Idr2 >= 40e-6 and Idr2 <= 44.7416e-6:
        I_di_sat = 1.77837*Idr2-61.34579e-6
    if Idr2 > 44.7416e-6:
        I_di_sat = 18.0660e-6
    
    return I_di_sat

def amp_fitter(time_vec,I_di):
    
    time_pts = [10,15,20,25,30,35,40,45,50]
    target_vals = [22.7608,37.0452,51.0996,65.3609,79.6264,93.8984,107.963,122.233,136.493]
    actual_vals = np.zeros([len(time_pts),1])
    
    for ii in range(len(time_pts)):
        ind = (np.abs(time_vec*1e9-time_pts[ii])).argmin()
        actual_vals[ii] = I_di[ind]
    
    error = 0
    for ii in range(len(time_pts)):
        error += abs( target_vals[ii]-actual_vals[ii]*1e9 )**2
    
    return error

def mu_fitter(data_dict,time_vec,I_di,mu1,mu2,amp):
    
    time_vec_spice = data_dict['time']
    target_vec = data_dict['L9#branch']

    # fig, ax = plt.subplots(nrows = 1, ncols = 1)   
    # fig.suptitle('Comparing WR and soen_sim')
    # plt.title('amp = {}; mu1 = {}; mu2 = {}'.format(amp,mu1,mu2))
    
    # ax.plot(time_vec_spice*1e9,target_vec*1e9,'o-', label = 'WR')        
    # ax.plot(time_vec*1e9,I_di*1e9,'o-', label = 'soen_sim')    
    # ax.legend()
    # ax.set_xlabel(r'Time [ns]')
    # ax.set_ylabel(r'$I_{di} [nA]$') 
    
    error = 0
    norm = 0
    for ii in range(len(time_vec)):
        ind = (np.abs(time_vec_spice-time_vec[ii])).argmin()
        error += abs( target_vec[ind]-I_di[ii] )**2
        norm += abs( target_vec[ind] )**2
    
    error = error/norm
    
    return error

def mu_fitter_3_4(data_dict,time_vec,I_di,mu3,mu4):
    
    time_vec_spice = data_dict['time']
    target_vec = data_dict['L9#branch']

    # fig, ax = plt.subplots(nrows = 1, ncols = 1)   
    # fig.suptitle('Comparing WR and soen_sim')
    # plt.title('amp = {}; mu1 = {}; mu2 = {}'.format(amp,mu1,mu2))
    
    # ax.plot(time_vec_spice*1e9,target_vec*1e9,'o-', label = 'WR')        
    # ax.plot(time_vec*1e9,I_di*1e9,'o-', label = 'soen_sim')    
    # ax.legend()
    # ax.set_xlabel(r'Time [ns]')
    # ax.set_ylabel(r'$I_{di} [nA]$') 
    
    error = 0
    norm = 0
    for ii in range(len(time_vec)):
        ind = (np.abs(time_vec_spice-time_vec[ii])).argmin()
        error += abs( target_vec[ind]-I_di[ii] )**2
        norm += abs( target_vec[ind] )**2
    
    error = error/norm
    
    return error

def chi_squared_error(target_data,actual_data):
    
    # print('calculating chi^2 ...')
    error = 0
    norm = 0
    for ii in range(len(actual_data[0,0:-1])):
        dt = actual_data[0,ii+1]-actual_data[0,ii]
        # print('ii = {} of {}'.format(ii+1,len(actual_data[0,:])))
        ind = (np.abs(target_data[0,:]-actual_data[0,ii])).argmin()        
        error += dt*abs( target_data[1,ind]-actual_data[1,ii] )**2
        norm += dt*abs( target_data[1,ind] )**2
    
    error = error/norm    
    
    return error

def read_wr_data(file_path):
    
    print('reading wr data file ...')
    f = open(file_path, 'rt')
    
    file_lines = f.readlines()
    
    counter = 0
    for line in file_lines:
        counter += 1
        if line.find('No. Variables:') != -1:
            ind_start = line.find('No. Variables:')
            num_vars = int(line[ind_start+15:])
        if line.find('No. Points:') != -1:
            ind_start = line.find('No. Points:')
            num_pts = int(line[ind_start+11:])
        if str(line) == 'Variables:\n':            
            break    

    var_list = []
    for jj in range(num_vars):
        if jj <= 9:
            var_list.append(file_lines[counter+jj][3:-3]) 
        if jj > 9:
            var_list.append(file_lines[counter+jj][4:-3]) 

    data_mat = np.zeros([num_pts,num_vars])
    tn = counter+num_vars+1
    for ii in range(num_pts):
        # print('\n\nii = {}\n'.format(ii))
        for jj in range(num_vars):
            ind_start = file_lines[tn+jj].find('\t')
            # print('tn+jj = {}'.format(tn+jj))
            data_mat[ii,jj] = float(file_lines[tn+jj][ind_start+1:])
            # print('data_mat[ii,jj] = {}'.format(data_mat[ii,jj]))
        tn += num_vars
    
    f.close
    
    data_dict = dict()
    for ii in range(num_vars):
        data_dict[var_list[ii]] = data_mat[:,ii]
        
    print('done reading wr data file.')
    
    return data_dict

def omega_LRC(L,R,C):
    
    omega_r = np.sqrt( (L*C)**(-1) - 0.25*(R/L)**(2) )
    omega_i = R/(2*L)
    
    return omega_r, omega_i  

def load_neuron_data(load_string):
        
    with open('data/'+load_string, 'rb') as data_file:         
        neuron_imported = pickle.load(data_file)
    
    return neuron_imported
    
def save_session_data(data_array = [],save_string = 'soen_sim'):
    
    tt = time.time()     
    s_str = 'session_data__'+save_string+'__'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(tt))+'.dat'
    with open('soen_sim_data/'+s_str, 'wb') as data_file:
            pickle.dump(data_array, data_file)
            
    return

def load_session_data(load_string):
        
    with open('soen_sim_data/'+load_string, 'rb') as data_file:         
        data_array_imported = pickle.load(data_file)

    return data_array_imported

def t_fq(I,Ic,R,mu1,mu2):
    
    p = physical_constants()
    t_fq_vec = (p['Phi0']/(Ic*R))*((I/Ic)**mu1-1)**(-mu2)
    
    return t_fq_vec

def V_fq(I,Ic,R,mu1,mu2):
    
    V_fq_vec = (Ic*R)*((I/Ic)**mu1-1)**(mu2)
    
    return V_fq_vec

def V_fq__fit(I,mu1,mu2,V0):
    
    Ic = 40e-6
    R = 4.125
    
    V_fq_vec = (Ic*R)*((I/Ic)**mu1-1)**(mu2)+V0
    
    return V_fq_vec

def inter_fluxon_interval__fit(I,mu1,mu2,V0):
    
    Ic = 40e-6
    R = 4.125
    
    V_fq_vec = (Ic*R)*((I/Ic)**mu1-1)**(mu2)+V0
    p = physical_constants()
    ifi_vec = p['Phi0']/V_fq_vec
    
    return ifi_vec

def inter_fluxon_interval__fit_2(I_di,t0,I_fluxon,mu1,mu2,V0):
    
    Ic = 40e-6
    R = 4.125
    Lj2 = Ljj(Ic,Ic)
    Lj3 = Lj2
    L2 = 77.5e-12
    I0 = 35.2699e-6
    Phi0 = 2.06783375e-15

    t_fq = np.zeros([len(I_di)])
    for ii in range(len(I_di)):
        I_loop2_from_di = (Lj3/(L2+Lj2))*I_di[ii]
        if I0+I_fluxon+I_loop2_from_di-I_di[ii] > Ic:
            t_fq[ii] = t0 + Phi0 * ( (Ic*R)*(((I0+I_fluxon+I_loop2_from_di-I_di[ii])/Ic)**mu1-1)**(mu2)+V0 )**(-1)
        else:
            t_fq[ii] = 1e-6
    
    return t_fq

def inter_fluxon_interval__fit_3(I_di,I_bar_1,I_bar_2):
    
    Ic = 40e-6
    R = 4.125
    Phi0 = 2.06783375e-15
    V0 = 105e-6
    mu1 = 2.8
    mu2 = 0.5

    t_1 = Phi0/((Ic*R)*((I_bar_1/Ic)**mu1-1)**mu2+V0)
    t_2 = np.zeros([len(I_di)])
    for ii in range(len(I_di)):
        if I_bar_2-I_di[ii] > Ic:
            t_2[ii] = Phi0/((Ic*R)*(((I_bar_2-I_di[ii])/Ic)**mu1-1)**mu2+V0)
        else:
            t_2[ii] = 1
    t_fq = t_1+t_2
    
    return t_fq

def inter_fluxon_interval(I):
    
    V_fq_vec = (40e-6*4.125)*((I/40e-6)**2.839-1)**(0.501)+103.047e-6    
    ifi_vec = 2.06783375e-15/V_fq_vec
    
    return ifi_vec