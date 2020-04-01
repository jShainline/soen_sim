import numpy as np
import pickle

from _plotting import plot_dendritic_drive

def synaptic_time_stepper(time_vec,present_time_index,input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall):
    
    _pti = present_time_index
    _pt = time_vec[_pti] #present time
    I_si = 0
    if len(input_spike_times) > 0:
        num_tau_retain = 5
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
                    
    return I_si*1e-6

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

def dendritic_time_stepper(time_vec,A_prefactor,I_drive,I_b,I_th,I_di_sat,tau_di,mu_1,mu_2,mu_3,mu_4):
    
    I_di_vec = np.zeros([len(time_vec),1])
    for ii in range(len(time_vec)-1):
        dt = time_vec[ii+1]-time_vec[ii]
        if I_drive[ii+1]+I_b > I_th:
            factor_1 = ( ( ((I_drive[ii+1]+I_b)/I_th)**mu_1-1)**mu_2 )
        else:
            factor_1 = 0
        factor_2 = ( 1-(I_di_vec[ii]/I_di_sat)**mu_3 )**mu_4
        I_di_vec[ii+1] = dt * A_prefactor * factor_1 * factor_2 + (1-dt/tau_di)*I_di_vec[ii]        
    
    return I_di_vec

def dendritic_drive__step_function(time_vec, amplitude = 5e-6, time_on = 5e-9):
    
    t_on_ind = (np.abs(np.asarray(time_vec)-time_on)).argmin()
    input_signal__dd = np.zeros([len(time_vec),1])
    input_signal__dd[t_on_ind:] = amplitude
    
    return input_signal__dd

def dendrite_current_splitting(Ic,Iflux,Ib1,Ib2,Ib3,M,Lm2,Ldr1,Ldr2,L1,L2,L3):
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
    while abs((Idr2_next-Idr2_prev)/Idr2_next) > 1e-5:
        
        print('num_it = {:d}'.format(num_it))
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
            print('dendrite_current_splitting _ num_it > 10 _ convergence unlikely')
            break
                                            
    Idr = Idr2_next
    
    return Idr

def Ljj(critical_current,current):
    
    norm_current = np.max([np.min([current/critical_current,1]),1e-9])
    L = (3.2910596281416393e-16/critical_current)*np.arcsin(norm_current)/(norm_current)
    
    return L

def physical_constants():

    p = dict(h = 6.62606957e-34,#Planck's constant in kg m^2/s
         hBar = 6.62606957e-34/2/np.pi,
         c = 299792458,#speed of light in meters per second
         epsilon0 = 8.854187817e-12,#permittivity of free space in farads per meter
         mu0 = 4*np.pi*1e-7,#permeability of free space in volt seconds per amp meter
         kB = 1.3806e-23,#Boltzmann's constant
         eE = 1.60217657e-19,#electron charge in coulombs
         mE = 9.10938291e-31,#mass of electron in kg
         eV = 1.60217657e-19,#joules per eV
         Ry = 9.10938291e-31*1.60217657e-19**4/(8*8.854187817e-12**2*(6.62606957e-34/2/np.pi)**3*299792458),#13.3*eV;#Rydberg in joules
         a0 = 4*np.pi*8.854187817e-12*(6.62606957e-34/2/np.pi)**2/(9.10938291e-31*1.60217657e-19**2),#estimate of Bohr radius
         Phi0 = 6.62606957e-34/(2*1.60217657e-19)#flux quantum
         )

    return p   

def load_neuron_data(load_string):
        
        with open('data/'+load_string, 'rb') as data_file: 
            
            neuron_imported = pickle.load(data_file)
    
        return neuron_imported