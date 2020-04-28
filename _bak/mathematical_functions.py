import numpy as np

def synaptic_time_stepper(time_vec,present_time_index,input_spike_times,I_0,I_si_sat,gamma1,gamma2,gamma3,tau_rise,tau_fall):
    
    _pti = present_time_index
    _pt = time_vec[_pti] #present time
    I_si = 0
    if len(input_spike_times) > 0:
        idx_start = (np.abs(_pt-5*tau_fall-input_spike_times)).argmin()
        for ii in range(len(input_spike_times[idx_start:])):
            if _pt > input_spike_times[ii]:
                if _pt-input_spike_times[ii] <= tau_rise:
                    I_si += synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si,tau_rise,tau_fall)*\
                        ( (1/tau_rise**gamma3)*(_pt - input_spike_times[ii])**gamma3 )*\
                            np.exp(tau_rise/tau_fall)*np.exp(-(_pt-input_spike_times[ii])/tau_fall)
                else:
                    I_si += synaptic_response_prefactor(I_0,I_si_sat,gamma1,gamma2,I_si,tau_rise,tau_fall)*\
                        np.exp(tau_rise/tau_fall)*np.exp(-(_pt-input_spike_times[ii])/tau_fall)
                    
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