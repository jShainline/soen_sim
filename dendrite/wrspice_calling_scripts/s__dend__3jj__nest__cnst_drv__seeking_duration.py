import numpy as np
import time

from _util import physical_constants
p = physical_constants()
Phi0 = p['Phi0__pH_ns']

import WRSpice

#%%

# tempCirFile only netlist
# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
# VERY IMPORTTANT Leave a blank line in the start and end of the  cir file

#%%
# dendritic series bias current
Ib = 85 # uA

# inductor in plasticity loop
Lp = 35 # pH

# dendritic plasticity bias current
Mp = np.sqrt(200*Lp)

# excitatory flux input

num_steps = 100 # 25 # for excitatory flux

max_flux = p['Phi0__pH_ns']/2
_fr = max_flux/num_steps # flux_resolution
_fr_0 = max_flux/200 # flux_resolution of first onset point

Mex = np.sqrt(400*12.5)
Iex_max = max_flux/Mex

# plasticity flux biases
Phi_p_list = [-1033.916875,-982.221031,-930.525188,-878.829344,-827.133500,-775.437656,-723.741813,-672.045969,-620.350125,-568.654281,-516.958438,-465.262594,-413.566750,-361.870906,-310.175063,-258.479219,-206.783375,-155.087531,-103.391688,-51.695844,0.000000,51.695844,103.391688,155.087531,206.783375,258.479219,310.175063,361.870906,413.566750,465.262594,516.958438,568.654281,620.350125,672.045969,723.741813,775.437656,827.133500,878.829344,930.525188,982.221031,1033.916875]

# excitatory flux 
Phi_ex_on_list = [779.780125,850.603432,909.536695,947.584837,956.373131,949.342496,936.625318,920.702998,902.919627,883.998948,863.940960,842.952447,821.550367,799.321154,776.678374,753.622027,730.048722,706.268633,681.764803,657.157581,632.033400,606.392261,580.440947,553.972674,526.987443,499.485254,471.362714,442.516432,412.843018,382.342469,350.601220,317.412488,282.155922,244.211171,201.717187,150.434909,0.000000,0.000000,0.000000,0.000000,0.000000]
# positive current with Mex positive


#%%
rate = WRSpice.WRSpice(tempCirFile = 'dend__3jj__nest__cnst_drv', OutDatFile = 'dend_3jj_nest_cnst_drv_seek_dur_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'v(2)' # list all vectors you want to save
rate.pathCir = ''


FileFormat = 'Ib{:06.2f}uA_Lp{:05.2f}pH_Ip{:09.6f}uA_Iex{:09.6f}'

for ii in range(len(Phi_p_list)):
    Phi_ex_on = Phi_ex_on_list[ii]
    Phi_ex_vec = np.append( np.insert( np.arange(Phi_ex_on,max_flux,_fr) ,0,Phi_ex_on-_fr_0 ) , max_flux ) 
    # print('Phi_a_vec = {}'.format(Phi_a_vec))
    # time.sleep(1)
    
    for jj in range(len(Phi_ex_vec)):
    
        print('ii = {} of {}; jj = {} of {}'.format(ii+1,len(Phi_p_list),jj+1,len(Phi_ex_vec)))
            
        Iex = Phi_ex_vec[jj]/Mex
        Ip = Phi_p_list[ii]/Mp
        FilePrefix = FileFormat.format(Ib,Lp,Ip,Iex)
        rate.FilePrefix = FilePrefix
                    
        Params = {          
        'Ip':'{:9.6f}u'.format(np.round(Ip,6)),
        'Ib':'{:6.2f}u'.format(np.round(Ib,2)),
        'Iex':'{:9.6f}u'.format(np.round(Iex,6)),
        'Lp':'{:05.2f}p'.format(Lp)
        }
    
        rate.Params = Params
        rate.stepTran = '10p'
        rate.stopTran = '100n'
        rate.doAll()
