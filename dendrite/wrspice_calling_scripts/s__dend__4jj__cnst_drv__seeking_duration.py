import numpy as np
import time

from _util import physical_constants
p = physical_constants()

import WRSpice

#%%

# tempCirFile only netlist
# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
# VERY IMPORTTANT Leave a blank line in the start and end of the  cir file

#%%
Ma = np.sqrt(400*12.5) # pH
Mp = 77.5 # Ph

num_steps = 20

max_flux = p['Phi0__pH_ns']/2
_fr = max_flux/num_steps # flux_resolution

# dendritic series bias current
dIb = 2
Ib_vec = np.arange(130,134+dIb,dIb) # uA

# plasticity flux biases
Phi_p_list = [-1033.916875,-930.525188,-827.133500,-723.741813,-620.350125,-516.958438,-413.566750,-310.175063,-206.783375,-103.391688,0.000000,103.391688,206.783375,310.175063,413.566750,516.958438,620.350125,723.741813,827.133500,930.525188,1033.916875]
                    

# activity flux 
Phi_a_on_neg__array = [[-651.597909,-632.369413,-612.727401,-592.775251,-572.512965,-551.940542,-531.057982,-509.865284,-488.362450,-466.549478,-444.426370,-421.889746,-398.939605,-375.472569,-351.488639,-326.884434,-301.659955,-275.505065,-248.213006,-219.577020,-189.080212],[-629.474800,-610.453062,-590.914429,-570.962280,-550.596615,-530.024191,-509.038252,-487.638797,-466.032583,-443.909475,-421.579609,-398.629468,-375.369190,-351.488639,-326.987813,-301.763334,-275.711823,-248.523143,-219.990536,-189.597107,-156.412444],[-607.041555,-588.226575,-568.687942,-548.839172,-528.473506,-507.694325,-486.605007,-465.102172,-443.289201,-420.959334,-398.215952,-375.059053,-351.281881,-326.884434,-301.763334,-275.815202,-248.729901,-220.300673,-189.907244,-156.825960,-119.196000]]
Phi_a_on_pos__array = [[481.539435,631.232244,727.684861,725.513902,707.319196,686.333257,663.796632,640.329597,616.035529,591.121187,565.483192,539.328303,512.553139,485.157701,457.038609,428.092486,398.319331,367.615765,335.568271,301.866713,266.097575],[386.534124,546.771591,669.379099,695.740747,679.406863,658.937819,636.607952,613.140917,588.743470,563.622370,537.880996,511.415969,484.227289,456.314956,427.678970,398.112573,367.512386,335.568271,302.073471,266.407712,227.743962],[287.186894,456.935230,597.220549,663.176358,650.770877,631.128865,609.005756,585.642100,561.141274,535.813416,509.761905,482.986741,455.384545,426.851938,397.492299,367.098870,335.361513,302.073471,266.614470,228.054099,184.634914]]
# neg has negative current with Ma positive


#%%
rate = WRSpice.WRSpice(tempCirFile = 'dend__4jj__one_bias__plasticity__mid_betaL__cnst_drv', OutDatFile = 'dend_4jj_one_bias_plstc_cnst_drv_seek_dur_') # no file extentions
rate.pathWRSSpice = '/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
# rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save = 'v(4)' # list all vectors you want to save
rate.pathCir = ''


FileFormat = 'Ib{:06.2f}uA_Ip{:09.6f}uA_Ia{:09.6f}'

for qq in range(len(Ib_vec)):
    Ib = Ib_vec[qq]
    for ii in range(len(Phi_p_list)):
        Phi_a_on_neg = Phi_a_on_neg__array[qq][ii]
        Phi_a_on_pos = Phi_a_on_pos__array[qq][ii]
        Phi_a_vec = np.concatenate( ( np.append(np.arange(-max_flux,Phi_a_on_neg,_fr),np.asarray([Phi_a_on_neg,Phi_a_on_neg+_fr])) , np.append(np.insert(np.arange(Phi_a_on_pos,max_flux,_fr),0,Phi_a_on_pos-_fr),max_flux) )  )
        # print('Phi_a_vec = {}'.format(Phi_a_vec))
        # time.sleep(1)
        
        for jj in range(len(Phi_a_vec)):
        
            print('qq = {} of {}; ii = {} of {}; jj = {} of {}'.format(qq+1,len(Ib_vec),ii+1,len(Phi_p_list),jj+1,len(Phi_a_vec)))
                
            Ia = Phi_a_vec[jj]/Ma
            Ip = Phi_p_list[ii]/Mp
            FilePrefix = FileFormat.format(Ib,Ip,Ia)
            rate.FilePrefix = FilePrefix
                        
            Params = {          
            'Ip':'{:9.6f}u'.format(np.round(Ip,6)),
            'Ib':'{:6.2f}u'.format(np.round(Ib,2)),
            'Ia':'{:9.6f}u'.format(Ia)
            }
        
            rate.Params = Params
            rate.stepTran = '10p'
            rate.stopTran = '500n'
            rate.doAll()
