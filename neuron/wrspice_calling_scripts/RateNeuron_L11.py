import WRSpice
from matplotlib import pyplot as plt
import numpy as np
import os

# tempCirFile only netlist
# export your netlist to a cir file using 'deck' command in xic
# open the cir file and remove all lines except the netlist lines
# replace all values by Params, you want to sweep, with white space around
# Component names don't need any changes
rate=WRSpice.WRSpice(tempCirFile='RateNeuron',OutDatFile='RateNeuron') # no file extentions
rate.pathWRSSpice='/raid/home/local/xictools/wrspice.current/bin/wrspice'  # for running on grumpy
#rate.pathWRSSpice='C:/usr/local/xictools/wrspice/bin/wrspice.bat' # for local computer

rate.save='v(6) i(l3) i(l13) i(l11) i(l12)' # list all vectors you want to save
f_vector1=np.arange(1,10,1)
f_vector2=np.arange(10,100,10)
f_vector=np.concatenate((f_vector1,f_vector2),axis=0)

L11_vector=np.arange(40,50,1)
for j in np.arange(len(L11_vector)):
    FilePrefix='L11='+str(int(L11_vector[j]))+'p_Threshold=65u_'
    os.makedirs(FilePrefix, exist_ok=True)  
    rate.pathCir=FilePrefix
    for i in np.arange(len(f_vector)):
        print  (FilePrefix+str(f_vector[i])+'MHz.cir')
        rate.FilePrefix=FilePrefix+str(f_vector[i])+'MHz'
        Params= {
          "L11": str(int(L11_vector[j]))+'p',
          "iThreshold":"65u",
          "Period": '1/'+ str(int(f_vector[i]))+'e6',
        }
        rate.Params=Params
        rate.stepTran='1p'
        rate.stopTran='3000n'
        rate.doAll()
    

