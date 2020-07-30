from matplotlib import pyplot as plt
import matplotlib as mp
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.signal import find_peaks
from IPython import get_ipython
import csv
from os import path
import glob
get_ipython().run_line_magic('matplotlib', 'qt')
from _functions import * 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider




FileFromat="master__dnd__rate_array__Llft{Lleft:05.2f}_Lrgt{LRight:05.2f}_Ide{I_de:05.2f}_d200"
plt.close('all')
JumpOnEmptyData=True
fig = plt.figure(figsize=[15, 10])
# fig.set

axL = plt.axes([0.25, 0.1, 0.65, 0.03])
axI_de = plt.axes([0.25, 0.05, 0.65, 0.03])

sL = Slider(axL, 'L_left', 7, 13, valinit=7,valfmt ='%0.0f')
sI_de = Slider(axI_de, 'I_de', 70, 80, valinit=70,valfmt ='%0.0f')

# border, can be set independently of all other quantities
left = 0.05; right=1-left
bottom=0.2; top=1-bottom
# wspace (=average relative space between subplots)
wspace = 0.1
# ax = fig.add_subplot( projection='3d')
# plt.subplots_adjust(0.1,0.5,0.5,0.6)
ax=plt.axes([left, bottom, right, top],projection='3d')
# fig.subplots_adjust(ax,top=1)
plt.sca(ax)

global L_left, I_de, prev_sL,prev_sI_de
L_left=7
I_de=70
prev_sL=L_left
prev_sI_de=I_de
def updatePlot(vald):
    global L_left
    global I_de
    
    plt.cla()
    maxRate=0
    maxI_de=0
    minIdrive=1e20
    prev_sL=L_left
    prev_sI_de=I_de
    L_left=int(sL.val)
    L_Right=30-L_left
    I_de=int(sI_de.val)
    
    FilePrefix=FileFromat.format(Lleft=L_left,LRight=L_Right,I_de=I_de)
    print(FilePrefix)
    Files=glob.glob(FilePrefix+"*")
    print(Files)
    if len(Files)>0:
        data_array=load_session_data(Files[0])
       
        master_rate_array=data_array['rate_array'] 
        I_drive_list= data_array['I_drive_list'] 
        I_di_array__scaled=data_array['I_di_array'] 
        if len(master_rate_array)>0:
            num_drives=len(I_drive_list)
            cmap = mp.cm.get_cmap('jet')
            for ii in range(num_drives):
            #    ax.plot(I_di_array__scaled[ii][:],master_rate_array[ii][:]*1e-3, '-', label = 'I_drive = {}'.format(I_drive_list[ii]))   
                    X3=I_di_array__scaled[ii][:]
                    if np.max(X3)>maxI_de:
                        maxI_de=np.max(X3)
                    Z3=I_drive_list[ii]
                    if Z3<minIdrive:
                        minIdrive=Z3
                    Y3=master_rate_array[ii][:]*1e-3
                    if np.max(Y3)>maxRate:
                        maxRate=np.max(Y3)
                    verts = [(X3[i],Y3[i]-0.5) for i in range(len(X3))] 
                    ax.add_collection3d(PolyCollection([verts],color=cmap(1-ii/num_drives),alpha=0.3),zs=Z3, zdir='y')
                    ax.plot(X3,Y3,Z3,linewidth=4, color=cmap(1-ii/num_drives), zdir='y',alpha=1)
            
                    
            ax.set_xticks([0, 10, 20])
            for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(24)
            for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(24)
            for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(24)
            
            ax.set_xlabel(r'$I_{di}$ [$\mu$A]',fontsize=24, fontweight='bold', labelpad=30) ; ax.set_xlim3d(0,maxI_de)
            
            ax.set_ylabel('Idrive [$\mu$A]',fontsize=24, fontweight='bold', labelpad=30) ; ax.set_ylim3d(minIdrive,30)
            # ax.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax.set_zlabel(r'$r_{j_{di}}$ [kilofluxons per $\mu$s]',fontsize=24, fontweight='bold', labelpad=10) ; ax.set_zlim3d(0,maxRate)

            
            ax.view_init(45, 30)
                
            plt.sca(ax)
            plt.draw()
        else:
            if JumpOnEmptyData==True:
                if prev_sI_de>I_de:
                    prev_sL=L_left
                    if I_de>70:
                        sI_de.set_val(I_de-1)
                else:
                    prev_sL=L_left
                    if I_de<80:
                        sI_de.set_val(I_de+1)
                    
                sI_de.on_changed(updatePlot)
                if prev_sL>L_left:
                    prev_sI_de=I_de
                    if L_left>7:
                        sL.set_val(L_left-1)
                else:
                    prev_sI_de=I_de
                    if L_left<14:
                        sL.set_val(L_left+1)
                sL.on_changed(updatePlot)
 
    else:
        if JumpOnEmptyData==True:
            if prev_sI_de>I_de:
                prev_sL=L_left
                if I_de>70:
                    sI_de.set_val(I_de-1)
            else:
                prev_sL=L_left
                if I_de<80:
                    sI_de.set_val(I_de+1)
            sI_de.on_changed(updatePlot)
            if prev_sL>L_left:
                prev_sI_de=I_de
                if L_left>7:
                    sL.set_val(L_left-1)
            else:
                prev_sI_de=I_de
                if L_left<14:
                    sL.set_val(L_left+1)
            sL.on_changed(updatePlot)

sL.on_changed(updatePlot)
sI_de.on_changed(updatePlot)
  
    

