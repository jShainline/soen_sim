import numpy as np
from util import color_dictionary, physical_constants
from _functions import dendrite_current_splitting
colors = color_dictionary()

p = physical_constants()

#%%

# dendrite_current_splitting(Ic,Iflux,Ib,M,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)

Ic = 40
If = 0
Ib = [77,36,35]

Lm1 = 130
Lm2 = 20
M = np.sqrt(Lm1*Lm2)
Ldr1 = 0
Ldr2 = 20

L1 = 200
L2 = 77.5
L3 = 77.5e-3

Idr1_prev = Ib[0]/2
Idr2_prev = Ib[0]/2
Ij2_prev = Ib[1]
Ij3_prev = Ib[2] 

for ii in range(5):
    Idr1_next, Idr2_next, Ij2_next, Ij3_next, I1, I2, I3 = dendrite_current_splitting(Ic,If,Ib,M,Lm2,Ldr1,Ldr2,L1,L2,L3,Idr1_prev,Idr2_prev,Ij2_prev,Ij3_prev)
    Idr1_prev = Idr1_next
    Idr2_prev = Idr2_next
    Ij2_prev = Ij2_next
    Ij3_prev = Ij3_next
    
print('Idr1 = {}'.format(Idr1_next))    