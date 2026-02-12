import cc3d
from cc3d.core.PySteppables import SteppableBasePy
#from cc3d.core.PyCoreSpecs import PixelTrackerPlugin, BoundaryPixelTrackerPlugin, NeighborTrackerPlugin
import numpy as np 
#import math
from pathlib import Path


#important! need square domain from wound creation logic
thick_f = 4
thick_w = 2
domain_size=200
boundary_size=2*(thick_f+thick_w)
grid_x = domain_size+boundary_size #multiple of 12 bc of layers 
grid_y = grid_x


#woundMakerTime=100 #mcs when wound is created -- No longer needed with WoundMakerForce.py instead of Steppable_S.py
wR=40 #wound radius in % of grid x 

wR_pix = int((wR/100)*(grid_x-boundary_size)/2) # truncates not rounding 
#wR = 40 # wound radius in pixels 
target_volume, lambda_volume = 100, 0.5
wound_mcs = None
relaxation_mcs = 200 #after domain completely filled wait relaxation_mcs before opening wound 
force=1200

N=5 #repeated runs
t=100001 #not inclusive: last mcs = t-1 ----- Maximum MCS 


r_fc = (min(grid_x, grid_y) // 2) - (thick_w + thick_f)
N_expected = int(np.pi * r_fc**2 / target_volume)

domain_filled = False
domain_filled_mcs = None

def parameter_tag():
    return f"lv{lambda_volume}_f{force}"



r_folder = ( Path("SolidRuns") / f"Lx{domain_size}_Ly{domain_size}" / f"R{wR}" / parameter_tag() ) 

