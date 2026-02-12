import cc3d
from cc3d.core.PySteppables import SteppableBasePy
#from cc3d.core.PyCoreSpecs import PixelTrackerPlugin, BoundaryPixelTrackerPlugin, NeighborTrackerPlugin
import numpy as np 
#import math

#from Parameters import *
import Parameters



class WoundMakerSteppable(SteppableBasePy):
    def __init__(self, frequency=1,run_id=0):
        super().__init__(frequency=frequency)
        self.run_id = run_id
        self.wound_made = False   
        self.domain_filled = False
        self.wound_mcs = None
        self.wait_time_counter = 0
        self.fluid_fluid=False




    def start(self):
        # safe: start() is called once per simulation
        print(f"Initializing wound for run {self.run_id}")

        
        #freezing fluid for relaxation phase 
        for cell in self.cell_list_by_type(self.FLUID):
            cell.lambdaVolume = 1e9
            #print(cell.lambda_volume)
            cell.targetVolume = cell.volume
            #print(cell.lambdaVolume,cell.targetVolume)
            print("Fluid cells frozen (for relaxation)")
            
            
            

    def step(self,mcs):
        #print(f"2: Measurements step called at MCS={mcs}")  # Debug line

        #if not Parameters.wound_mcs:
            #n_cells = len(self.cell_list_by_type(self.CELL))
            #print(f"MCS {mcs}: Cells = {n_cells}, Expected ≈ {Parameters.N_expected}")
        #if Parameters.domain_filled and not self.wound_made:
            #print(f"Relaxation Phase for {Parameters.relaxation_mcs} mcs")
            #if self.wait_time_counter >= 1:
                #print(f"Domain fully occupied at MCS {Parameters.domain_filled_mcs}")
        #if self.wound_made:
            #print("Healing Phase")

        min_vol = 1.0

        for cell in self.cell_list_by_type(self.CELL):
            cell.lambdaVolume = Parameters.lambda_volume*(cell.volume + Parameters.target_volume)/(max(cell.volume,min_vol))

        if not self.wound_made:
            if not self.domain_filled:
                self.domain_filled = self.is_domain_filled(tolerance=0) #tolerance=0 domain must be fully occupied

            if self.domain_filled:
                Parameters.domain_filled=True
                if self.wait_time_counter == 0: 
                    Parameters.domain_filled_mcs=mcs
                    #print(f"Domain fully occupied at MCS {mcs}")
                self.wait_time_counter += 1
                if self.wait_time_counter >= Parameters.relaxation_mcs: #wait another relaxation_mcs before opening wound
                    self.make_wound(mcs)
                    if self.fluid_fluid:
                        for cell in self.cell_list_by_type(self.FLUID):
                            cell.lambdaVolume = 0
                            cell.targetVolume= 1
                        print(f"Fluid cells unfrozen at MCS {mcs}")

        if self.wound_made and mcs > self.wound_mcs:
        #if mcs > woundMakerTime: 
            for cell in self.cell_list_by_type(self.CELL):
                if cell is None:
                    print("cell_list_by_type returned None!")
                vector = self.get_local_polarity_vector(cell)
                force = Parameters.force
                # Make sure ExternalPotential plugin is loaded
                cell.lambdaVecX = -force*vector[0]  # force component pointing along X axis - towards positive X's
                cell.lambdaVecY = -force*vector[1]  # force component pointing along Y axis - towards negative Y's
                # cell.lambdaVecZ = 0.0  force component pointing along Z axis
        
        #helping to fill the domain faster
        if not self.domain_filled:
        #if mcs > woundMakerTime: 
            for cell in self.cell_list_by_type(self.CELL):
                if cell is None:
                    print("cell_list_by_type returned None!")
                vector = self.get_local_polarity_vector(cell)
                # Make sure ExternalPotential plugin is loaded
                cell.lambdaVecX = -Parameters.force*vector[0]  # force component pointing along X axis - towards positive X's
                cell.lambdaVecY = -Parameters.force*vector[1]  # force component pointing along Y axis - towards negative Y's
                # cell.lambdaVecZ = 0.0  force component pointing along Z axis
    
    def is_domain_filled(self, tolerance=0):
        medium_pixels = 0
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                if self.cellField[x, y, 0] is None:
                    medium_pixels += 1
                    if medium_pixels > tolerance:
                        return False
        return True
            

    def make_wound(self, mcs):
        cx = self.dim.x // 2
        cy = self.dim.y // 2
        rw = Parameters.wR_pix ** 2

        counter = 0
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                if (x - cx)**2 + (y - cy)**2 <= rw:
                    cell = self.cellField[x, y, 0]
                    if cell and cell.type == self.CELL:
                        self.deleteCell(cell)
                        counter += 1

        self.wound_made = True
        self.wound_mcs = mcs
        Parameters.wound_mcs = mcs
        print("Circular wound created at MCS =", mcs)
        print("Wound size in cells =", counter)

    def get_local_polarity_vector(self, cell):
        boundary_pixels = self.get_cell_boundary_pixel_list(cell)
        #fixing circular domain issue with this function
        if not boundary_pixels:
            return np.array([0.0, 0.0])

        pixels = np.array([[p.pixel.x,p.pixel.y] for p in boundary_pixels])
        #another safeguard
        if pixels.size == 0:
            return np.array([0.0, 0.0])
        vec = pixels - np.array([cell.xCOM,cell.yCOM])
        norm = np.linalg.norm(vec,axis=1,keepdims=True)
        vec_norm = np.divide(vec,norm,where=norm>1e-6)
        pixels_proj = np.round(pixels + vec_norm).astype(int)
        mask_inside = (
            (pixels_proj[:,0] >= 0) &
            (pixels_proj[:,0] < self.dim.x) &
            (pixels_proj[:,1] >= 0) &
            (pixels_proj[:,1] < self.dim.y)
        )
        pixels_proj = pixels_proj[mask_inside]

        medium_check = [False if self.cellField[int(p[0]), int(p[1]), 0] else True for p in pixels_proj]

        if np.any(medium_check): #any entry is 1 (true) 
            vec_norm=vec_norm[mask_inside][medium_check]
            vector = np.average(vec_norm, axis=0)
            return vector
            
        else:
            vector = np.array([0.0, 0.0])
            return vector
            
        

    
