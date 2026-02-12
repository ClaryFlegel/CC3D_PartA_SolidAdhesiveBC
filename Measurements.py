import cc3d
from cc3d.core.PySteppables import SteppableBasePy
#from WoundMakerForce import WoundMakerSteppable
import numpy as np 
import math

#from Parameters import *
import Parameters
from pathlib import Path

class Measurements(SteppableBasePy):
    def __init__(self, frequency=1, run_id=0):
        super().__init__(frequency=frequency)

        self.run_id=run_id
        self.wound_closed_flag = False # it is not yet opened really
        self.header_updated = False
        self.closed_counter = 0
        self.closed_counter2 = 0

        self.MEAN_WINDOW = 50       # MCS per averaging window
        self.MEAN_TOL = 10         # pixels
        self.REL_TOL = 0.005  # 0.5%
        self.area_buffer = []
        self.prev_mean_area = None
        #self.stop_reason = None
        self.MIN_AREA_FOR_FAILURE_CHECK = 10  # pixels (tune this)

    def start(self,run_id=0):
        self.run_dir = ( Path("SolidRuns") / f"Lx{Parameters.domain_size}_Ly{Parameters.domain_size}" / f"R{Parameters.wR}" / Parameters.parameter_tag() )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Optional: delete all old measurement files for this wound folder
        #for f in self.run_dir.glob("simulation_results_*.txt"):
        #    f.unlink()
            #print(f"[Measurements] Deleted old file {f.name}")

        # create file path
        self.output_file = self.run_dir / f"simulation_results_{self.run_id}.txt"

        #self.output_file = f"simulation_results_{run_id}.txt"
        #with open(self.output_file, "a") as f:
        #    f.write("mcs, wound Area\n")

        with open(self.output_file, "w") as f:
            f.write(f"# Domain Size: Lx={Parameters.domain_size}, Ly={Parameters.domain_size}\n")
            f.write(f"# Wound Radius Created: R={Parameters.wR} % of domain size, R={Parameters.wR_pix} in pixels\n")
            f.write("# Wound created at mcs: {wound_mcs}\n")  # placeholder
            f.write("mcs,woundArea\n")
            #f.write("# Stop reason:\n") #placeholder
            
  
    def step(self,mcs):
        
        #print("wound_mcs seen by Measurements:", Parameters.wound_mcs)
        if not self.header_updated:
            #wound_steppable = self.get_steppable_by_class(WoundMakerSteppable)
            #wound_mcs = wound_steppable.wound_mcs

            if Parameters.wound_mcs is not None:
                self._update_wound_header(Parameters.wound_mcs)
                self.header_updated = True
        #print(f"Measurements step called at MCS={mcs}")  # Debug line
        # woundArea=0
        # occupiedArea=0
        # for cell in self.cell_list:
        #     occupiedArea += cell.volume
        # woundArea=(grid_x-3)*(grid_y-3) - occupiedArea
        
        #if self.wound_closed_flag:
        #    return
        
        woundArea = self.compute_wound_area()
        #print(woundArea)
        
        with open(self.output_file, "a") as f:
            f.write(f"{mcs},{woundArea}\n")
        

        # -------------------------------------------------
        # Condition 1: exact closure (area == 0)
        # -------------------------------------------------

        if self.header_updated and not self.wound_closed_flag:
            if woundArea == 0:
                self.closed_counter += 1
            else:
                self.closed_counter = 0

            if self.closed_counter >= 3:
                print(f"Stopped at mcs {mcs}: wound area reached zero")
                self.wound_closed_flag = True
                #self.simulator.setStopSimulationFlag(True)
                #return
                self.stop_simulation()
                return

        # -------------------------------------------------
        # Condition 2: mean-area comparison (healing failure)
        # -------------------------------------------------
        if woundArea > self.MIN_AREA_FOR_FAILURE_CHECK:
            self.area_buffer.append(woundArea)
            

            # Optional: wait until wound exists and buffers are full
            if Parameters.wound_mcs is None:
                return

            if mcs < Parameters.wound_mcs + 2 * self.MEAN_WINDOW:
                return

            if len(self.area_buffer) < 2 * self.MEAN_WINDOW:
                return

            prev_window = self.area_buffer[:self.MEAN_WINDOW]
            curr_window = self.area_buffer[self.MEAN_WINDOW:2 * self.MEAN_WINDOW]
            #print(prev_window)
            prev_mean = np.mean(prev_window)
            curr_mean = np.mean(curr_window)

            if self.header_updated and not self.wound_closed_flag:
                if curr_mean >= prev_mean * (1 - self.REL_TOL):
                #if curr_mean >= prev_mean - self.MEAN_TOL:
                    self.closed_counter2 += 1
                    print(f"area stopping criterion activated for {self.closed_counter2}/3.")
                else:
                    self.closed_counter2 = 0
                    print(f"Reset area stopping criterion.")

                if self.closed_counter2 >= 3:            
                    print(f"Stopped at mcs {mcs}: wound not closing")
                    self.wound_closed_flag = True
                    self.stop_simulation()
                    #self.simulator.setMaxMCS(mcs + 1)
                    return
        
            # -------------------------------------------------
            # Slide window forward
            # -------------------------------------------------
            if not self.wound_closed_flag:
                self.area_buffer = self.area_buffer[self.MEAN_WINDOW:]


    def compute_wound_area(self):
        woundArea = 0
        for x in range(Parameters.grid_x):
            for y in range(Parameters.grid_y):
                if self.cellField[x, y, 0] is None:  # medium pixel
                    woundArea += 1
        return woundArea
    
    def _update_wound_header(self, wound_mcs):
        with open(self.output_file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith("# Wound created at mcs:"):
                lines[i] = f"# Wound created at mcs: {wound_mcs}\n"
                break

        with open(self.output_file, "w") as f:
            f.writelines(lines)
