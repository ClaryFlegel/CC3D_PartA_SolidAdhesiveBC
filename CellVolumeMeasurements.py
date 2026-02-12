import cc3d
from cc3d.core.PySteppables import SteppableBasePy
#from Parameters import *
import Parameters
from pathlib import Path
import numpy as np
import glob
#from WoundMakerForce import WoundMakerSteppable

class CellVolumeMeasurement(SteppableBasePy):
    """
    Steppable to record CELL type volumes and positions every MCS,
    including lambda_volume and radial distance from wound center.
    Cells are written sorted by radial distance for easier post-processing.
    """

    def __init__(self, frequency=1, run_id=0):
        super().__init__(frequency=frequency)
        self.run_id = run_id
        self.output_file = None
        # Wound center
        self.wound_center = np.array([Parameters.grid_x/2, Parameters.grid_y/2])
        self.header_updated = False

    def start(self, run_id=0):
        # Build folder path: Runs/Lx*_Ly*/R*/
        self.run_dir = ( Path("SolidRuns") / f"Lx{Parameters.domain_size}_Ly{Parameters.domain_size}" / f"R{Parameters.wR}" / Parameters.parameter_tag() )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # File name: cell_field_data_<run_id>.txt
        self.output_file = self.run_dir / f"cell_field_data_{self.run_id}.txt"
        
        #for f in self.run_dir.glob("cell_field_data_*.txt"): #deletes existing files from previous runs 
        #    f.unlink()
        #wound_steppable = self.get_steppable_by_class(WoundMakerSteppable)
        #wound_mcs = wound_steppable.wound_mcs 

        # Write header
        with open(self.output_file, "w") as f:
            f.write(f"# Domain Size: Lx={Parameters.domain_size}, Ly={Parameters.domain_size}\n")
            f.write(f"# Wound Radius: R={Parameters.wR} % of domain size, R={Parameters.wR_pix} in pixels\n")
            f.write(f"# Fixed Target Volume: {Parameters.target_volume}\n")
            f.write(f"# Fixed Lambda Volume: {Parameters.lambda_volume}\n")
            f.write(f"# Wound created at mcs: {Parameters.wound_mcs}\n")
            f.write("# Columns: mcs, cell_id, xCOM, yCOM, volume, radial_distance, lambda_volume\n")

    def step(self, mcs):
        #for cell in self.cell_list_by_type(self.FLUID):
            #print("2:",cell.lambdaVolume,cell.targetVolume)

        if not self.header_updated:
            #wound_steppable = self.get_steppable_by_class(WoundMakerSteppable)
            #wound_mcs = wound_steppable.wound_mcs

            if Parameters.wound_mcs is not None:
                self._update_wound_header(Parameters.wound_mcs)
                self.header_updated = True

        # Collect data for all CELLs
        cell_data = []
        for cell in self.cell_list_by_type(self.CELL):
            if cell is None:
                continue  # safety
            com = np.array([cell.xCOM, cell.yCOM])
            radial_dist = np.linalg.norm(com - self.wound_center)
            cell_data.append((
                radial_dist,
                cell.id,
                cell.xCOM,
                cell.yCOM,
                cell.volume,
                cell.lambdaVolume
            ))

        # Sort cells by radial distance
        cell_data.sort(key=lambda x: x[0])

        # Write to file
        with open(self.output_file, "a") as f:
            for dist, cid, x, y, vol, lam in cell_data:
                f.write(f"{mcs},{cid},{x:.3f},{y:.3f},{vol},{dist:.3f},{lam}\n")

    def finish(self):
        print(f"[CellVolumeMeasurement] All data saved to {self.output_file.name}")
    

    def _update_wound_header(self, wound_mcs):
        with open(self.output_file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith("# Wound created at mcs:"):
                lines[i] = f"# Wound created at mcs: {wound_mcs}\n"
                break

        with open(self.output_file, "w") as f:
            f.writelines(lines)