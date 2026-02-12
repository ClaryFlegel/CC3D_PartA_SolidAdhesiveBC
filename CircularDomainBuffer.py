import cc3d
from cc3d.core.PySteppables import SteppableBasePy
import Parameters
import numpy as np

class CircularDomainInitialiser(SteppableBasePy):
    def __init__(self, frequency=1):
        super().__init__(frequency)

    def start(self):
        thick_f = 4
        thick_w = 2

        cx = Parameters.grid_x // 2
        cy = Parameters.grid_y // 2

        r_fw = (min(Parameters.grid_x, Parameters.grid_y) // 2) - thick_w  # outer radius (wall)
        r_fc = (min(Parameters.grid_x, Parameters.grid_y) // 2) - (thick_w + thick_f)  # inner radius (cells)

        spacing = np.sqrt(Parameters.target_volume*0.8)   # distance between cell centers
        seed_radius = 2                       # multi-pixel seed radius

        wall = self.new_cell(self.WALL)
        fluid = self.new_cell(self.FLUID)

        for cell in self.cell_list_by_type(self.WALL):
            cell.targetVolume = 1
            cell.lambdaVolume = 1
        for cell in self.cell_list_by_type(self.FLUID):
            cell.targetVolume = 0
            cell.lambdaVolume = 0

        cells_created = 0
        r = spacing / 2  # start radius

        # --- Radial multi-pixel seeding ---
        while r < r_fc and cells_created < Parameters.N_expected:
            circumference = 2 * np.pi * r
            n_on_ring = max(1, int(circumference / spacing))

            for k in range(n_on_ring):
                if cells_created >= Parameters.N_expected:
                    break

                theta = 2 * np.pi * k / n_on_ring
                x0 = int(cx + r * np.cos(theta))
                y0 = int(cy + r * np.sin(theta))

                if not (0 <= x0 < Parameters.grid_x and 0 <= y0 < Parameters.grid_y):
                    continue
                if (x0 - cx)**2 + (y0 - cy)**2 > r_fc**2:
                    continue

                # Check footprint is free
                footprint_ok = True
                for dx in range(-seed_radius, seed_radius + 1):
                    for dy in range(-seed_radius, seed_radius + 1):
                        if dx*dx + dy*dy <= seed_radius**2:
                            x = x0 + dx
                            y = y0 + dy
                            if not (0 <= x < Parameters.grid_x and 0 <= y < Parameters.grid_y and self.cell_field[x, y, 0] is None):
                                footprint_ok = False
                                break
                    if not footprint_ok:
                        break
                if not footprint_ok:
                    continue

                # Create cell and paint multi-pixel disk
                cell = self.new_cell(self.CELL)
                cell.targetVolume = Parameters.target_volume
                cell.lambdaVolume = Parameters.lambda_volume
                
                
                for dx in range(-seed_radius, seed_radius + 1):
                    for dy in range(-seed_radius, seed_radius + 1):
                        if dx*dx + dy*dy <= seed_radius**2:
                            self.cell_field[x0 + dx, y0 + dy, 0] = cell

                cells_created += 1

            r += spacing

        print(f"Seeded {cells_created} cells (expected ~{Parameters.N_expected})")

        # --- Paint wall and fluid outside ---
        for x, y, z in self.every_pixel():
            r2 = (x - cx)**2 + (y - cy)**2
            if r2 > r_fw**2:
                self.cell_field[x, y, 0] = wall
            elif r_fw**2 >= r2 > r_fc**2:
                self.cell_field[x, y, 0] = fluid

        # --- Initial slow growth for all cells ---
        #for cell in self.cell_list_by_type(self.CELL):
        #    if cell.targetVolume < Parameters.target_volume:
        #        cell.lambdaVolume = 0.05 * Parameters.lambda_volume
