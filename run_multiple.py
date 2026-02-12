import subprocess
import cc3d
from pathlib import Path
import shutil

from Parameters import *
from Parameters import parameter_tag

simulation_script = "SolidBC_main.py"
#optional: clean folder
#r_folder = ( Path("SolidRuns") / f"Lx{domain_size}_Ly{domain_size}" / f"R{wR} " / parameter_tag() ) 
if r_folder.exists():
    for f in r_folder.iterdir():
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f) #erases subfolders 
    print(f"[run_multiple] Cleared folder {r_folder}")

for run_id in range(N):
    print(f"\n=== Starting simulation {run_id} ===")
    
    # Run each simulation in a separate Python process
    result = subprocess.run(
        ["python", simulation_script, str(run_id)],
        capture_output=True, text=True
    )

    # Print simulation stdout and stderr
    print(result.stdout)
    if result.stderr:
        print("Errors / Warnings:")
    #    print(result.stderr)
    
    print(f"=== Finished simulation {run_id} ===\n")
