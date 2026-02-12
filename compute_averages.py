from pathlib import Path
import numpy as np
import re
from Parameters import * #only using this for target_volume normalisation

## this script computes the mean wound area (over replicates) for each mcs 
## data is saved in 'simulation_results_averages.txt' in the Averages > LxLy > R folder 


# -----------------------------
# Configuration
# -----------------------------
RUNS_ROOT = Path("SolidRuns")
PAD_VALUE = 0.0      # wound area after closure

# -----------------------------
# Helper functions
# -----------------------------
def load_runs(param_dir):
    """Load all replicate runs for one domain."""
    """Cut runs to start at individual wound_maker_time."""
    runs = []
    for filepath in sorted(param_dir.glob("simulation_results_*.txt")):
        with open(filepath, "r") as fp:
            lines = fp.readlines()
        for line in lines:
            if line.startswith("# Wound created at mcs:"):
                wound_made_mcs = int(line.split(":", 1)[1].strip())

        header_length=4 #check if header is 4 lines
        total_skip=header_length+wound_made_mcs
        data = np.loadtxt(filepath, delimiter=",", skiprows=total_skip) 
        # rebase MCS so wound starts at 0
        data[:, 0] -= data[0, 0]
        if data.shape[1] != 2:
            raise ValueError(f"{filepath.name} has wrong format")
        runs.append(data)
    return runs


def pad_runs(runs):
    """Pad runs to the same maximum MCS using PAD_VALUE."""
    max_mcs = int(max(run[-1, 0] for run in runs))
    padded = []

    for run in runs:
        last_mcs = int(run[-1, 0])
        if last_mcs < max_mcs:
            pad_mcs = np.arange(last_mcs + 1, max_mcs + 1)
            pad_vals = np.full_like(pad_mcs, PAD_VALUE)
            run = np.vstack((run, np.column_stack((pad_mcs, pad_vals))))
        padded.append(run)

    return np.stack(padded)


def compute_statistics(padded_runs):
    """Compute mean, std, SEM across runs."""
    wound_areas = padded_runs[:, :, 1]
    n_runs = padded_runs.shape[0]

    mean = np.mean(wound_areas, axis=0)
    std = np.std(wound_areas, axis=0, ddof=1)
    sem = std / np.sqrt(n_runs)

    return mean, std, sem


def save_averages(domain_dir, wound_dir, param_dir, Lx, Ly, radius, mcs, mean, std, sem):
    """Save averaged data to file."""
    #out_file = wound_dir / "simulation_results_averages.txt"
    avg_root = RUNS_ROOT / "Averages"
    out_dir = avg_root / domain_dir.name / wound_dir.name / param_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "simulation_results_averages.txt"  

    with open(out_file, "w") as f:
        f.write(f"# Domain: Lx={Lx}, Ly={Ly}\n")
        f.write(f"# Wound radius={radius}\n")
        f.write(f"# targetVolume={target_volume}\n")
        f.write(f"# data padded by={PAD_VALUE}\n")
        f.write("# Columns: mcs, mean wound area, std , sem \n")
        #f.write("mcs,mean,std,sem\n")

        for i in range(len(mcs)):
            f.write(
                f"{int(mcs[i])},"
                f"{mean[i]/target_volume},"
                f"{std[i]/target_volume},"
                f"{sem[i]/target_volume}\n"
            )

    print(f"  Saved averages → {out_file.name}")


# -----------------------------
# Main loop over domains
# -----------------------------
for domain_dir in sorted(RUNS_ROOT.glob("Lx*_Ly*")):
    print(f"\nProcessing {domain_dir.name}")

    match = re.search(r"Lx(\d+)_Ly(\d+)", domain_dir.name)
    if not match: # this avoids accidentially processing junk folders 
        print("  Skipping (cannot parse domain size)")
        continue

    Lx, Ly = match.groups()

    wound_dirs = [f for f in domain_dir.iterdir() if f.is_dir() and re.match(r"R\d+", f.name)]
    if not wound_dirs:
        print("  No wound size folders found, skipping domain")
        continue

    for wound_dir in sorted(wound_dirs):
        for param_dir in wound_dir.iterdir():
            if not param_dir.is_dir():
                continue
            radius = wound_dir.name  # e.g., "R50"
            print(f"  Processing wound size {radius}")    

            # delete old averages file if present
            #avg_file = wound_dir / "simulation_results_averages.txt"
            #if avg_file.exists():
            #    avg_file.unlink()
            #    print("  Deleted old averages file")
            # delete old averages file if present (in Averages folder)
            avg_root = RUNS_ROOT / "Averages"
            avg_dir = avg_root / domain_dir.name / wound_dir.name / param_dir.name
            avg_file = avg_dir / "simulation_results_averages.txt"

            if avg_file.exists():
                avg_file.unlink()
                print("  Deleted old averages file")

            runs = load_runs(param_dir)
            if len(runs) < 1:
                print("  Skipping (not enough runs)")
                continue
            
            padded = pad_runs(runs)
            for i, run in enumerate(padded):
                print(i, run.shape)

            mcs = padded[0, :, 0]
            mean, std, sem = compute_statistics(padded)

            save_averages(domain_dir, wound_dir, param_dir, Lx, Ly, radius, mcs, mean, std, sem)

print("\nAll averages computed.")
