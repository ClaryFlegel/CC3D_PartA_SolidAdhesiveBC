from pathlib import Path
import numpy as np
from Parameters import *
import re 

# -----------------------------
# Configuration
# -----------------------------
RUNS_DIR = Path("SolidRuns")
AVERAGES_DIR = RUNS_DIR / "Averages"
AVERAGES_DIR.mkdir(exist_ok=True)

STABLE_ZERO_REQUIRED = 1  # easy to change later if we want stability logic


# -----------------------------
# Helper functions
# -----------------------------
def read_closure_mcs(result_file):
    """
    Reads a simulation_results_*.txt file and returns
    the MCS at which woundArea first becomes zero.
    Returns None if closure never occurs.
    MCS is relative to wound_made_mcs 
    """
    with open(result_file, "r") as f:
        lines = f.readlines()

    # Skip header lines starting with '#'
    data_lines = [l for l in lines if not l.startswith("#")]

    # Remove column header
    data_lines = data_lines[1:]

    for line in lines:
            if line.startswith("# Wound created at mcs:"):
                wound_made_mcs = int(line.split(":", 1)[1].strip())
                break

    zero_counter = 0
    for line in data_lines:
        mcs, wound_area = line.strip().split(",")
        mcs = int(mcs)
        wound_area = int(wound_area)

        if wound_area == 0 and mcs >= wound_made_mcs:
            zero_counter += 1
            if zero_counter >= STABLE_ZERO_REQUIRED:
                mcs 
                return mcs
        else:
            zero_counter = 0

    return None


def process_domain(param_dir):
    """
    Processes one domain folder (e.g. Lx306_Ly306)
    Returns a dict with statistics.
    """
    closure_times = []
    rel_closure_times = []
    wound_maker_mcs = []

    for result_file in sorted(param_dir.glob("simulation_results_*.txt")):
        if result_file.name == "simulation_results_averages.txt": #should not be there but just in case
            continue
    
        with open(result_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("# Wound created at mcs:"):
                wound_made_mcs = int(line.split(":", 1)[1].strip())
                if wound_made_mcs is not None:
                    wound_maker_mcs.append(wound_made_mcs)
                break

        mcs = read_closure_mcs(result_file)
        if mcs is not None:
            closure_times.append(mcs)
            rel_closure_time=mcs-wound_made_mcs
            rel_closure_times.append(rel_closure_time)


    closure_times = np.array(closure_times)
    n_runs = len(closure_times)
    rel_closure_times = np.array(rel_closure_times)
    wound_maker_mcs = np.array(wound_maker_mcs)

    stats = {
        "n_runs": n_runs,
        "mean_closure_mcs": np.mean(closure_times) if n_runs else np.nan,
        "std_closure_mcs": np.std(closure_times,ddof=1) if n_runs else np.nan,
        "sem_closure_mcs": (np.std(closure_times,ddof=1) / np.sqrt(n_runs) if n_runs > 1 else np.nan),
        "min_closure_mcs": np.min(closure_times) if n_runs else np.nan,
        "max_closure_mcs": np.max(closure_times) if n_runs else np.nan,
        "all_closure_mcs": closure_times,
        "all_rel_closure_mcs": rel_closure_times,
        "wound_maker_mcs": wound_maker_mcs,
        "mean_rel_closure_mcs": np.mean(rel_closure_times) if n_runs else np.nan,
        "std_rel_closure_mcs": np.std(rel_closure_times,ddof=1) if n_runs else np.nan,
        "sem_rel_closure_mcs": (np.std(rel_closure_times,ddof=1) / np.sqrt(n_runs) if n_runs > 1 else np.nan),
        "min_rel_closure_mcs": np.min(rel_closure_times) if n_runs else np.nan,
        "max_rel_closure_mcs": np.max(rel_closure_times) if n_runs else np.nan,
        "mean_wound_maker_mcs": np.mean(wound_maker_mcs) if n_runs else np.nan
    }

    return stats


def save_stats(domain_dir, domain_name, wound_dir, param_dir, stats):
    """
    Saves statistics to Runs/Averages/<domain>_avg.txt
    """
    domain_out = AVERAGES_DIR / domain_dir.name / wound_dir.name / param_dir.name
    domain_out.mkdir(exist_ok=True)

    out_file = domain_out /f"{domain_name}_avg.txt"


    if out_file.exists():
        out_file.unlink()
        print("  Deleted old avg file")

    with open(out_file, "w") as f:
        f.write(f"# Domain: {domain_name}\n")
        f.write(f"# Wound Radius Created: R={wound_dir.name} % of domain size\n")
        f.write(f"# Number of successful runs: {stats['n_runs']}\n\n")

        f.write("# Nominal closure time statistics\n")
        f.write("mean_closure_mcs,"
                f"{stats['mean_closure_mcs']}\n")
        f.write("std_closure_mcs,"
                f"{stats['std_closure_mcs']}\n")
        f.write("sem_closure_mcs,"
                f"{stats['sem_closure_mcs']}\n")
        f.write("min_closure_mcs,"
                f"{stats['min_closure_mcs']}\n")
        f.write("max_closure_mcs,"
                f"{stats['max_closure_mcs']}\n")
        f.write("wound_maker_mcs,"
                f"{stats['wound_maker_mcs']}\n\n")
        
        f.write("# Relative closure time statistics (closure times - wound_made_mcs) \n")
        f.write("mean_relative_closure_mcs,"
                f"{stats['mean_rel_closure_mcs']}\n")
        f.write("std_relative_closure_mcs,"
                f"{stats['std_rel_closure_mcs']}\n")
        f.write("sem_relative_closure_mcs,"
                f"{stats['sem_rel_closure_mcs']}\n")
        f.write("min_relative_closure_mcs,"
                f"{stats['min_rel_closure_mcs']}\n")
        f.write("max_relative_closure_mcs,"
                f"{stats['max_rel_closure_mcs']}\n")
        f.write("mean_wound_maker_mcs,"
                f"{stats['mean_wound_maker_mcs']}\n\n")


        f.write("# Individual nominal closure times (per run)\n")
        for mcs in stats["all_closure_mcs"]:
            f.write(f"{mcs}\n")

        f.write("# Individual relative closure times (per run)\n")
        for mcs in stats["all_rel_closure_mcs"]:
            f.write(f"{mcs}\n")

        f.write("# Individual wound maker times (per run)\n")
        for mcs in stats["wound_maker_mcs"]:
            f.write(f"{mcs}\n")

    print(f"Saved averages to {out_file}")


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":

    for domain_dir in sorted(RUNS_DIR.glob("Lx*_Ly*")):
        if not domain_dir.is_dir():
            continue
        wound_dirs = [f for f in domain_dir.iterdir() if f.is_dir() and re.match(r"R\d+", f.name)]
        if not wound_dirs:
            print("  No wound size folders found, skipping domain")
            continue
        
        for wound_dir in sorted(wound_dirs):
            for param_dir in wound_dir.iterdir():
                if not param_dir.is_dir():
                    continue
                # check if wound folder contains (the correct) data files
                has_raw_data = any(param_dir.glob("simulation_results_*.txt"))
                if not has_raw_data:
                    print(f"Skipping {domain_dir.name}_{wound_dir.name}_{param_dir.name} (no simulation_results_*.txt files)")
                    continue

                domain_name = f"{domain_dir.name}_{wound_dir.name}_{param_dir.name}"
                print(f"Processing domain: {domain_name}")

                stats = process_domain(param_dir)
                save_stats(domain_dir,domain_name, wound_dir, param_dir, stats)
