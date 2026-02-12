from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re



RUNS_ROOT = Path("SolidRuns")
AVERAGES_ROOT = RUNS_ROOT / "Averages"

# -----------------------------
# Main loop over domains
# -----------------------------
for domain_dir in sorted(AVERAGES_ROOT.glob("Lx*_Ly*")):

    match = re.search(r"Lx(\d+)_Ly(\d+)", domain_dir.name)
    if not match:
        continue

    Lx, Ly = match.groups()
    print(f"\nProcessing {domain_dir.name}")

    # find wound-size subfolders (Rxx)
    wound_dirs = [
        f for f in domain_dir.iterdir()
        if f.is_dir() and re.match(r"R\d+", f.name)
    ]

    if not wound_dirs:
        print("  No wound size folders found, skipping domain")
        continue

    # -----------------------------
    # Loop over wound sizes
    # -----------------------------
    for wound_dir in sorted(wound_dirs):
        radius = wound_dir.name  # e.g. R50
        for param_dir in wound_dir.iterdir():
            if not param_dir.is_dir():
                continue
            #avg_file = wound_dir / "simulation_results_averages.txt"
            avg_file = (AVERAGES_ROOT/ domain_dir.name/ wound_dir.name/ param_dir.name / "simulation_results_averages.txt")

            if not avg_file.exists():
                print(f"  Skipping {radius} (no averages file)")
                continue

            print(f"  Plotting {radius}")

            # NOTE: header length must match compute_averages.py
            data = np.loadtxt(avg_file, delimiter=",", skiprows=5)

            mcs = data[:, 0]
            mean = data[:, 1]
            sem = data[:, 3]

            # -----------------------------
            # Plot
            # -----------------------------
            plt.figure()
            # --- testing fittin ---
            #plt.semilogy(mcs, mean)
            #plt.loglog(mcs, mean)

            # --- original plotting ---
            plt.plot(mcs, mean, label="Mean wound area")
            plt.fill_between(mcs, mean - sem, mean + sem, alpha=0.3, label="SEM")

            plt.xlabel("MCS")
            plt.ylabel("Normalized wound area (no. of cells)")
            #plt.xscale("log") 
            #plt.yscale("log")
            plt.title(f"Mean wound area vs MCS (Lx={Lx}, Ly={Ly}, {radius}%)")
            plt.legend()
            plt.tight_layout()

            # save figure inside wound directory
            out_fig = param_dir / "mean_wound_vs_mcs.png"
            if out_fig.exists():
                out_fig.unlink()
                print("    Deleted old figure")

            plt.savefig(out_fig, dpi=300)
            plt.close()

            print(f"    Saved plot → {out_fig.name}")

print("\nAll plots generated.")
