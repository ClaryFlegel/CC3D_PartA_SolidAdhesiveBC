#!/usr/bin/env python3
"""
One-command post-processing workflow.

Runs all post-processing scripts in the correct order.
Aborts immediately if any stage fails.
Safe to rerun at any time.
"""

import subprocess
import sys
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------

SCRIPTS = [
    ("Computing time-resolved averages", "compute_averages.py"),
    ("Computing closure-time statistics", "avg.py"),
    ("Plotting mean wound vs MCS", "plot_mean_wound_vs_mcs.py"),
    ("Plotting closure time statistics", "plot_closuretime_multipanel.py"),
    ("Plotting sensitivity analysis", "plot_sensitivity_fold_change.py"),
    ("Computing and plotting relative strain vs radial distance", "binning_plot_relative_strain.py"),
    ("Curve Fitting", "curve_fit.py")
]
#    ("Performing curve fitting", "mean_wound_curve_fit.py"),
#]

PYTHON_EXECUTABLE = sys.executable  # ensures same venv/interpreter


# -----------------------------
# Runner
# -----------------------------

def run_script(description, script_name, index, total):
    script_path = Path(script_name)

    if not script_path.exists():
        print(f"[{index}/{total}] ERROR: {script_name} not found")
        sys.exit(1)

    print(f"[{index}/{total}] {description}")

    result = subprocess.run(
        [PYTHON_EXECUTABLE, script_path.name],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        print(f"\nERROR in stage {index}: {script_name}")
        print("Aborting post-processing.")
        sys.exit(result.returncode)


def main():
    total = len(SCRIPTS)

    for i, (description, script_name) in enumerate(SCRIPTS, start=1):
        run_script(description, script_name, i, total)

    print("\nPost-processing complete.")


if __name__ == "__main__":
    main()
