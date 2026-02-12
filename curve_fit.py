from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Configuration
# -----------------------------
RUNS_ROOT = Path("SolidRuns")
CURVEFIT_ROOT = RUNS_ROOT / "CurveFit"
PAD_VALUE = 0.0

# -----------------------------
# Models
# -----------------------------
def arctan_model(x, A, B):
    return A * np.arctan(B * x) * x

def quad_model(x, a, b, c):
    return a * x**2 + b * x + c

def stretched_exp(x, A, tau, n):
    return A * np.exp(-(x / tau)**n)

# -----------------------------
# Helper functions
# -----------------------------
def load_average_file(avg_file):
    data = np.loadtxt(avg_file, delimiter=",", comments="#")
    mcs = data[:, 0]
    mean_area = data[:, 1]
    return mcs, mean_area

def trim_data(mcs, mean_area):
    """Start at mcs=1 and remove trailing zeros."""
    mask = mcs >= 1
    mcs = mcs[mask]
    mean_area = mean_area[mask]

    nonzero_idx = np.where(mean_area > 0)[0]
    if len(nonzero_idx) == 0:
        return np.array([]), np.array([])

    last_idx = nonzero_idx[-1] + 1
    return mcs[:last_idx], mean_area[:last_idx]

def fit_curve(x, y, model):
    try:
        popt, _ = curve_fit(model, x, y, maxfev=5000)
    except Exception as e:
        print("  Curve fitting failed:", e)
        popt = [np.nan] * (2 if model == arctan_model else 3)
    return popt

def fit_stretched_exp(x, y):
    try:
        popt, _ = curve_fit(
            stretched_exp,
            x,
            y,
            bounds=(
                [0.0, 1e-6, 0.1],     # A > 0, tau > 0, n >= 0.1
                [np.inf, np.inf, 3.0]
            ),
            maxfev=10000
        )
    except Exception as e:
        print("  Stretched exponential fit failed:", e)
        popt = [np.nan, np.nan, np.nan]
    return popt

def save_fit_parameters(fit_file, arctan_params, quad_params, stretched_params):
    with open(fit_file, "w") as f:
        f.write("# arctan model: y = A*arctan(B*x)*x\n")
        f.write(f"A,B = {','.join(map(str, arctan_params))}\n\n")

        f.write("# quadratic model: y = a*x^2 + b*x + c\n")
        f.write(f"a,b,c = {','.join(map(str, quad_params))}\n\n")

        f.write("# stretched exponential: y = A*exp(-(x/tau)^n)\n")
        f.write(f"A,tau,n = {','.join(map(str, stretched_params))}\n")

def plot_fit(mcs, mean_area,
             arctan_params, quad_params, stretched_params,
             fig_file):

    plt.figure(figsize=(6, 4))
    plt.plot(mcs, mean_area, "ko", label="Data")

    if not np.isnan(arctan_params).any():
        plt.plot(
            mcs,
            arctan_model(mcs, *arctan_params),
            "r-",
            label="Arctan fit"
        )

    if not np.isnan(quad_params).any():
        plt.plot(
            mcs,
            quad_model(mcs, *quad_params),
            "b--",
            label="Quadratic fit"
        )

    if not np.isnan(stretched_params).any():
        plt.plot(
            mcs,
            stretched_exp(mcs, *stretched_params),
            "g:",
            linewidth=2,
            label="Stretched exp fit"
        )

    plt.xlabel("MCS")
    plt.ylabel("Normalized wound area")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_file)
    plt.close()

    print(f"  Saved figure → {fig_file.name}")

# -----------------------------
# Main loop over datasets
# -----------------------------
for domain_dir in sorted((RUNS_ROOT / "Averages").glob("Lx*_Ly*")):
    print(f"\nProcessing domain {domain_dir.name}")

    wound_dirs = [
        f for f in domain_dir.iterdir()
        if f.is_dir() and re.match(r"R\d+", f.name)
    ]

    if not wound_dirs:
        print("  No wound size folders found, skipping")
        continue

    for wound_dir in sorted(wound_dirs):
        for param_dir in wound_dir.iterdir():
            if not param_dir.is_dir():
                continue

            avg_file = param_dir / "simulation_results_averages.txt"
            if not avg_file.exists():
                print(f"  Missing averages file in {param_dir.name}, skipping")
                continue

            print(f"  Processing {wound_dir.name}/{param_dir.name}")

            mcs, mean_area = load_average_file(avg_file)
            mcs, mean_area = trim_data(mcs, mean_area)

            if len(mcs) == 0:
                print("  No data to fit after trimming, skipping")
                continue

            arctan_params = fit_curve(mcs, mean_area, arctan_model)
            quad_params = fit_curve(mcs, mean_area, quad_model)
            stretched_params = fit_stretched_exp(mcs, mean_area)

            out_dir = CURVEFIT_ROOT / domain_dir.name / wound_dir.name / param_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            fit_file = out_dir / "fit_parameters.txt"
            save_fit_parameters(
                fit_file,
                arctan_params,
                quad_params,
                stretched_params
            )

            fig_file = out_dir / "fit_plot.png"
            plot_fit(
                mcs,
                mean_area,
                arctan_params,
                quad_params,
                stretched_params,
                fig_file
            )

print("\nAll curve fits completed.")
