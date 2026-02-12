# plot_closure_time_multipanel.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re
import math

# ============================================================
# Paths
# ============================================================

AVG_ROOT = Path("SolidRuns/Averages")

# ============================================================
# Regex
# ============================================================

DOMAIN_RE = re.compile(r"Lx(\d+)_Ly\1")
WOUND_RE  = re.compile(r"R([\d\.]+)")          # percent
PARAM_RE  = re.compile(r"lv([\d\.]+)_f([\d\.]+)")

# ============================================================
# Discover data structure
# ============================================================

# structure[lv][domain][force][wound] = avg_file
structure = {}

for domain_dir in AVG_ROOT.iterdir():
    if not domain_dir.is_dir():
        continue

    dmatch = DOMAIN_RE.match(domain_dir.name)
    if not dmatch:
        continue
    domain = int(dmatch.group(1))

    for wound_dir in domain_dir.iterdir():
        if not wound_dir.is_dir():
            continue

        wmatch = WOUND_RE.match(wound_dir.name)
        if not wmatch:
            continue
        wound_frac = float(wmatch.group(1))

        for param_dir in wound_dir.iterdir():
            if not param_dir.is_dir():
                continue

            pmatch = PARAM_RE.match(param_dir.name)
            if not pmatch:
                continue

            lv = float(pmatch.group(1))
            force = float(pmatch.group(2))

            # Find *_R*_avg.txt files
            avg_files = sorted(param_dir.glob("*_avg.txt"))
            if not avg_files:
                continue

            for avg_file in avg_files:
                match = re.search(r"R([\d\.]+)", avg_file.name)
                if not match:
                    continue
                wound = float(match.group(1))

                structure.setdefault(lv, {}) \
                         .setdefault(domain, {}) \
                         .setdefault(force, {})[wound] = avg_file

if not structure:
    raise RuntimeError("No averaged data found.")

# ============================================================
# Helper: read mean and sem
# ============================================================

def read_average_file(avg_file):
    mean = sem = None
    with open(avg_file, "r") as f:
        for line in f:
            if line.startswith("mean_relative_closure_mcs"):
                mean = float(line.strip().split(",")[1])
            elif line.startswith("sem_relative_closure_mcs"):
                sem = float(line.strip().split(",")[1])
    if mean is None or sem is None:
        raise ValueError(f"No mean/sem found in {avg_file}")
    return mean, sem

# ============================================================
# FIGURE 1: Closure vs domain size (multi-panel: rows=λᵥ, cols=force)
# ============================================================

all_lvs = sorted(structure.keys())
all_forces = sorted({f for lv_dict in structure.values()
                          for d_dict in lv_dict.values()
                          for f in d_dict})

nrows = len(all_lvs)
ncols = len(all_forces)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 4 * nrows),
    sharex=True, sharey=True
)
axes = np.atleast_2d(axes)

for i, lv in enumerate(all_lvs):
    domain_dict = structure[lv]
    for j, force in enumerate(all_forces):
        ax = axes[i, j]
        wound_fracs = sorted({
            w
            for d in domain_dict.values()
            for f, wd in d.items() if f == force
            for w in wd
        })
        has_data = False
        for w in wound_fracs:
            xs, ys, es = [], [], []
            for domain, fdict in domain_dict.items():
                if force not in fdict or w not in fdict[force]:
                    continue
                mean, sem = read_average_file(fdict[force][w])
                xs.append(domain)
                ys.append(mean)
                es.append(sem)
            if xs:
                xs, ys, es = map(np.array, zip(*sorted(zip(xs, ys, es))))
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                            label=f"R={w}%")
                has_data = True
        ax.set_title(f"λᵥ={lv}, Force={force}")
        ax.set_xlabel("Domain size (px)")
        ax.set_ylabel("Mean closure time (mcs)")
        ax.grid(alpha=0.3)
        if has_data:
            ax.legend(title="Wound fraction")

fig.tight_layout()
fig_path = AVG_ROOT / "closure_vs_domain.png"
if fig_path.exists():
    fig_path.unlink()
    print("    Deleted old figure")
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"    Saved closure_vs_domain.png")



# ============================================================
# FIGURE 2: Closure vs force (multi-panel: rows=domain, cols=λᵥ)
# ============================================================

all_domains = sorted({domain for lv_dict in structure.values() for domain in lv_dict.keys()})
all_lvs = sorted(structure.keys())

nrows = len(all_domains)
ncols = len(all_lvs)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 4 * nrows),
    sharex=True, sharey=True
)
axes = np.atleast_2d(axes)

for i, domain in enumerate(all_domains):
    for j, lv in enumerate(all_lvs):
        ax = axes[i, j]
        if lv not in structure or domain not in structure[lv]:
            continue
        fdict = structure[lv][domain]
        wound_fracs = sorted({w for wd in fdict.values() for w in wd})
        has_data = False
        for w in wound_fracs:
            xs, ys, es = [], [], []
            for force in sorted(fdict.keys()):
                if w not in fdict[force]:
                    continue
                mean, sem = read_average_file(fdict[force][w])
                xs.append(force)
                ys.append(mean)
                es.append(sem)
            if xs:
                xs, ys, es = map(np.array, zip(*sorted(zip(xs, ys, es))))
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                            label=f"R={w}%")
                has_data = True
        ax.set_title(f"Domain={domain} px, λᵥ={lv}")
        ax.set_xlabel("Force")
        ax.set_ylabel("Mean closure time (mcs)")
        ax.grid(alpha=0.3)
        if has_data:
            ax.legend(title="Wound fraction")

fig.tight_layout()
fig_path = AVG_ROOT / "closure_vs_force.png"
if fig_path.exists():
    fig_path.unlink()
    print("    Deleted old figure")
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"    Saved closure_vs_force.png")


# ============================================================
# FIGURE 3: Closure vs wound fraction (multi-panel: rows=λᵥ, cols=domain)
# ============================================================

all_domains = sorted({domain for lv_dict in structure.values() for domain in lv_dict.keys()})
all_lvs = sorted(structure.keys())

nrows = len(all_lvs)
ncols = len(all_domains)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 4 * nrows),
    sharex=True, sharey=True
)
axes = np.atleast_2d(axes)

for i, lv in enumerate(all_lvs):
    for j, domain in enumerate(all_domains):
        ax = axes[i, j]
        if domain not in structure[lv]:
            continue
        fdict = structure[lv][domain]
        has_data = False
        for force in sorted(fdict.keys()):
            xs, ys, es = [], [], []
            for w, file in sorted(fdict[force].items()):
                mean, sem = read_average_file(file)
                xs.append(w)
                ys.append(mean)
                es.append(sem)
            if xs:
                xs, ys, es = map(np.array, zip(*sorted(zip(xs, ys, es))))
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                            label=f"Force={force}")
                has_data = True
        ax.set_title(f"λᵥ={lv}, Domain={domain} px")
        ax.set_xlabel("Wound radius (%)")
        ax.set_ylabel("Mean closure time (mcs)")
        ax.grid(alpha=0.3)
        if has_data:
            ax.legend(title="Force")

fig.tight_layout()
fig_path = AVG_ROOT / "closure_vs_wound_fraction.png"
if fig_path.exists():
    fig_path.unlink()
    print("    Deleted old figure")
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"    Saved closure_vs_wound_fraction.png")

print("\nAll multi-panel closure-time figures generated.")
