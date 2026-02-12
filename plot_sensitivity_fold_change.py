from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import warnings

# ============================================================
# Configuration
# ============================================================

AVG_ROOT = Path("SolidRuns/Averages")

DEFAULT_DOMAIN = 250
DEFAULT_R = 30
DEFAULT_LV = 0.5
DEFAULT_FORCE = 1200

# ============================================================
# Regex
# ============================================================

DOMAIN_RE = re.compile(r"Lx(\d+)_Ly\1")
WOUND_RE  = re.compile(r"R([\d\.]+)")
PARAM_RE  = re.compile(r"lv([\d\.]+)_f([\d\.]+)")

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
# Discover data structure
# structure[(domain, R, lv, force)] = avg_file
# ============================================================

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
        R = float(wmatch.group(1))

        for param_dir in wound_dir.iterdir():
            if not param_dir.is_dir():
                continue

            pmatch = PARAM_RE.match(param_dir.name)
            if not pmatch:
                continue

            lv = float(pmatch.group(1))
            force = float(pmatch.group(2))

            avg_files = list(param_dir.glob("*_avg.txt"))
            if not avg_files:
                continue

            # Expect exactly one avg file per folder
            avg_file = avg_files[0]
            structure[(domain, R, lv, force)] = avg_file

if not structure:
    raise RuntimeError("No averaged data found.")

# ============================================================
# Default reference
# ============================================================

default_key = (DEFAULT_DOMAIN, DEFAULT_R, DEFAULT_LV, DEFAULT_FORCE)

if default_key not in structure:
    raise RuntimeError(
        f"Default simulation not found: {default_key}"
    )

default_mean, default_sem = read_average_file(structure[default_key])

# ============================================================
# Helper: fold change + propagated SEM
# ============================================================

def fold_change(mean, sem, ref_mean, ref_sem):
    # Raw fold change
    fc = mean / ref_mean

    # Propagated SEM (unchanged by subtraction)
    fc_sem = fc * math.sqrt(
        (sem / mean) ** 2 +
        (ref_sem / ref_mean) ** 2
    )

    # Shift so default = 0
    return fc - 1.0, fc_sem

# ============================================================
# Plot helper
# ============================================================

def make_bar_plot(xs, ys, es, xlabel, title, outfile):
    xs = np.array(xs)
    ys = np.array(ys)
    es = np.array(es)

    pos = np.arange(len(xs))  # categorical positions

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        pos, ys,
        yerr=es,
        capsize=5,
        width=0.6
    )

    ax.set_xticks(pos)
    #ax.set_xticklabels(xs)
    ax.set_xticklabels([str(x) for x in xs])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative change in closure time")
    ax.set_title(title)
    

    ax.axhline(0.0, linestyle="--", color="black", alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if outfile.exists():
        outfile.unlink()
        print(f"    Deleted old plot")
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    print(f"    Saved {outfile.name}")


# ============================================================
# 1) Force sensitivity
# ============================================================

xs, ys, es = [], [], []

for (domain, R, lv, force), avg_file in sorted(structure.items()):
    if (
        domain == DEFAULT_DOMAIN and
        R == DEFAULT_R and
        lv == DEFAULT_LV
    ):
        mean, sem = read_average_file(avg_file)
        fc, fc_sem = fold_change(mean, sem, default_mean, default_sem)
        xs.append(force)
        ys.append(fc)
        es.append(fc_sem)

if xs:
    xs, ys, es = map(np.array, zip(*sorted(zip(xs, ys, es))))
    make_bar_plot(
        xs, ys, es,
        xlabel="Force",
        title="Force sensitivity",
        outfile=AVG_ROOT / f"Sensitivity_force_Lx{DEFAULT_DOMAIN}_Ly{DEFAULT_DOMAIN}_R{DEFAULT_R}_lv{DEFAULT_LV}.png"
    )
else:
    warnings.warn("No data for force sensitivity")

# ============================================================
# 2) Lambda-volume sensitivity
# ============================================================

xs, ys, es = [], [], []

for (domain, R, lv, force), avg_file in sorted(structure.items()):
    if (
        domain == DEFAULT_DOMAIN and
        R == DEFAULT_R and
        force == DEFAULT_FORCE
    ):
        mean, sem = read_average_file(avg_file)
        fc, fc_sem = fold_change(mean, sem, default_mean, default_sem)
        xs.append(lv)
        ys.append(fc)
        es.append(fc_sem)

if xs:
    xs, ys, es = map(np.array, zip(*sorted(zip(xs, ys, es))))
    make_bar_plot(
        xs, ys, es,
        xlabel="λᵥ",
        title="Lambda-volume sensitivity",
        outfile=AVG_ROOT / f"Sensitivity_lambda_volume_Lx{DEFAULT_DOMAIN}_Ly{DEFAULT_DOMAIN}_R{DEFAULT_R}_f{DEFAULT_FORCE}.png"
    )
else:
    warnings.warn("No data for lambda-volume sensitivity")

# ============================================================
# 3) Wound radius sensitivity
# ============================================================

xs, ys, es = [], [], []

for (domain, R, lv, force), avg_file in sorted(structure.items()):
    if (
        domain == DEFAULT_DOMAIN and
        lv == DEFAULT_LV and
        force == DEFAULT_FORCE
    ):
        mean, sem = read_average_file(avg_file)
        fc, fc_sem = fold_change(mean, sem, default_mean, default_sem)
        xs.append(R)
        ys.append(fc)
        es.append(fc_sem)

if xs:
    xs, ys, es = map(np.array, zip(*sorted(zip(xs, ys, es))))
    make_bar_plot(
        xs, ys, es,
        xlabel="Wound radius",
        title="Wound radius sensitivity",
        outfile=AVG_ROOT / f"Sensitivity_wound_radius_Lx{DEFAULT_DOMAIN}_Ly{DEFAULT_DOMAIN}_lv{DEFAULT_LV}_f{DEFAULT_FORCE}.png"
    )
else:
    warnings.warn("No data for wound radius sensitivity")

# ============================================================
# 4) Domain size sensitivity
# ============================================================

xs, ys, es = [], [], []

for (domain, R, lv, force), avg_file in sorted(structure.items()):
    if (
        R == DEFAULT_R and
        lv == DEFAULT_LV and
        force == DEFAULT_FORCE
    ):
        mean, sem = read_average_file(avg_file)
        fc, fc_sem = fold_change(mean, sem, default_mean, default_sem)
        xs.append(domain)
        ys.append(fc)
        es.append(fc_sem)

if xs:
    xs, ys, es = map(np.array, zip(*sorted(zip(xs, ys, es))))
    make_bar_plot(
        xs, ys, es,
        xlabel="Domain size (px)",
        title="Domain size sensitivity",
        outfile=AVG_ROOT / f"Sensitivity_domain_size_R{DEFAULT_R}_lv{DEFAULT_LV}_f{DEFAULT_FORCE}.png"
    )
else:
    warnings.warn("No data for domain size sensitivity")

print("\nAll sensitivity plots generated.")
