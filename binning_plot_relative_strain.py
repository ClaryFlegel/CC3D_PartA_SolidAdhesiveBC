from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import re

# ============================================================
# User-defined parameters
# ============================================================

V_t = 100  # target volume
BIN_WIDTH = int(np.sqrt(V_t))

RUNS_ROOT = Path("SolidRuns")
AVG_ROOT = RUNS_ROOT / "Averages"

# ---------------- Plotting toggles ----------------
PLOT_FROM_WOUND_ONLY = True     # crop plot to wound MCS onward
SAVE_BOTH_PLOTS = True         # save both full + cropped plots
PLOT_WOUND_LINE = True         # vertical line at wound MCS (full plot only)

# ============================================================
# Helper functions
# ============================================================

def parse_domain_and_radius(path):
    lxly_match = re.search(r"Lx(\d+)_Ly(\d+)", path.as_posix())
    r_match = re.search(r"/R(\d+)", path.as_posix())
    if lxly_match is None or r_match is None:
        return None
    return (
        int(lxly_match.group(1)),
        int(lxly_match.group(2)),
        int(r_match.group(1)),
    )


def read_wound_mcs_from_header(path):
    """
    Reads wound creation MCS from a header line:
    # Wound created at mcs = <int>
    """
    with path.open("r") as f:
        for line in f:
            if line.startswith("#") and " Wound created at mcs" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    return int(match.group(1))
            elif not line.startswith("#"):
                break
    return None


def make_plot(
    strain_matrix,
    unique_mcs,
    bin_edges,
    domain_name,
    wound_name,
    data_file_name,
    out_path,
    vmax,
    wound_mcs=None,
    show_wound_line=False,
    title_suffix="",
):
    cmap = plt.cm.seismic.copy()
    cmap.set_bad(color="black")

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        strain_matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[
            unique_mcs.min(),
            unique_mcs.max(),
            bin_edges[0],
            bin_edges[-1],
        ],
    )

    if show_wound_line and wound_mcs is not None:
        ax.axvline(
            wound_mcs,
            color="red",
            linestyle="-",
            linewidth=1.5,
            label="Wound creation",
        )
        ax.legend(loc="upper right")

    ax.set_xlabel("MCS")
    ax.set_ylabel("Radial distance (pixels)")
    ax.set_title(
        f"Relative strain {title_suffix}\n"
        f"{domain_name}, {wound_name}, {data_file_name}"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean relative strain")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ============================================================
# Main processing loop
# ============================================================

for lxly_dir in RUNS_ROOT.glob("Lx*_Ly*"):

    print(f"\nProcessing domain {lxly_dir.name}")

    for r_dir in lxly_dir.glob("R*"):
        print(f"  Processing {r_dir.name}")

        for p_dir in r_dir.iterdir():
            if not p_dir.is_dir():
                continue

            print(f"    Processing {p_dir.name}")

            parsed = parse_domain_and_radius(r_dir)
            if parsed is None:
                continue

            Lx, Ly, R = parsed
            domain_name = lxly_dir.name
            wound_name = r_dir.name
            param_name = p_dir.name

            for data_file in p_dir.glob("cell_field_data_*.txt"):

                rows = []
                invalid_file = False

                wound_mcs = read_wound_mcs_from_header(data_file)

                with data_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        parts = line.split(",")

                        if len(parts) <= 5:
                            invalid_file = True
                            break

                        try:
                            mcs = int(parts[0])
                            volume = float(parts[4])
                            radial_distance = float(parts[5])
                        except ValueError:
                            invalid_file = True
                            break

                        rows.append((mcs, volume, radial_distance))

                if invalid_file or len(rows) == 0:
                    print(f"    Skipped {data_file.name} (invalid or empty)")
                    continue

                rows = np.array(rows)
                mcs_values = rows[:, 0].astype(int)
                volumes = rows[:, 1]
                radial_distances = rows[:, 2]

                relative_strain = (volumes - V_t) / V_t
                unique_mcs = np.unique(mcs_values)

                max_radius = int(Lx) / 2
                n_bins = int(np.floor(max_radius / BIN_WIDTH)) + 1
                bin_edges = np.arange(0, (n_bins + 1) * BIN_WIDTH, BIN_WIDTH)

                strain_matrix = np.full((n_bins, len(unique_mcs)), np.nan)

                for j, mcs in enumerate(unique_mcs):
                    mask_mcs = mcs_values == mcs
                    r_mcs = radial_distances[mask_mcs]
                    strain_mcs = relative_strain[mask_mcs]
                    bin_indices = np.floor(r_mcs / BIN_WIDTH).astype(int)

                    for b in range(n_bins):
                        mask_bin = bin_indices == b
                        if np.any(mask_bin):
                            strain_matrix[b, j] = np.mean(strain_mcs[mask_bin])

                # ------------------------------------------------------------
                # Shared color scale (per file)
                # ------------------------------------------------------------

                vmax_global = np.nanmax(np.abs(strain_matrix))

                if vmax_global == 0 or np.isnan(vmax_global):
                    print(f"    Warning: zero or NaN strain range in {data_file.name}")
                    continue

                # ------------------------------------------------------------
                # Output directory
                # ------------------------------------------------------------

                out_dir = AVG_ROOT / domain_name / wound_name / param_name / "Bins"
                out_dir.mkdir(parents=True, exist_ok=True)

                # ------------------------------------------------------------
                # Save averages file (always full data)
                # ------------------------------------------------------------

                out_file = out_dir / data_file.name.replace(".txt", "_bin_average.txt")

                with out_file.open("w") as f:
                    f.write("# Bin-averaged relative strain\n")
                    f.write("# Rows: radial distance bins\n")
                    f.write("# Columns: mcs values\n")
                    f.write(f"# Bin width = {BIN_WIDTH}\n")
                    f.write(f"# Domain Size: Lx={Lx}, Ly={Ly}\n")
                    f.write(f"# Wound Radius: R={R} % of domain size\n")
                    f.write(f"# Wound created at mcs = {wound_mcs}\n")
                    f.write(f"# V_t = {V_t}\n")
                    f.write("# mcs values:\n")
                    f.write("# " + " ".join(map(str, unique_mcs)) + "\n")
                    np.savetxt(f, strain_matrix, fmt="%.6e")

                # ------------------------------------------------------------
                # Plot full MCS range
                # ------------------------------------------------------------

                if not PLOT_FROM_WOUND_ONLY or SAVE_BOTH_PLOTS:
                    full_plot_path = out_dir / data_file.name.replace(
                        ".txt", "_bin_average_full.png"
                    )

                    make_plot(
                        strain_matrix,
                        unique_mcs,
                        bin_edges,
                        domain_name,
                        wound_name,
                        data_file.name,
                        full_plot_path,
                        vmax=vmax_global,
                        wound_mcs=wound_mcs,
                        show_wound_line=PLOT_WOUND_LINE,
                        title_suffix="(All MCS)",
                    )

                # ------------------------------------------------------------
                # Plot wound-onward only
                # ------------------------------------------------------------

                if PLOT_FROM_WOUND_ONLY and wound_mcs is not None:
                    mask = unique_mcs >= wound_mcs

                    if np.any(mask):
                        cropped_plot_path = out_dir / data_file.name.replace(
                            ".txt", "_bin_average_from_wound.png"
                        )

                        make_plot(
                            strain_matrix[:, mask],
                            unique_mcs[mask],
                            bin_edges,
                            domain_name,
                            wound_name,
                            data_file.name,
                            cropped_plot_path,
                            vmax=vmax_global,
                            wound_mcs=wound_mcs,
                            show_wound_line=False,
                            title_suffix=f"(MCS ≥ {wound_mcs})",
                        )

            print(f"      Saved all files and plots in → {domain_name}_{wound_name}_{param_name}_Bins")
