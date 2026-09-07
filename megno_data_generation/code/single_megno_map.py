"""
Generate a single MEGNO(a,e) map for the Sun-Jupiter Greek-cloud restricted
3-body problem at one epoch.

Test particles: same inc/Omega/omega as Jupiter, mean anomaly = Jupiter's
mean anomaly + 60 deg (Greek/L4-leading configuration), matching the
convention in Greek_ReBound_Sun_Jup.py.
"""

import os
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_grid, save_megno_grid, plot_megno_grid

BASELINE_DATE = "2012-09-30 00:00"

SMA_MIN, SMA_MAX, N_GRID_SMA = 5, 7, 80
ECC_MIN, ECC_MAX, N_GRID_ECC = 0.001, 0.9, 40
TSPAN_YEARS = 50
INTEGRATOR = "whfast"
MA_CONFIG_DEG = 60
N_JOBS = 16

OUT_DIR = "../Data/MEGNO_maps"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    simfile, MJD, jorb = build_sunjup_sim(BASELINE_DATE)
    print(f"Sun+Jupiter built at {BASELINE_DATE} (MJD {MJD})")
    print(f"Jupiter orbit: {jorb}")

    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    SMA_grid, ECC_grid, MEGNO_grid = compute_megno_grid(
        simfile, SMA_lin, ECC_lin,
        MA_config_deg=MA_CONFIG_DEG,
        tspan_years=TSPAN_YEARS,
        integrator=INTEGRATOR,
        n_jobs=N_JOBS,
    )

    n_nan = int(np.isnan(MEGNO_grid).sum())
    print(f"MEGNO grid: {MEGNO_grid.shape}, nan={n_nan}/{MEGNO_grid.size}, "
          f"min={np.nanmin(MEGNO_grid):.3f}, max={np.nanmax(MEGNO_grid):.3f}")

    out_path = os.path.join(OUT_DIR, f"megno_MJD{MJD:.1f}_{N_GRID_SMA}x{N_GRID_ECC}.npz")
    save_megno_grid(out_path, SMA_grid, ECC_grid, MEGNO_grid, MJD, TSPAN_YEARS,
                     INTEGRATOR, MA_CONFIG_DEG)
    print("Saved grid to", out_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = plot_megno_grid(SMA_grid, ECC_grid, MEGNO_grid,
                          title=f"Greek-case MEGNO(a,e), MJD {MJD:.1f}, {TSPAN_YEARS}yr, {INTEGRATOR}")
    fig_path = os.path.join(OUT_DIR, f"megno_MJD{MJD:.1f}_{N_GRID_SMA}x{N_GRID_ECC}.png")
    ax.figure.savefig(fig_path, dpi=150)
    print("Saved plot to", fig_path)


if __name__ == "__main__":
    main()
