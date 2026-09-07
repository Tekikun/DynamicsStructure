"""
Generate a sequence of MEGNO(a,e) maps for the Sun-Jupiter co-orbital region
at a single fixed epoch, sweeping the test-particle mean-anomaly offset
relative to Jupiter (MA_config_deg) instead of the epoch. Same (a,e) grid /
integration settings as single_megno_map.py / megno_time_series.py.

MA_config_deg = 60 is the "Greek" (L4-leading) case used elsewhere in this
repo; 300 (= -60) would be the "Trojan" (L5-trailing) case; 0/360 co-locates
the test particles with Jupiter itself; 180 is near L3, opposite Jupiter.
"""

import os
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_grid, save_megno_grid, plot_megno_grid

BASELINE_DATE = "2012-09-30 00:00"
MA_OFFSETS_DEG = [60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
                  390, 420, 450, 480]

SMA_MIN, SMA_MAX, N_GRID_SMA = 5, 7, 80
ECC_MIN, ECC_MAX, N_GRID_ECC = 0.001, 0.9, 40
TSPAN_YEARS = 50
INTEGRATOR = "whfast"
N_JOBS = 16

OUT_DIR = "../Data/MEGNO_maps"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    simfile, MJD, jorb = build_sunjup_sim(BASELINE_DATE)
    print(f"Sun+Jupiter built at {BASELINE_DATE} (MJD {MJD})")
    print(f"Jupiter orbit: {jorb}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(MA_OFFSETS_DEG)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, ma_deg in enumerate(MA_OFFSETS_DEG):
        SMA_grid, ECC_grid, MEGNO_grid = compute_megno_grid(
            simfile, SMA_lin, ECC_lin,
            MA_config_deg=ma_deg,
            tspan_years=TSPAN_YEARS,
            integrator=INTEGRATOR,
            n_jobs=N_JOBS,
        )

        n_nan = int(np.isnan(MEGNO_grid).sum())
        print(f"MA+{ma_deg:>3d}deg  nan={n_nan}/{MEGNO_grid.size}  "
              f"min={np.nanmin(MEGNO_grid):.2f}  max={np.nanmax(MEGNO_grid):.2f}")

        base = f"megno_MJD{MJD:.1f}_MA{ma_deg:03d}deg_{N_GRID_SMA}x{N_GRID_ECC}"
        save_megno_grid(os.path.join(OUT_DIR, base + ".npz"),
                         SMA_grid, ECC_grid, MEGNO_grid, MJD, TSPAN_YEARS,
                         INTEGRATOR, ma_deg)

        plot_megno_grid(SMA_grid, ECC_grid, MEGNO_grid,
                         title=f"MA + {ma_deg} deg", ax=axes[i])

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"MEGNO(a,e) vs. mean-anomaly offset from Jupiter, "
                 f"fixed epoch MJD {MJD:.1f} "
                 f"(a in [{SMA_MIN},{SMA_MAX}] AU, e in [{ECC_MIN},{ECC_MAX}], "
                 f"{TSPAN_YEARS}yr, {INTEGRATOR})")
    fig.tight_layout()
    combined_path = os.path.join(OUT_DIR, "megno_ma_sweep_grid.png")
    fig.savefig(combined_path, dpi=150)
    print("Saved combined figure to", combined_path)


if __name__ == "__main__":
    main()
