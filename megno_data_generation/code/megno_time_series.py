"""
Generate a sequence of MEGNO(a,e) maps for the Sun-Jupiter Greek-cloud
restricted 3-body problem, at the same (a,e) grid / integration settings as
single_megno_map.py, but at epochs offset from the baseline by
0, 2, 4, ..., 20 years.
"""

import os
import datetime
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_grid, save_megno_grid, plot_megno_grid

BASELINE_DATE = "2012-09-30 00:00"
OFFSET_YEARS = list(range(0, 21, 2))  # 0,2,4,...,20

SMA_MIN, SMA_MAX, N_GRID_SMA = 5, 7, 80
ECC_MIN, ECC_MAX, N_GRID_ECC = 0.001, 0.9, 40
TSPAN_YEARS = 50
INTEGRATOR = "whfast"
MA_CONFIG_DEG = 60
N_JOBS = 16

OUT_DIR = "../Data/MEGNO_maps"


def date_plus_years(date_str, years):
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    d2 = d + datetime.timedelta(days=years * 365.25)
    return d2.strftime("%Y-%m-%d %H:%M")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(OFFSET_YEARS)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, offset in enumerate(OFFSET_YEARS):
        date_str = BASELINE_DATE if offset == 0 else date_plus_years(BASELINE_DATE, offset)
        simfile, MJD, jorb = build_sunjup_sim(date_str)

        SMA_grid, ECC_grid, MEGNO_grid = compute_megno_grid(
            simfile, SMA_lin, ECC_lin,
            MA_config_deg=MA_CONFIG_DEG,
            tspan_years=TSPAN_YEARS,
            integrator=INTEGRATOR,
            n_jobs=N_JOBS,
        )

        n_nan = int(np.isnan(MEGNO_grid).sum())
        print(f"+{offset:>2d}yr  date={date_str}  MJD={MJD:.1f}  "
              f"nan={n_nan}/{MEGNO_grid.size}  "
              f"min={np.nanmin(MEGNO_grid):.2f}  max={np.nanmax(MEGNO_grid):.2f}")

        base = f"megno_offset{offset:02d}yr_MJD{MJD:.1f}_{N_GRID_SMA}x{N_GRID_ECC}"
        save_megno_grid(os.path.join(OUT_DIR, base + ".npz"),
                         SMA_grid, ECC_grid, MEGNO_grid, MJD, TSPAN_YEARS,
                         INTEGRATOR, MA_CONFIG_DEG)

        plot_megno_grid(SMA_grid, ECC_grid, MEGNO_grid,
                         title=f"+{offset} yr (MJD {MJD:.1f})", ax=axes[i])

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Greek-case MEGNO(a,e) vs. epoch offset "
                 f"(a in [{SMA_MIN},{SMA_MAX}] AU, e in [{ECC_MIN},{ECC_MAX}], "
                 f"{TSPAN_YEARS}yr, {INTEGRATOR})")
    fig.tight_layout()
    combined_path = os.path.join(OUT_DIR, "megno_time_series_grid.png")
    fig.savefig(combined_path, dpi=150)
    print("Saved combined figure to", combined_path)


if __name__ == "__main__":
    main()
