"""
Verify genuine MEGNO time evolution (not epoch-restart) for the Greek case at
MA=60deg: ONE continuous 100-year integration per (a,e) grid point, sampling
the running MEGNO value at 10,20,...,100 years.
"""

import os
import time
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_timeseries_grid, plot_megno_grid

BASELINE_DATE = "2012-09-30 00:00"
MA_VALUE = 60

CHECKPOINT_YEARS = list(range(10, 101, 10))  # 10,20,...,100

SMA_MIN, SMA_MAX, N_GRID_SMA = 5, 7, 32
ECC_MIN, ECC_MAX, N_GRID_ECC = 0.001, 0.9, 32
INTEGRATOR = "whfast"
N_JOBS = 16

OUT_DIR = "../Data/MEGNO_maps"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    simfile, MJD, jorb = build_sunjup_sim(BASELINE_DATE)
    print(f"Sun+Jupiter built at {BASELINE_DATE} (MJD {MJD}); Jupiter orbit: {jorb}")
    print(f"Checkpoint years: {CHECKPOINT_YEARS}")

    t0 = time.time()
    SMA_grid, ECC_grid, MEGNO_stack = compute_megno_timeseries_grid(
        simfile, SMA_lin, ECC_lin, CHECKPOINT_YEARS, MA_config_deg=MA_VALUE,
        integrator=INTEGRATOR, n_jobs=N_JOBS,
    )
    print(f"Computed in {time.time() - t0:.1f}s, stack shape {MEGNO_stack.shape}")

    for i, yr in enumerate(CHECKPOINT_YEARS):
        frame = MEGNO_stack[i]
        n_nan = int(np.isnan(frame).sum())
        print(f"  t={yr:>3d}yr  nan={n_nan}/{frame.size}  "
              f"min={np.nanmin(frame):.2f}  max={np.nanmax(frame):.2f}  "
              f"mean={np.nanmean(frame):.3f}")

    out_path = os.path.join(OUT_DIR, f"megno_timeevolution_MA{MA_VALUE}_32x32.npz")
    np.savez(out_path, MEGNO_stack=MEGNO_stack, checkpoint_years=np.array(CHECKPOINT_YEARS),
             SMA_lin=SMA_lin, ECC_lin=ECC_lin, MA_value=MA_VALUE)
    print("Saved to", out_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(CHECKPOINT_YEARS)
    ncols = 5
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, yr in enumerate(CHECKPOINT_YEARS):
        plot_megno_grid(SMA_grid, ECC_grid, MEGNO_stack[i], title=f"t={yr}yr", ax=axes[i])
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"MEGNO(a,e) time evolution, Greek case MA={MA_VALUE}deg "
                 f"(single continuous integration per point)")
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, f"megno_timeevolution_MA{MA_VALUE}_32x32.png")
    fig.savefig(fig_path, dpi=150)
    print("Saved figure to", fig_path)


if __name__ == "__main__":
    main()
