"""
Verify whether the Sun-Jupiter Greek-cloud MEGNO(a,e) map is periodic with
Jupiter's orbital period (~11.86 yr).

Method: build the Sun+Jupiter system fresh (real NASA Horizons ephemeris,
independent Sun+Jupiter fetch per epoch, not a two-body extrapolation of one
baseline) at:
  - "aligned" epochs: baseline + k * P_Jup, for k = 0,1,2,3
  - "misaligned" control epochs: baseline + (k + 0.25/0.5/0.75) * P_Jup

If the periodicity hypothesis holds, MEGNO maps at aligned epochs should be
much more similar to the baseline map (high correlation, low RMSE) than maps
at misaligned epochs.
"""

import os
import csv
import time
import datetime

import numpy as np
from scipy.stats import pearsonr

from megno_map import build_sunjup_sim, compute_megno_grid, save_megno_grid, plot_megno_grid

BASELINE_DATE = "2012-09-30 00:00"
SMA_MIN, SMA_MAX = 5, 7
ECC_MIN, ECC_MAX = 0.00001, 0.99999
N_GRID_SMA = 100
N_GRID_ECC = 40
TSPAN_YEARS = 50
INTEGRATOR = "mercurius"
N_JOBS = 16

OUT_DIR = "../Data/MEGNO_periodicity"

ALIGNED_K = [0, 1, 2, 3]
MISALIGNED_K = [0.25, 0.5, 0.75, 1.25, 1.5]


def date_plus_years(date_str, years):
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    d2 = d + datetime.timedelta(days=years * 365.25)
    return d2.strftime("%Y-%m-%d %H:%M")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID)

    t0 = time.time()
    simfile0, MJD0, jorb0 = build_sunjup_sim(BASELINE_DATE)
    P_years = jorb0.P / (2 * np.pi)
    print(f"Jupiter orbit at baseline: {jorb0}")
    print(f"Jupiter period at baseline epoch: {P_years:.4f} yr")

    all_k = ALIGNED_K + MISALIGNED_K
    maps = {}
    for k in all_k:
        years = k * P_years
        date_str = BASELINE_DATE if k == 0 else date_plus_years(BASELINE_DATE, years)
        t_epoch = time.time()
        simfile, MJD, jorb = build_sunjup_sim(date_str)
        SMA_grid, ECC_grid, MEGNO_grid = compute_megno_grid(
            simfile, SMA_lin, ECC_lin, tspan_years=TSPAN_YEARS,
            integrator=INTEGRATOR, n_jobs=N_JOBS,
        )
        maps[k] = (MJD, SMA_grid, ECC_grid, MEGNO_grid)
        save_megno_grid(
            os.path.join(OUT_DIR, f"megno_k{k:.2f}_MJD{MJD:.1f}.npz"),
            SMA_grid, ECC_grid, MEGNO_grid, MJD, TSPAN_YEARS, INTEGRATOR, 60,
        )
        n_nan = int(np.isnan(MEGNO_grid).sum())
        print(f"k={k:>5.2f}  date={date_str}  MJD={MJD:.1f}  "
              f"nan_frac={n_nan}/{MEGNO_grid.size}  "
              f"({time.time() - t_epoch:.1f}s)")

    base_vals = maps[0][3].flatten()
    base_mask = np.isfinite(base_vals)

    results = []
    for k in all_k:
        vals = maps[k][3].flatten()
        mask = base_mask & np.isfinite(vals)
        if mask.sum() < 2:
            corr, rmse = np.nan, np.nan
        else:
            corr, _ = pearsonr(base_vals[mask], vals[mask])
            rmse = float(np.sqrt(np.mean((base_vals[mask] - vals[mask]) ** 2)))
        kind = "aligned" if k in ALIGNED_K else "misaligned"
        results.append((k, k * P_years, kind, corr, rmse))
        print(f"k={k:>5.2f} [{kind:10s}] corr_vs_baseline={corr:.4f}  rmse={rmse:.4f}")

    summary_path = os.path.join(OUT_DIR, "periodicity_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "offset_years", "kind", "corr_vs_baseline", "rmse_vs_baseline"])
        w.writerows(results)
    print("Saved summary to", summary_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for kind, color in [("aligned", "tab:blue"), ("misaligned", "tab:red")]:
        pts = sorted((off, corr) for (k, off, kd, corr, rmse) in results if kd == kind)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=color, label=kind)
    for k in ALIGNED_K:
        ax.axvline(k * P_years, color="tab:blue", linestyle=":", alpha=0.3)
    ax.set_xlabel("Epoch offset from baseline (years)")
    ax.set_ylabel("Pearson correlation of MEGNO map vs. baseline")
    ax.set_title(f"MEGNO(a,e) map similarity vs. epoch offset (P_Jup={P_years:.2f} yr)")
    ax.legend()
    fig.tight_layout()
    corr_plot_path = os.path.join(OUT_DIR, "periodicity_correlation.png")
    fig.savefig(corr_plot_path, dpi=150)
    print("Saved plot to", corr_plot_path)

    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, k, label in zip(axes, [0, 1, 0.5],
                             ["baseline (k=0)", "aligned k=1", "misaligned k=0.5"]):
        _, SMA_grid, ECC_grid, MEGNO_grid = maps[k]
        plot_megno_grid(SMA_grid, ECC_grid, MEGNO_grid, title=label, ax=ax)
    fig2.tight_layout()
    heatmap_path = os.path.join(OUT_DIR, "periodicity_heatmaps.png")
    fig2.savefig(heatmap_path, dpi=150)
    print("Saved heatmaps to", heatmap_path)

    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
