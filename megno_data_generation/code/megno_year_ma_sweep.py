"""
2D (year x MA) sweep of the Greek-case MEGNO(a,e) map, all from the same
starting MJD/epoch. Unlike megno_ma_sweep_dense.py (which varies MA at a
FIXED epoch, revealing the exact 360-deg phase-space periodicity), this
varies the *epoch itself* by real calendar years while also sweeping MA at
each one -- testing whether the map "morphs" smoothly across years (real
ephemeris drift/perturbation accumulation), which might behave more like a
smoothly-evolving physical field than the MA axis does.

Resolution is configurable (kept at 32x32 by default -- small/cheap, and
already a valid Walrus resolution with no resizing needed).
"""

import os
import datetime
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_grid

BASELINE_DATE = "2012-09-30 00:00"
YEARS_OFFSET = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
MA_VALUES = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480]

SMA_MIN, SMA_MAX = 5, 7
ECC_MIN, ECC_MAX = 0.001, 0.9
N_GRID_SMA = int(os.environ.get("MEGNO_N_SMA", "32"))
N_GRID_ECC = int(os.environ.get("MEGNO_N_ECC", "32"))
TSPAN_YEARS = 50
INTEGRATOR = "whfast"
N_JOBS = 16
OUT_SUFFIX = os.environ.get("MEGNO_OUT_SUFFIX", "_32x32")
TIMEOUT_SECONDS = int(os.environ.get("MEGNO_TIMEOUT_SECONDS", "0")) or None

OUT_DIR = "../Data/MEGNO_maps"


def date_plus_years(date_str, years):
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    d2 = d + datetime.timedelta(days=years * 365.25)
    return d2.strftime("%Y-%m-%d %H:%M")


def main():
    import time
    os.makedirs(OUT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    grid = np.zeros((len(YEARS_OFFSET), len(MA_VALUES), N_GRID_ECC, N_GRID_SMA))
    t0 = time.time()
    for yi, yr in enumerate(YEARS_OFFSET):
        date_str = date_plus_years(BASELINE_DATE, yr)
        simfile, MJD, jorb = build_sunjup_sim(date_str)
        print(f"year+{yr:>3d}  date={date_str}  MJD={MJD:.1f}  a={jorb.a:.5f} e={jorb.e:.5f}",
              flush=True)
        for mi, ma in enumerate(MA_VALUES):
            _, _, MEGNO_grid = compute_megno_grid(
                simfile, SMA_lin, ECC_lin, MA_config_deg=ma,
                tspan_years=TSPAN_YEARS, integrator=INTEGRATOR, n_jobs=N_JOBS,
                timeout_seconds=TIMEOUT_SECONDS,
            )
            grid[yi, mi] = MEGNO_grid
            n_nan = int(np.isnan(MEGNO_grid).sum())
            print(f"  MA={ma:>3d}deg  nan={n_nan}/{MEGNO_grid.size}", flush=True)

    out_path = os.path.join(OUT_DIR, f"megno_year_ma_grid{OUT_SUFFIX}.npz")
    np.savez(out_path, MEGNO_grid=grid, years_offset=np.array(YEARS_OFFSET),
             ma_values=np.array(MA_VALUES), SMA_lin=SMA_lin, ECC_lin=ECC_lin)
    print(f"Saved {grid.shape} to {out_path} in {time.time() - t0:.1f}s")
    print(f"nan total: {int(np.isnan(grid).sum())}/{grid.size}")


if __name__ == "__main__":
    main()
