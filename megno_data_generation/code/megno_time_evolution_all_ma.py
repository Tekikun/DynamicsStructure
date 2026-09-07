"""
Genuine MEGNO time evolution (single continuous 100-year integration per
(a,e) point, checkpointed at 10,20,...,100 yr) for all 16 Greek-case MA
values. Verified correct for MA=60 in megno_time_evolution_ma60.py.
"""

import os
import time
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_timeseries_grid

BASELINE_DATE = "2012-09-30 00:00"
MA_VALUES = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480]
CHECKPOINT_YEARS = list(range(10, 101, 10))

SMA_MIN, SMA_MAX = 5, 7
ECC_MIN, ECC_MAX = 0.001, 0.9
N_GRID_SMA = int(os.environ.get("MEGNO_N_SMA", "32"))
N_GRID_ECC = int(os.environ.get("MEGNO_N_ECC", "32"))
INTEGRATOR = "whfast"
N_JOBS = 16
OUT_SUFFIX = os.environ.get("MEGNO_OUT_SUFFIX", "_32x32")
TIMEOUT_SECONDS = int(os.environ.get("MEGNO_TIMEOUT_SECONDS", "0")) or None

OUT_DIR = "../Data/MEGNO_maps"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    simfile, MJD, jorb = build_sunjup_sim(BASELINE_DATE)
    print(f"Sun+Jupiter built at {BASELINE_DATE} (MJD {MJD}); Jupiter orbit: {jorb}")
    print(f"MA values: {MA_VALUES}")
    print(f"Checkpoint years: {CHECKPOINT_YEARS}")

    all_stack = np.zeros((len(MA_VALUES), len(CHECKPOINT_YEARS), N_GRID_ECC, N_GRID_SMA))
    t0 = time.time()
    for mi, ma in enumerate(MA_VALUES):
        _, _, MEGNO_stack = compute_megno_timeseries_grid(
            simfile, SMA_lin, ECC_lin, CHECKPOINT_YEARS, MA_config_deg=ma,
            integrator=INTEGRATOR, n_jobs=N_JOBS, timeout_seconds=TIMEOUT_SECONDS,
        )
        all_stack[mi] = MEGNO_stack
        n_nan = int(np.isnan(MEGNO_stack).sum())
        print(f"MA={ma:>3d}deg  nan={n_nan}/{MEGNO_stack.size}  ({time.time() - t0:.1f}s elapsed)",
              flush=True)

    out_path = os.path.join(OUT_DIR, f"megno_time_evolution_all_ma{OUT_SUFFIX}.npz")
    np.savez(out_path, MEGNO_stack=all_stack, ma_values=np.array(MA_VALUES),
             checkpoint_years=np.array(CHECKPOINT_YEARS), SMA_lin=SMA_lin, ECC_lin=ECC_lin)
    print(f"Saved {all_stack.shape} to {out_path} in {time.time() - t0:.1f}s")
    print(f"nan total: {int(np.isnan(all_stack).sum())}/{all_stack.size}")


if __name__ == "__main__":
    main()
