"""
Fine-grained MEGNO time-evolution dataset: 2-degree MA sampling over one full
360-degree period (exact periodicity already verified, so no need to repeat
the old 450-degree/16-value convention with its redundant 390-480deg
duplicate block), and 2-year checkpoint sampling from 2 to 100 years.

Checkpoint-safe: each MA value's result is written to its own .npy file in
CKPT_DIR as soon as it's computed. Re-running this script skips any MA value
whose file already exists, so a crash/kill partway through only costs the
in-progress MA value, not the whole run. Final combined dataset (float32, to
halve disk vs. the float64 megno_time_evolution_all_ma*.npz files) is
assembled from the per-MA files once all of them are present.
"""

import os
import time
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_timeseries_grid

BASELINE_DATE = "2012-09-30 00:00"
MA_STEP = int(os.environ.get("MEGNO_MA_STEP", "2"))
YEAR_STEP = int(os.environ.get("MEGNO_YEAR_STEP", "2"))
MA_VALUES = list(range(0, 360, MA_STEP))
CHECKPOINT_YEARS = list(range(YEAR_STEP, 101, YEAR_STEP))

SMA_MIN, SMA_MAX = 5, 7
ECC_MIN, ECC_MAX = 0.001, 0.9
N_GRID_SMA = int(os.environ.get("MEGNO_N_SMA", "1024"))
N_GRID_ECC = int(os.environ.get("MEGNO_N_ECC", "512"))
INTEGRATOR = "whfast"
N_JOBS = 16
TIMEOUT_SECONDS = int(os.environ.get("MEGNO_TIMEOUT_SECONDS", "300"))
OUT_SUFFIX = os.environ.get("MEGNO_OUT_SUFFIX", f"_finegrained_{N_GRID_SMA}x{N_GRID_ECC}")

OUT_DIR = "../Data/MEGNO_maps"
CKPT_DIR = os.path.join(OUT_DIR, f"megno_time_evolution_finegrained_ckpt_{N_GRID_SMA}x{N_GRID_ECC}")


def ckpt_path(ma):
    return os.path.join(CKPT_DIR, f"ma_{ma:04d}.npy")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    simfile, MJD, jorb = build_sunjup_sim(BASELINE_DATE)
    print(f"Sun+Jupiter built at {BASELINE_DATE} (MJD {MJD}); Jupiter orbit: {jorb}")
    print(f"{len(MA_VALUES)} MA values (step {MA_STEP}deg): {MA_VALUES[0]}..{MA_VALUES[-1]}")
    print(f"{len(CHECKPOINT_YEARS)} checkpoint years (step {YEAR_STEP}yr): "
          f"{CHECKPOINT_YEARS[0]}..{CHECKPOINT_YEARS[-1]}")
    print(f"grid {N_GRID_SMA}x{N_GRID_ECC}, checkpoint dir: {CKPT_DIR}")

    already_done = [ma for ma in MA_VALUES if os.path.exists(ckpt_path(ma))]
    todo = [ma for ma in MA_VALUES if ma not in already_done]
    print(f"{len(already_done)}/{len(MA_VALUES)} MA values already checkpointed, "
          f"{len(todo)} remaining")

    t0 = time.time()
    for i, ma in enumerate(todo):
        t_ma = time.time()
        _, _, MEGNO_stack = compute_megno_timeseries_grid(
            simfile, SMA_lin, ECC_lin, CHECKPOINT_YEARS, MA_config_deg=ma,
            integrator=INTEGRATOR, n_jobs=N_JOBS, timeout_seconds=TIMEOUT_SECONDS,
        )
        MEGNO_stack = MEGNO_stack.astype(np.float32)
        np.save(ckpt_path(ma), MEGNO_stack)
        n_nan = int(np.isnan(MEGNO_stack).sum())
        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        eta = rate * (len(todo) - i - 1)
        print(f"[{i + 1}/{len(todo)}] MA={ma:>3d}deg  nan={n_nan}/{MEGNO_stack.size}  "
              f"({time.time() - t_ma:.1f}s, {elapsed:.1f}s elapsed, ETA {eta / 60:.1f}min)",
              flush=True)

    missing = [ma for ma in MA_VALUES if not os.path.exists(ckpt_path(ma))]
    if missing:
        print(f"WARNING: {len(missing)} MA values still missing after run: {missing}")
        print("Re-run this script to fill in the rest (already-computed MA values are skipped).")
        return

    print("All MA values present, assembling final combined dataset...")
    all_stack = np.zeros((len(MA_VALUES), len(CHECKPOINT_YEARS), N_GRID_ECC, N_GRID_SMA),
                          dtype=np.float32)
    for mi, ma in enumerate(MA_VALUES):
        all_stack[mi] = np.load(ckpt_path(ma))

    out_path = os.path.join(OUT_DIR, f"megno_time_evolution_all_ma{OUT_SUFFIX}.npz")
    np.savez(out_path, MEGNO_stack=all_stack, ma_values=np.array(MA_VALUES),
              checkpoint_years=np.array(CHECKPOINT_YEARS), SMA_lin=SMA_lin, ECC_lin=ECC_lin)
    print(f"Saved {all_stack.shape} ({all_stack.nbytes / 1e9:.2f} GB) to {out_path} "
          f"in {time.time() - t0:.1f}s total")
    print(f"nan total: {int(np.isnan(all_stack).sum())}/{all_stack.size}")


if __name__ == "__main__":
    main()
