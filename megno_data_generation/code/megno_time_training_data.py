"""
Dense MEGNO(a,e) *temporal* training/validation data for the custom
Fourier-conditioned decoder (Custom_MEGNO_Model/train_time.py) -- the direct
temporal analog of megno_training_data.py's dense-MA sweep.

Where the spatial case fixes a single 50-year snapshot and sweeps MA densely
(1-degree steps, half-degree held out), this fixes a single MA (the standard
Greek-case offset, 60deg) and sweeps time densely instead: one continuous
100-year integration per (a,e) point, checkpointed every half year (200
checkpoints total). Half-integer years (0.5, 1.5, ..., 99.5) are held out for
validation -- the temporal analog of "half-degree offsets never seen during
training", a genuine interpolation test rather than memorization.

Checkpoint density is nearly free (see megno_time_evolution_all_ma.py's
docstring/comments): this is a SINGLE MA value, so the whole 200-checkpoint,
512x1024 grid finishes in well under a minute, unlike the many-MA sweeps.
"""

import os
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_timeseries_grid

BASELINE_DATE = "2012-09-30 00:00"
MA_CONFIG_DEG = float(os.environ.get("MEGNO_TIME_MA", "60"))

SMA_MIN, SMA_MAX = 5, 7
ECC_MIN, ECC_MAX = 0.001, 0.9
N_GRID_SMA = int(os.environ.get("MEGNO_N_SMA", "1024"))
N_GRID_ECC = int(os.environ.get("MEGNO_N_ECC", "512"))
INTEGRATOR = "whfast"
N_JOBS = 16
OUT_SUFFIX = os.environ.get("MEGNO_OUT_SUFFIX", f"_{N_GRID_SMA}x{N_GRID_ECC}")
TIMEOUT_SECONDS = int(os.environ.get("MEGNO_TIMEOUT_SECONDS", "300"))

OUT_DIR = "../Data/MEGNO_maps"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    simfile, MJD, jorb = build_sunjup_sim(BASELINE_DATE)
    print(f"Sun+Jupiter built at {BASELINE_DATE} (MJD {MJD}); Jupiter orbit: {jorb}")
    print(f"MA_CONFIG_DEG={MA_CONFIG_DEG}, grid {N_GRID_SMA}x{N_GRID_ECC}")

    all_years = np.round(np.arange(0.5, 100.5, 0.5), 1)  # 0.5, 1.0, ..., 100.0 (200 values)
    print(f"{len(all_years)} checkpoint years: {all_years[0]}..{all_years[-1]} (step 0.5yr)")

    import time as _time
    t0 = _time.time()
    _, _, MEGNO_stack = compute_megno_timeseries_grid(
        simfile, SMA_lin, ECC_lin, all_years, MA_config_deg=MA_CONFIG_DEG,
        integrator=INTEGRATOR, n_jobs=N_JOBS, timeout_seconds=TIMEOUT_SECONDS,
    )
    print(f"Computed {MEGNO_stack.shape} in {_time.time() - t0:.1f}s, "
          f"nan={int(np.isnan(MEGNO_stack).sum())}/{MEGNO_stack.size}")

    is_integer = np.isclose(all_years, np.round(all_years))
    train_idx = np.where(is_integer)[0]
    val_idx = np.where(~is_integer)[0]

    train_path = os.path.join(OUT_DIR, f"megno_time_training_set{OUT_SUFFIX}.npz")
    val_path = os.path.join(OUT_DIR, f"megno_time_validation_set{OUT_SUFFIX}.npz")
    np.savez(train_path, MEGNO_stack=MEGNO_stack[train_idx], year=all_years[train_idx],
             SMA_lin=SMA_lin, ECC_lin=ECC_lin, MA_config_deg=MA_CONFIG_DEG)
    np.savez(val_path, MEGNO_stack=MEGNO_stack[val_idx], year=all_years[val_idx],
             SMA_lin=SMA_lin, ECC_lin=ECC_lin, MA_config_deg=MA_CONFIG_DEG)
    print(f"Saved train {MEGNO_stack[train_idx].shape} to {train_path}")
    print(f"Saved val {MEGNO_stack[val_idx].shape} to {val_path}")


if __name__ == "__main__":
    main()
