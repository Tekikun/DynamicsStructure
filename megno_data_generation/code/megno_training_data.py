"""
Dense MEGNO(a,e) training/validation data for the custom Fourier-conditioned
decoder model (Custom_MEGNO_Model/). Since the map is exactly periodic in MA
offset mod 360 deg (confirmed in megno_ma_sweep.py), one full period at fine
resolution is sufficient ground truth -- no need for multiple periods.

- Training set: integer degrees 0..359 (360 frames).
- Validation set: half-degree offsets 0.5..359.5 (360 frames), never seen
  during training -- tests genuine interpolation, not memorization.
"""

import os
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_grid

import os as _os

BASELINE_DATE = "2012-09-30 00:00"

SMA_MIN, SMA_MAX, N_GRID_SMA = 5, 7, int(_os.environ.get("MEGNO_N_SMA", "32"))
ECC_MIN, ECC_MAX, N_GRID_ECC = 0.001, 0.9, int(_os.environ.get("MEGNO_N_ECC", "32"))
TSPAN_YEARS = 50
INTEGRATOR = "whfast"
N_JOBS = 16
OUT_SUFFIX = _os.environ.get("MEGNO_OUT_SUFFIX", "")  # e.g. "_1024x512"

OUT_DIR = "../Data/MEGNO_maps"


TIMEOUT_SECONDS = int(_os.environ.get("MEGNO_TIMEOUT_SECONDS", "60"))


CHECKPOINT_EVERY = int(_os.environ.get("MEGNO_CHECKPOINT_EVERY", "10"))


def generate_set(simfile, SMA_lin, ECC_lin, ma_values, out_path):
    import time as _time
    if os.path.exists(out_path):
        print(f"Skipping {out_path} (already exists)")
        return

    ckpt_path = out_path + ".checkpoint.npz"
    frames = []
    start_i = 0
    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        frames = list(ckpt["MEGNO_stack"])
        start_i = len(frames)
        assert np.allclose(ckpt["MA_deg"], ma_values[:start_i]), "checkpoint MA values don't match"
        print(f"Resuming from checkpoint: {start_i}/{len(ma_values)} frames already done")

    for i in range(start_i, len(ma_values)):
        ma = ma_values[i]
        t0 = _time.time()
        _, _, MEGNO_grid = compute_megno_grid(
            simfile, SMA_lin, ECC_lin, MA_config_deg=ma,
            tspan_years=TSPAN_YEARS, integrator=INTEGRATOR, n_jobs=N_JOBS,
            timeout_seconds=TIMEOUT_SECONDS,
        )
        frames.append(MEGNO_grid)
        print(f"  MA={ma:.1f}deg done in {_time.time() - t0:.1f}s "
              f"(nan={int(np.isnan(MEGNO_grid).sum())}/{MEGNO_grid.size})", flush=True)

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(ma_values) - 1:
            np.savez(ckpt_path, MEGNO_stack=np.stack(frames, axis=0),
                     MA_deg=np.array(ma_values[:i + 1]))

    stack = np.stack(frames, axis=0)
    np.savez(out_path, MEGNO_stack=stack, MA_deg=np.array(ma_values),
             SMA_lin=SMA_lin, ECC_lin=ECC_lin)
    n_nan = int(np.isnan(stack).sum())
    print(f"Saved {stack.shape} to {out_path} (nan={n_nan}/{stack.size}, "
          f"min={np.nanmin(stack):.2f}, max={np.nanmax(stack):.2f})")
    os.remove(ckpt_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    simfile, MJD, jorb = build_sunjup_sim(BASELINE_DATE)
    print(f"Sun+Jupiter built at {BASELINE_DATE} (MJD {MJD}); Jupiter orbit: {jorb}")

    train_ma = np.arange(0, 360, 1).astype(float)
    val_ma = np.arange(0, 360, 1).astype(float) + 0.5

    generate_set(simfile, SMA_lin, ECC_lin, train_ma,
                 os.path.join(OUT_DIR, f"megno_training_set{OUT_SUFFIX}.npz"))
    generate_set(simfile, SMA_lin, ECC_lin, val_ma,
                 os.path.join(OUT_DIR, f"megno_validation_set{OUT_SUFFIX}.npz"))


if __name__ == "__main__":
    main()
