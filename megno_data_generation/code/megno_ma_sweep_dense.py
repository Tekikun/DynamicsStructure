"""
Dense mean-anomaly sweep for the Sun-Jupiter co-orbital MEGNO(a,e) map, used
as the "video" sequence fed to Walrus: one fixed epoch, same (a,e) grid as
single_megno_map.py, stepping MA offset from Jupiter in fine increments.

Confirmed in megno_ma_sweep.py: the map is exactly periodic in MA offset mod
360 deg (MA+420deg reproduces MA+60deg bit-for-bit). We sweep past 360 deg on
purpose (up to 430 deg) so the tail of the sequence is a built-in ground-truth
check for any model that claims to predict "future" (higher MA) frames.
"""

import os
import time
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_grid, save_megno_grid

BASELINE_DATE = "2012-09-30 00:00"
MA_START_DEG, MA_STOP_DEG, MA_STEP_DEG = 0, 900, 10  # inclusive of stop (~2.5 periods)

# Walrus's adaptive-stride encoder requires each non-singleton spatial axis
# to be a multiple of 32 (or exactly 1); 32x32 is the smallest valid grid.
SMA_MIN, SMA_MAX, N_GRID_SMA = 5, 7, 32
ECC_MIN, ECC_MAX, N_GRID_ECC = 0.001, 0.9, 32
TSPAN_YEARS = 50
INTEGRATOR = "whfast"
N_JOBS = 16

OUT_DIR = "../Data/MEGNO_maps"
STACK_PATH = os.path.join(OUT_DIR, "megno_dense_ma_sequence.npz")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    SMA_lin = np.linspace(SMA_MIN, SMA_MAX, N_GRID_SMA)
    ECC_lin = np.linspace(ECC_MIN, ECC_MAX, N_GRID_ECC)

    simfile, MJD, jorb = build_sunjup_sim(BASELINE_DATE)
    print(f"Sun+Jupiter built at {BASELINE_DATE} (MJD {MJD})")
    print(f"Jupiter orbit: {jorb}")

    ma_offsets = list(range(MA_START_DEG, MA_STOP_DEG + 1, MA_STEP_DEG))
    print(f"{len(ma_offsets)} frames: MA offsets {ma_offsets[0]}..{ma_offsets[-1]} "
          f"step {MA_STEP_DEG} deg")

    t0 = time.time()
    frames = []
    for ma_deg in ma_offsets:
        t_frame = time.time()
        SMA_grid, ECC_grid, MEGNO_grid = compute_megno_grid(
            simfile, SMA_lin, ECC_lin,
            MA_config_deg=ma_deg,
            tspan_years=TSPAN_YEARS,
            integrator=INTEGRATOR,
            n_jobs=N_JOBS,
        )
        n_nan = int(np.isnan(MEGNO_grid).sum())
        frames.append(MEGNO_grid)
        print(f"MA+{ma_deg:>3d}deg  nan={n_nan}/{MEGNO_grid.size}  "
              f"({time.time() - t_frame:.1f}s)")

    MEGNO_stack = np.stack(frames, axis=0)  # (n_frames, N_GRID_ECC, N_GRID_SMA)
    np.savez(
        STACK_PATH,
        MEGNO_stack=MEGNO_stack,
        MA_offsets_deg=np.array(ma_offsets),
        SMA_grid=SMA_grid,
        ECC_grid=ECC_grid,
        SMA_lin=SMA_lin,
        ECC_lin=ECC_lin,
        MJD_model=MJD,
        tspan_years=TSPAN_YEARS,
        integrator=INTEGRATOR,
    )
    print(f"Saved stacked sequence {MEGNO_stack.shape} to {STACK_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
