"""
Benchmark MEGNO grid generation time at various resolutions, to scope out
what's tractable for a dense (many MA values) high-resolution training set.
"""

import time
import numpy as np

from megno_map import build_sunjup_sim, compute_megno_grid

BASELINE_DATE = "2012-09-30 00:00"
SMA_MIN, SMA_MAX = 5, 7
ECC_MIN, ECC_MAX = 0.001, 0.9
TSPAN_YEARS = 50
INTEGRATOR = "whfast"
N_JOBS = 16

RESOLUTIONS = [(32, 32), (128, 64), (256, 128), (512, 256), (1024, 512)]


def main():
    simfile, MJD, jorb = build_sunjup_sim(BASELINE_DATE)
    print(f"Sun+Jupiter built at {BASELINE_DATE} (MJD {MJD})")

    for n_sma, n_ecc in RESOLUTIONS:
        SMA_lin = np.linspace(SMA_MIN, SMA_MAX, n_sma)
        ECC_lin = np.linspace(ECC_MIN, ECC_MAX, n_ecc)
        n_points = n_sma * n_ecc
        t0 = time.time()
        _, _, MEGNO_grid = compute_megno_grid(
            simfile, SMA_lin, ECC_lin, MA_config_deg=60,
            tspan_years=TSPAN_YEARS, integrator=INTEGRATOR, n_jobs=N_JOBS,
        )
        dt = time.time() - t0
        print(f"{n_sma}x{n_ecc} ({n_points:>8,} points): {dt:8.2f}s  "
              f"({dt / n_points * 1000:.4f} ms/point)  "
              f"-> est. per 360-frame sweep: {dt * 360 / 3600:.2f} hr")


if __name__ == "__main__":
    main()
