"""
Standalone worker invoked as a subprocess by megno_map.py's timeout-guarded
compute_megno_grid. Runs in its own process group so a hung rebound Kepler
solver (pure C, doesn't respond to Python-level signals) can be killed
unconditionally with SIGKILL from the parent, without touching the parent's
own process.
"""

import sys
import numpy as np
from joblib import Parallel, delayed

from megno_map import _megno_single_plain, _megno_timeseries_single


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    d = np.load(in_path, allow_pickle=True)

    mode = str(d["mode"]) if "mode" in d else "snapshot"
    simfile = str(d["simfile"])
    SMA_flat = d["SMA_flat"]
    ECC_flat = d["ECC_flat"]
    inc, Omega, omega = float(d["inc"]), float(d["Omega"]), float(d["omega"])
    MA_config_deg = float(d["MA_config_deg"])
    integrator = str(d["integrator"])
    dt_frac = float(d["dt_frac"])
    exit_max_distance = float(d["exit_max_distance"])
    n_jobs = int(d["n_jobs"])

    # n_jobs==1: no joblib/multiprocessing pool at all -- used for bisection
    # retries, where spinning up a fresh 16-worker pool (and its POSIX
    # semaphores) for every retry attempt across many bisection levels turned
    # out to exhaust the OS semaphore namespace during a real run (a
    # contiguous band of unstable points, not just one, triggered many
    # repeated kill+retry cycles in a short window).
    if mode == "timeseries":
        checkpoint_years = d["checkpoint_years"].tolist()
        if n_jobs == 1:
            results = [
                _megno_timeseries_single(simfile, a, e, inc, Omega, omega, MA_config_deg,
                                          checkpoint_years, integrator, dt_frac, exit_max_distance)
                for a, e in zip(SMA_flat, ECC_flat)
            ]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_megno_timeseries_single)(simfile, a, e, inc, Omega, omega, MA_config_deg,
                                                   checkpoint_years, integrator, dt_frac,
                                                   exit_max_distance)
                for a, e in zip(SMA_flat, ECC_flat)
            )
        # (n_points, n_years)
        np.save(out_path, np.array(results, dtype=np.float64))
    else:
        tspan_years = float(d["tspan_years"])
        if n_jobs == 1:
            results = [
                _megno_single_plain(simfile, a, e, inc, Omega, omega, MA_config_deg,
                                     tspan_years, integrator, dt_frac, exit_max_distance)
                for a, e in zip(SMA_flat, ECC_flat)
            ]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_megno_single_plain)(simfile, a, e, inc, Omega, omega, MA_config_deg,
                                              tspan_years, integrator, dt_frac, exit_max_distance)
                for a, e in zip(SMA_flat, ECC_flat)
            )
        np.save(out_path, np.array(results, dtype=np.float64))


if __name__ == "__main__":
    main()
