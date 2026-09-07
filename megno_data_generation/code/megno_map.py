"""
MEGNO(a,e) map generation for the Sun-Jupiter Greek-cloud restricted 3-body
problem. Reuses the Sun+Jupiter setup / Greek grid convention from
Greek_ReBound_Sun_Jup.py, but adds rebound's built-in MEGNO chaos indicator
(sim.init_megno() / sim.megno()), which does not otherwise exist in this repo.

Written against rebound 5.x's API (particle.orbit(), sim.megno()) rather than
the older 3.x API (calculate_orbit(), calculate_megno()) used by some of the
legacy scripts in this folder.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import psutil
import rebound
from joblib import Parallel, delayed


def _kill_process_tree(pid):
    """Kill a process and every descendant, regardless of process-group/session
    nesting. os.killpg() alone isn't reliable here: loky's own worker spawning
    can put its child processes in a new session, which escapes a killpg() on
    the parent's group -- observed in practice as orphaned LokyProcess workers
    surviving a "successful" killpg and running indefinitely (82+ minutes in
    one case) after the parent subprocess was killed."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=5)


def build_sunjup_sim(date, folder="./simfiles_megno"):
    """Build a fresh Sun+Jupiter rebound sim at `date` (NASA Horizons lookup),
    save it to a binary, and return (simfile, MJD_model, jup_orbit).

    Binaries are written to a dedicated subfolder (not the legacy
    SunJup_*.bin files in the repo root) since those were saved with an
    older, now-incompatible rebound simulationarchive format.
    """
    import datetime
    import julian

    date_info = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M")
    MJD_model = julian.to_jd(date_info, fmt="mjd")

    sim = rebound.Simulation()
    sim.add("Sun", date=date)
    sim.add("Jupiter")

    os.makedirs(folder, exist_ok=True)
    simfile = os.path.join(folder, f"SunJup_{MJD_model}.bin")
    if os.path.exists(simfile):
        os.remove(simfile)
    sim.save_to_file(simfile)

    jup_orbit = sim.particles[1].orbit(primary=sim.particles[0])
    return simfile, MJD_model, jup_orbit


def _megno_single_plain(simfile, a, e, inc, Omega, omega, MA_config_deg,
                         tspan_years, integrator, dt_frac, exit_max_distance):
    """Compute MEGNO for one Greek-case test particle. Returns np.nan if the
    particle is ejected / has a close encounter (treated as maximally
    chaotic / undefined rather than crashing the whole grid).

    Note: some pathological (a,e) points can send whfast's Kepler solver into
    a near-infinite slow-convergence loop *inside rebound's C core*, which
    never raises a Python exception and can't be interrupted by a Python-level
    signal handler (SIGALRM) either, since the C loop never yields back to the
    interpreter. That failure mode is guarded at a higher level, in
    compute_megno_grid's subprocess+timeout wrapper (see run_worker_safe),
    not here -- this function has no timeout of its own.
    """
    sim = rebound.Simulation(simfile)
    jup = sim.particles[1]
    jorb = jup.orbit(primary=sim.particles[0])
    MA = jorb.M + np.deg2rad(MA_config_deg)

    sim.add(primary=sim.particles[0], a=a, e=e, inc=inc,
            Omega=Omega, omega=omega, M=MA)

    sim.integrator = integrator
    period = 2 * np.pi * a ** 1.5  # test particle's own Keplerian period, code units
    sim.dt = dt_frac * period
    sim.exit_max_distance = exit_max_distance

    sim.move_to_com()
    sim.init_megno()

    try:
        sim.integrate(tspan_years * 2 * np.pi)
        return sim.megno()
    except (rebound.Encounter, rebound.Escape):
        return np.nan


def _megno_timeseries_single(simfile, a, e, inc, Omega, omega, MA_config_deg,
                              checkpoint_years, integrator, dt_frac, exit_max_distance):
    """Genuine MEGNO *time evolution* for one Greek-case test particle: ONE
    continuous integration from t=0, sampling the running (cumulative) MEGNO
    value at each of `checkpoint_years` -- not independent restarts from
    different real-ephemeris epochs (that conflates "how has Jupiter's real
    orbit drifted over calendar decades" with "how does chaos develop as a
    trajectory integrates forward", two different questions). MEGNO is a
    running statistic over a simulation's own history, so calling
    sim.integrate(t) then sim.megno() repeatedly with increasing t, on the
    SAME sim object, gives the standard MEGNO(t) growth curve for that one
    trajectory.

    Returns a list of length len(checkpoint_years); once ejected, remaining
    entries are NaN."""
    sim = rebound.Simulation(simfile)
    jup = sim.particles[1]
    jorb = jup.orbit(primary=sim.particles[0])
    MA = jorb.M + np.deg2rad(MA_config_deg)

    sim.add(primary=sim.particles[0], a=a, e=e, inc=inc,
            Omega=Omega, omega=omega, M=MA)

    sim.integrator = integrator
    period = 2 * np.pi * a ** 1.5
    sim.dt = dt_frac * period
    sim.exit_max_distance = exit_max_distance

    sim.move_to_com()
    sim.init_megno()

    results = []
    for yr in checkpoint_years:
        try:
            sim.integrate(yr * 2 * np.pi)
            results.append(sim.megno())
        except (rebound.Encounter, rebound.Escape):
            results.extend([np.nan] * (len(checkpoint_years) - len(results)))
            break
    return results


def compute_megno_timeseries_grid(simfile, SMA_linspace, ECC_linspace, checkpoint_years,
                                   MA_config_deg=60, integrator="whfast", dt_frac=1 / 20,
                                   exit_max_distance=100.0, n_jobs=16, timeout_seconds=None):
    """Like compute_megno_grid, but returns the full MEGNO(t) evolution per
    (a,e) point from a single continuous integration per point, checkpointed
    at `checkpoint_years`. Returns (SMA_grid, ECC_grid, MEGNO_stack) where
    MEGNO_stack has shape (len(checkpoint_years), len(ECC_linspace), len(SMA_linspace)).

    timeout_seconds: same subprocess+bisection safety net as compute_megno_grid
    (see its docstring) -- important at high resolution, where a single
    pathological (a,e) point can hang rebound's C Kepler solver forever.
    """
    sim0 = rebound.Simulation(simfile)
    inc = sim0.particles[1].orbit(primary=sim0.particles[0]).inc
    Omega = sim0.particles[1].orbit(primary=sim0.particles[0]).Omega
    omega = sim0.particles[1].orbit(primary=sim0.particles[0]).omega

    SMA_grid, ECC_grid = np.meshgrid(SMA_linspace, ECC_linspace)
    SMA_flat, ECC_flat = SMA_grid.flatten(), ECC_grid.flatten()

    if timeout_seconds is None:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_megno_timeseries_single)(simfile, a, e, inc, Omega, omega, MA_config_deg,
                                               checkpoint_years, integrator, dt_frac,
                                               exit_max_distance)
            for a, e in zip(SMA_flat, ECC_flat)
        )
        results = np.array(results)  # (n_points, n_years)
    else:
        results = _run_worker_subprocess(
            simfile, SMA_flat, ECC_flat, inc, Omega, omega, MA_config_deg,
            None, integrator, dt_frac, exit_max_distance, n_jobs, timeout_seconds,
            mode="timeseries", checkpoint_years=checkpoint_years,
        )

    # results: (n_points, n_years) -> (n_years, n_points) -> (n_years, ECC, SMA)
    MEGNO_stack = results.T.reshape(len(checkpoint_years), *SMA_grid.shape)
    return SMA_grid, ECC_grid, MEGNO_stack


_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "megno_worker.py")


def _try_subprocess_once(simfile, SMA_flat, ECC_flat, inc, Omega, omega, MA_config_deg,
                          tspan_years, integrator, dt_frac, exit_max_distance, n_jobs,
                          timeout_seconds, mode="snapshot", checkpoint_years=None):
    """Single attempt: run the whole chunk in one subprocess. Returns the
    result array on success, or None on timeout/crash (caller decides what
    to do next -- bisect or give up)."""
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.npz")
        out_path = os.path.join(td, "out.npy")
        payload = dict(simfile=simfile, SMA_flat=SMA_flat, ECC_flat=ECC_flat,
                       inc=inc, Omega=Omega, omega=omega, MA_config_deg=MA_config_deg,
                       integrator=integrator, dt_frac=dt_frac,
                       exit_max_distance=exit_max_distance, n_jobs=n_jobs, mode=mode)
        if mode == "timeseries":
            payload["checkpoint_years"] = np.array(checkpoint_years)
        else:
            payload["tspan_years"] = tspan_years
        np.savez(in_path, **payload)

        proc = subprocess.Popen(
            [sys.executable, _WORKER_SCRIPT, in_path, out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        try:
            proc.communicate(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)  # gets loky grandchildren too, not just proc itself
            proc.wait()
            timed_out = proc.returncode != 0  # it may have actually finished OK just in time

        if not timed_out and proc.returncode == 0 and os.path.exists(out_path):
            return np.load(out_path)
    return None


def _retry_sequential(simfile, SMA_flat, ECC_flat, inc, Omega, omega, MA_config_deg,
                       tspan_years, integrator, dt_frac, exit_max_distance,
                       retry_timeout, min_chunk, mode="snapshot", checkpoint_years=None):
    """Retry a chunk that failed at n_jobs=16, using n_jobs=1 (no joblib pool,
    no POSIX semaphores at all) so repeated retries -- e.g. a whole contiguous
    band of unstable points, not just one -- can't exhaust the OS semaphore
    namespace the way repeatedly spinning up fresh 16-worker pools did in
    practice (observed: a real run hit "OSError: No space left on device" on
    semaphore creation after enough kill+retry cycles, which then made every
    subsequent frame fail outright). Bisects on failure down to `min_chunk`,
    then gives up and marks the remainder NaN."""
    n = len(SMA_flat)
    n_years = len(checkpoint_years) if mode == "timeseries" else None
    if n == 0:
        return np.zeros((0, n_years)) if mode == "timeseries" else np.array([])

    result = _try_subprocess_once(simfile, SMA_flat, ECC_flat, inc, Omega, omega, MA_config_deg,
                                   tspan_years, integrator, dt_frac, exit_max_distance, 1,
                                   retry_timeout, mode, checkpoint_years)
    if result is not None:
        return result

    if n <= min_chunk:
        return np.full((n, n_years), np.nan) if mode == "timeseries" else np.full(n, np.nan)
    mid = n // 2
    left = _retry_sequential(simfile, SMA_flat[:mid], ECC_flat[:mid], inc, Omega, omega,
                              MA_config_deg, tspan_years, integrator, dt_frac,
                              exit_max_distance, retry_timeout, min_chunk, mode, checkpoint_years)
    right = _retry_sequential(simfile, SMA_flat[mid:], ECC_flat[mid:], inc, Omega, omega,
                               MA_config_deg, tspan_years, integrator, dt_frac,
                               exit_max_distance, retry_timeout, min_chunk, mode, checkpoint_years)
    return np.concatenate([left, right])


def _run_worker_subprocess(simfile, SMA_flat, ECC_flat, inc, Omega, omega, MA_config_deg,
                            tspan_years, integrator, dt_frac, exit_max_distance, n_jobs,
                            timeout_seconds, retry_chunk=2048, retry_timeout=10, min_chunk=128,
                            mode="snapshot", checkpoint_years=None):
    """Run _megno_single_plain (or _megno_timeseries_single, if mode="timeseries")
    over (SMA_flat,ECC_flat) in a subprocess of its own process group
    (n_jobs-way parallel via joblib), killable with SIGKILL on timeout
    regardless of what's happening inside rebound's C code.

    On timeout, split into fixed-size `retry_chunk` pieces and retry each
    sequentially (see _retry_sequential) -- flat chunking rather than binary
    halving from the full size, so no single retry attempt is ever awkwardly
    large (halving 524288 down would make the *first* retry level itself
    262144 points, too big to bound a sane timeout for)."""
    n = len(SMA_flat)
    if n == 0:
        return np.array([])

    result = _try_subprocess_once(simfile, SMA_flat, ECC_flat, inc, Omega, omega, MA_config_deg,
                                   tspan_years, integrator, dt_frac, exit_max_distance, n_jobs,
                                   timeout_seconds, mode, checkpoint_years)
    if result is not None:
        return result

    pieces = []
    for i in range(0, n, retry_chunk):
        pieces.append(_retry_sequential(
            simfile, SMA_flat[i:i + retry_chunk], ECC_flat[i:i + retry_chunk],
            inc, Omega, omega, MA_config_deg, tspan_years, integrator, dt_frac,
            exit_max_distance, retry_timeout, min_chunk, mode, checkpoint_years,
        ))
    return np.concatenate(pieces)


def compute_megno_grid(simfile, SMA_linspace, ECC_linspace, MA_config_deg=60,
                        tspan_years=50, integrator="whfast", dt_frac=1 / 20,
                        exit_max_distance=100.0, n_jobs=16, timeout_seconds=None):
    """Compute a MEGNO(a,e) grid for the Greek case (+MA_config_deg ahead of
    Jupiter), reusing a pre-built Sun+Jupiter binary (see build_sunjup_sim).

    timeout_seconds: if None (default), compute directly in-process via
    joblib -- fast, no subprocess overhead, matches all prior behavior. If
    set, route through a subprocess+bisection safety net that can survive a
    rare pathological (a,e) point hanging rebound's C Kepler solver forever
    (observed at very high grid resolution: 524k points/frame across a
    360-frame sweep was enough to hit this in practice). Use this for large
    dense sweeps; leave it off for small/interactive runs.

    Returns (SMA_grid, ECC_grid, MEGNO_grid), all shape
    (len(ECC_linspace), len(SMA_linspace)) matching np.meshgrid convention.
    """
    sim0 = rebound.Simulation(simfile)
    inc = sim0.particles[1].orbit(primary=sim0.particles[0]).inc
    Omega = sim0.particles[1].orbit(primary=sim0.particles[0]).Omega
    omega = sim0.particles[1].orbit(primary=sim0.particles[0]).omega

    SMA_grid, ECC_grid = np.meshgrid(SMA_linspace, ECC_linspace)
    SMA_flat, ECC_flat = SMA_grid.flatten(), ECC_grid.flatten()

    if timeout_seconds is None:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_megno_single_plain)(simfile, a, e, inc, Omega, omega, MA_config_deg,
                                          tspan_years, integrator, dt_frac, exit_max_distance)
            for a, e in zip(SMA_flat, ECC_flat)
        )
        results = np.array(results)
    else:
        results = _run_worker_subprocess(
            simfile, SMA_flat, ECC_flat, inc, Omega, omega, MA_config_deg,
            tspan_years, integrator, dt_frac, exit_max_distance, n_jobs, timeout_seconds,
        )

    MEGNO_grid = np.array(results).reshape(SMA_grid.shape)
    return SMA_grid, ECC_grid, MEGNO_grid


def save_megno_grid(path, SMA_grid, ECC_grid, MEGNO_grid, MJD_model, tspan_years,
                     integrator, MA_config_deg):
    np.savez(path, SMA_grid=SMA_grid, ECC_grid=ECC_grid, MEGNO_grid=MEGNO_grid,
             MJD_model=MJD_model, tspan_years=tspan_years, integrator=integrator,
             MA_config_deg=MA_config_deg)


def plot_megno_grid(SMA_grid, ECC_grid, MEGNO_grid, title="", ax=None, vmax=6):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(SMA_grid, ECC_grid, MEGNO_grid, shading="auto",
                        cmap="viridis", vmin=2, vmax=vmax)
    ax.set_xlabel("a (AU)")
    ax.set_ylabel("e")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="MEGNO")
    return ax
