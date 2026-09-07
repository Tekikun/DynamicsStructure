# MEGNO map generation for the Sun-Jupiter Greek/Trojan co-orbital region

Generates MEGNO (Mean Exponential Growth of Nearby Orbits) chaos-indicator
maps over an (a, e) grid of test-particle orbits in the Greek/Trojan
co-orbital configuration (L4-leading, mean anomaly = Jupiter's MA + 60 deg),
using `rebound`'s N-body integrator and built-in MEGNO computation.

## Directory structure

```
code/
  megno_map.py                  core module: build_sunjup_sim, compute_megno_grid,
                                 compute_megno_timeseries_grid, plus a subprocess/
                                 timeout safety net for pathological (a,e) points
                                 that can hang rebound's C Kepler solver forever
  megno_worker.py                subprocess entry point used by that safety net
  megno_training_data.py         dense MA (spatial-axis) sweep: 1-degree-step
                                 training grid + held-out half-degree validation
                                 grid, one 50-year MEGNO snapshot per MA value
  megno_time_training_data.py    dense time (temporal-axis) sweep: single fixed
                                 MA, half-year-step checkpoints from one continuous
                                 100-year integration, integer-year training set +
                                 held-out half-year validation set
  megno_time_evolution_all_ma.py       time-evolution dataset (single continuous
                                 100yr integration per (a,e) point, checkpointed
                                 every 10yr), 16 MA values
  megno_time_evolution_finegrained.py  same idea, 180 MA values (2-degree step),
                                 checkpoint-safe: re-running skips MA values
                                 already computed, so an interrupted run resumes
                                 cleanly instead of restarting

Data/
  MEGNO_maps/                    generated .npz datasets land here
```

## Key physical/methodological notes

- **MEGNO is a running (cumulative) statistic** over one continuous
  integration. `compute_megno_timeseries_grid` reflects this correctly: one
  `rebound.Simulation` per (a,e) point, integrated forward with repeated
  `sim.integrate(t)` + `sim.megno()` calls at increasing checkpoint times --
  *not* independent restarts from different epochs. Sampling more checkpoint
  times from the same trajectory is nearly free (it doesn't change the total
  integration length), so time-axis resolution can be increased cheaply;
  increasing the *number of MA values* is not free, since each one requires
  an independent full-grid integration.
- **MA-offset periodicity is exact** (confirmed to floating-point precision:
  MA and MA+360 give bit-identical maps), so a dense MA sweep only needs to
  cover one 360-degree period.
- A handful of (a,e) grid points are dynamically pathological and can send
  `rebound`'s C-level Kepler solver into a near-infinite slow-convergence
  loop that never raises a Python exception (so it can't be caught with a
  try/except, and a Python-level `SIGALRM` doesn't fire either, since the C
  loop never yields back to the interpreter). `megno_map.py`'s
  `timeout_seconds` argument routes computation through a subprocess that
  *can* be `SIGKILL`ed on timeout, then retries the surviving chunk at
  `n_jobs=1` (see "gotchas" below for why not a fresh pool).

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install rebound numpy pandas joblib julian matplotlib progressbar2 scipy psutil
```

`julian` and `rebound`'s NASA Horizons lookup need network access the first
time a given epoch's Sun+Jupiter simulation is built (cached to a local
`.bin` file after that).

## Commands

Run from inside `code/` (scripts use bare relative imports and
`../Data/...`-relative output paths):

```bash
cd code

# Dense MA sweep (spatial axis): 32x32 default resolution
python megno_training_data.py

# Dense MA sweep at higher resolution
MEGNO_N_SMA=1024 MEGNO_N_ECC=512 MEGNO_OUT_SUFFIX=_1024x512 python megno_training_data.py

# Dense time sweep (temporal axis), fixed MA=60 by default
MEGNO_N_SMA=1024 MEGNO_N_ECC=512 python megno_time_training_data.py

# Time-evolution dataset, 16 MA values, 10yr checkpoints
MEGNO_N_SMA=1024 MEGNO_N_ECC=512 MEGNO_OUT_SUFFIX=_1024x512 python megno_time_evolution_all_ma.py

# Fine-grained time-evolution dataset, 180 MA values (2yr checkpoints),
# checkpoint-safe -- interrupt and re-run any time, already-computed MA
# values are skipped
MEGNO_N_SMA=1024 MEGNO_N_ECC=512 python megno_time_evolution_finegrained.py
```

Every script is configurable via env vars (grep `os.environ.get` in each
file for the full list) -- grid bounds are fixed in-file (a: 5-7 AU,
e: 0.001-0.9), but resolution (`MEGNO_N_SMA`/`MEGNO_N_ECC`), output suffix
(`MEGNO_OUT_SUFFIX`), integrator, and timeout behavior
(`MEGNO_TIMEOUT_SECONDS`) are all overridable.

## Output format

Each script saves one or more `.npz` files to `Data/MEGNO_maps/`, with keys
that vary by script but always include a `MEGNO_stack` array and the
corresponding axis values (`MA_deg`, `year`, or `ma_values`/
`checkpoint_years` for the multi-MA time-evolution datasets), plus the
`SMA_lin`/`ECC_lin` grid linspaces. `MEGNO_stack` entries are `nan` where a
test particle was ejected or had a close encounter during integration
(treated as maximally chaotic / undefined, not a computation failure).

## Gotchas

- **Resolution scaling is asymmetric.** More checkpoint years per MA value
  is nearly free; more MA values is not (each is an independent full-grid
  integration). Budget compute accordingly -- see `megno_time_evolution_finegrained.py`'s
  docstring for a concrete cost breakdown.
- **On macOS, avoid rapid repeated kill + new joblib-pool-creation cycles**
  across many runs. It can exhaust the OS's POSIX semaphore namespace
  (`OSError: No space left on device` on semaphore creation), which then
  breaks every subsequent parallel job until a reboot. This is why the
  timeout-safety-net's retry path in `megno_map.py` uses `n_jobs=1` (no
  pool at all) rather than spinning up a fresh smaller pool.
- A pathological (a,e) point hanging the Kepler solver is a real, observed
  failure mode at high grid resolution (524k points/frame was enough to hit
  it in practice) -- always pass `timeout_seconds` for large sweeps.
