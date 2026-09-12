# Test fixtures

## contrail_4d.casadi

A 4D CasADi bspline interpolant of ATR20 contrail cost for a real-world
route (EDDB → LEMD) on 2023-01-05, built from ERA5 meteo fields with a
Gaussian smoothing (σ=2) applied to the raw contrail forcing grid.

Axes (in this order):
- longitude (degrees)
- latitude (degrees)
- height (metres)
- ts (seconds from start)

Used by `tests/test_grid_4d.py`, `tests/benchmark.py`, and
`tests/compare_nlp_scaling.py` to exercise the time-dependent grid-cost
optimization path with physically realistic inputs.

**Regeneration:** run `tests/fixtures/build_contrail_4d.py` via `uv run`;
the script has PEP 723 inline deps (`fastmeteo`, `scipy`, `openap`) so no
project-level install is needed. Requires access to an ERA5 Zarr store
(default `/tmp/era5-zarr`; first run downloads ~GB of data from the ARCO
public mirror).

    uv run tests/fixtures/build_contrail_4d.py \
        --origin 52.362,13.501 --dest 40.472,-3.563 \
        --start 2023-01-05T09:48 --stop 2023-01-05T12:00 \
        --sigma 2 \
        --out tests/fixtures/contrail_4d.casadi

## complete_flight_golden.json

Numerical objective/fuel/iteration baseline for the continuous-control
CompleteFlight EHAM-LGAV A320 case: 7,303.786712 kg, with 43 intervals and
degree-3 collocation. The fixture pins mesh and phase allocation and records
source hashes for `b30086d` plus the initialization correction. The existing
1% regression tolerance is unchanged.

A one-time 43/86/172/344-interval study changes fuel by 0.110%. A separate
43-interval solution with 63 extra force/energy constraint points costs only
0.083816 kg more. An independent 1,602-point-per-interval audit and RK4 replay
find maximum equivalent-force residuals below 0.003 N for that warm-started
constrained solution. The fixture records these extra points and audit results
as provenance; the golden test retains its original constraint configuration.
The mesh study and adaptive comparison are not permanent test cases.

This is a numerical regression baseline, not a continuous-path feasibility
certificate. Default boundary/collocation checks still miss between-node peaks.
Extra points must be selected and re-audited for each problem; points from this
fixture are not a general route-independent constraint preset.

## flight_ryr880w_2023-01-05.parquet

Real OpenSky trace for RYR880W on 2023-01-05 (EDDB→LEMD, B738). Paired
with `contrail_4d.casadi` (same flight, same date). Used by
`tests/test_replay_fetch.py` and `tests/test_replay_end_to_end.py`.

**Regeneration (requires OpenSky credentials):**

    uv run --no-project --with "pandas<3" --with traffic --with pyarrow python -c "
    from traffic.data import opensky
    f = opensky.history(
        '2023-01-05T09:00', '2023-01-05T13:00',
        callsign='RYR880W', return_flight=True,
    )
    f.to_parquet('tests/fixtures/flight_ryr880w_2023-01-05.parquet')
    "
