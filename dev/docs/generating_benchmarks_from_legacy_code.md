# Generating Benchmark Rasters from Legacy Code

How to produce each refactored model version's benchmark `.tif` using the
legacy codebases. These benchmarks live in `tests/benchmarks/`
and are compared against the refactored pipeline output by `test_benchmarks.py`.

## Environment

All legacy runs use the `oldvs30_venv` conda environment. Because the system
`gdal_calc.py` is incompatible with the environment's NumPy, prepend the
environment's bin directory to PATH:

```bash
PATH="/home/arr65/miniforge-pypy3/envs/oldvs30_venv/bin:$PATH"
LEGACY_PY=/home/arr65/miniforge-pypy3/envs/oldvs30_venv/bin/python3
```

Always use `--nproc 1` — both legacy codebases deadlock with nproc > 1.

## Grid Bounds

The benchmark test uses two grids (defined in `tests/test_benchmarks.py`):

| Grid | xmin | xmax | ymin | ymax | Used by |
|------|------|------|------|------|---------|
| `BENCHMARK_NZ_GRID` | 1060100 | 2120100 | 4730100 | 6250100 | modified_foster_2019, jaehwi_v1p0, viktor_cpt_clustering |
| `FOSTER_2019_GRID` | 1000000 | 2126400 | 4700000 | 6338400 | foster_2019_approx |

Pass `--dx 5000 --dy 5000` for benchmark resolution.

## modified_foster_2019

**Legacy repo:** `/home/arr65/src/pre-refactor-Vs30-for-comparison`

```bash
$LEGACY_PY run_vs30calc.py \
    --source original \
    --gupdate posterior_paper --tupdate posterior_paper \
    --xmin 1060100 --xmax 2120100 --ymin 4730100 --ymax 6250100 \
    --dx 5000 --dy 5000 \
    --out /tmp/bench_modified_foster_2019 \
    --nproc 1 --overwrite
```

Copy `combined_mvn.tif` to `tests/benchmarks/modified_foster_2019.tif`.

**Key flags:**
- `--source original` selects the original (non-CPT) observation dataset.
- `--gupdate posterior_paper --tupdate posterior_paper` — NOT `posterior`.
  Using `--gupdate posterior` produces different (wrong) values.

## viktor_cpt_clustering

**Legacy repo:** `/home/arr65/src/pre-refactor-Vs30-for-comparison`

```bash
$LEGACY_PY run_vs30calc.py \
    --source cpt \
    --gupdate posterior --tupdate posterior \
    --xmin 1060100 --xmax 2120100 --ymin 4730100 --ymax 6250100 \
    --dx 5000 --dy 5000 \
    --out /tmp/bench_viktor_cpt_clustering \
    --nproc 1 --overwrite
```

Copy `combined_mvn.tif` to `tests/benchmarks/viktor_cpt_clustering.tif`.

**Key flags:**
- `--source cpt` selects the CPT-derived observation dataset.
- `--gupdate posterior --tupdate posterior` (not `posterior_paper`).

## jaehwi_v1p0

**Legacy repo:** `/home/arr65/src/jaehwi_fork_vs30/Vs30_2026`
(Jaehwi's fork, NOT the pre-refactor repo — it has model-specific modifications.)

```bash
$LEGACY_PY run_vs30calc_V1.py \
    --gupdate posterior --tupdate posterior \
    --xmin 1060100 --xmax 2120100 --ymin 4730100 --ymax 6250100 \
    --dx 5000 --dy 5000 \
    --out /tmp/bench_jaehwi_v1p0 \
    --nproc 1 --overwrite
```

Copy `combined_mvn.tif` to `tests/benchmarks/jaehwi_v1p0.tif`.

Then **gap-fill** the benchmark (the Jaehwi fork does not fill nodata gaps,
but the refactored pipeline does):

```bash
python dev/gapfill_benchmark.py tests/benchmarks/jaehwi_v1p0.tif
```

**Key flags:**
- `--gupdate posterior --tupdate posterior` — NOT `posterior_paper` (the
  Jaehwi fork defaults to `posterior_paper`, but the refactored jaehwi_v1p0
  config uses Bayesian update from priors, which corresponds to `posterior`
  in the legacy code).
- No `--source` flag needed — the fork defaults to `original`.

**Coordinate note:** Jaehwi's fork has different default grid bounds
(xmin=1060100 vs pre-refactor's 1060050, ymin=4730000 vs 4730050). At 5000m
resolution with explicit bounds, both codebases produce identical pixel grids
so no coordinate adjustment is needed.

## foster_2019_approx

Not yet documented. The foster_2019_approx model uses a Matern correlation
function and different observation filtering, so generating its benchmark from
legacy code requires a different approach. See
`wiki/differences_between_foster_2019_approx_and_modified_foster_2019.md`.
