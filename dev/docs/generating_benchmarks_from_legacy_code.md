# Regenerating Benchmark Rasters from Legacy Code

Three of the four benchmark `.tif` files in `tests/benchmarks/` are
produced by the original (pre-refactor) Vs30 codebases and committed as
fixtures the refactored pipeline is compared against. This page records
the legacy CLI commands used to produce each one.

The legacy CLIs deadlock with `--nproc > 1`, so always pass `--nproc 1`.

## Grid bounds

Two grids are used:

| Grid              | xmin    | xmax    | ymin    | ymax    | Used by                                                         |
|-------------------|---------|---------|---------|---------|-----------------------------------------------------------------|
| Benchmark NZ grid | 1060050 | 2120050 | 4730050 | 6250050 | `modified_foster_2019`, `jaehwi_v1p0`, `viktor_cpt_clustering`  |
| Foster 2019 grid  | 1000000 | 2126400 | 4700000 | 6338400 | `foster_2019_approx`                                            |

Pass `--dx 5000 --dy 5000` for benchmark resolution. The benchmark NZ
grid bounds were chosen so 5 km pixel centres land on IwahashiPike pixel
centres — see [Grid bounds semantics](grid_bounds_semantics_investigation.md).

## modified_foster_2019

Run from the pre-refactor Vs30 codebase:

```bash
run_vs30calc.py \
    --source original \
    --gupdate posterior_paper --tupdate posterior_paper \
    --xmin 1060050 --xmax 2120050 --ymin 4730050 --ymax 6250050 \
    --dx 5000 --dy 5000 \
    --out <out_dir> \
    --nproc 1 --overwrite
```

The output `combined_mvn.tif` becomes
`tests/benchmarks/modified_foster_2019.tif`.

Key flags:

- `--source original` selects the original (non-CPT) observation dataset.
- `--gupdate posterior_paper --tupdate posterior_paper` — *not*
  `posterior`. Using `--gupdate posterior` produces different (wrong)
  values.

## viktor_cpt_clustering

Run from the same pre-refactor Vs30 codebase:

```bash
run_vs30calc.py \
    --source cpt \
    --gupdate posterior --tupdate posterior \
    --xmin 1060050 --xmax 2120050 --ymin 4730050 --ymax 6250050 \
    --dx 5000 --dy 5000 \
    --out <out_dir> \
    --nproc 1 --overwrite
```

The output `combined_mvn.tif` becomes
`tests/benchmarks/viktor_cpt_clustering.tif`.

Key flags:

- `--source cpt` selects the CPT-derived observation dataset.
- `--gupdate posterior --tupdate posterior` — *not* `posterior_paper`.

## jaehwi_v1p0

Run from Jaehwi's fork (which has model-specific modifications, not the
pre-refactor Vs30 codebase used above):

```bash
run_vs30calc_V1.py \
    --gupdate posterior --tupdate posterior \
    --xmin 1060050 --xmax 2120050 --ymin 4730050 --ymax 6250050 \
    --dx 5000 --dy 5000 \
    --out <out_dir> \
    --nproc 1 --overwrite
```

The output `combined_mvn.tif` becomes
`tests/benchmarks/jaehwi_v1p0.tif` after a separate gap-fill pass — the
Jaehwi fork does not fill nodata gaps, but the refactored pipeline does,
so the benchmark needs to match.

Key flags:

- `--gupdate posterior --tupdate posterior` — *not* `posterior_paper`.
  The fork defaults to `posterior_paper`, but the refactored
  `jaehwi_v1p0` config uses Bayesian update from priors, which
  corresponds to `posterior` in the legacy code.
- No `--source` flag needed — the fork defaults to `original`.

## foster_2019_approx

The `foster_2019_approx` model uses a Matérn correlation function and
different observation filtering, so generating its benchmark from legacy
code requires a different approach. See
[The modified_foster_2019 model](modified_foster_2019.md).
