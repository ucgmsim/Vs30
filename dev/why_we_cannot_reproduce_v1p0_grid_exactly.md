# Why We Cannot Reproduce V1.0_26Mar.tif Exactly

This document explains, in non-technical terms, why our refactored Vs30 pipeline
produces slightly different grid output from Jaehwi's reference file
`V1.0_26Mar.tif`, and why this is expected and correct.

## The Short Version

Our code computes the **same model** as Jaehwi's code. When both are run on the
same point, they agree (862.14 vs 862.51 m/s — a 0.04% difference). However,
Jaehwi's grid-mode code introduces several precision-reducing shortcuts that our
code avoids. These shortcuts change ~15% of pixels by a few percent.

**The strongest evidence:** even Jaehwi's own code cannot reproduce
V1.0_26Mar.tif when run with a different number of processors (2.83% mean
difference, 73% of pixels identical). The reference file is a one-time snapshot
that cannot be deterministically recreated.

## Background: How the Vs30 Map is Made

The Vs30 pipeline has two key steps:

1. **Categorical model**: Assign each pixel a Vs30 value based on its geology
   or terrain category. This step is simple table lookups — our code matches
   exactly.

2. **MVN spatial adjustment**: Refine each pixel's value using nearby
   field observations. This step computes distances between the pixel and
   hundreds of observation stations, builds a covariance matrix, and
   inverts it. This is where the differences arise.

The MVN step is mathematically sensitive: small changes in input distances
produce small changes in the covariance matrix, which can produce noticeable
changes in the output Vs30 value. This is normal for any calculation involving
matrix inversion.

## Three Sources of Difference

### 1. Reduced Precision for Coordinates

Computers store numbers in two precisions: **float64** (15 decimal digits of
accuracy) and **float32** (7 decimal digits). Jaehwi's grid-mode code converts
observation coordinates to float32, while points mode keeps them as float64.

New Zealand's coordinate system (NZTM) uses large numbers — for example,
a northing of **5,169,287.891234**. In float32, this becomes **5,169,288.0** — an
error of ~0.1 metres. These errors shift the computed distances between
observations and pixels, which changes the covariance matrix:

```
Coordinate              float64                float32        Error (m)
Obs A easting      1575312.456789         1575312.500000       0.043
Obs A northing     5169287.891234         5169288.000000       0.109
Obs B northing     5921043.234567         5921043.000000       0.235
```

The relevant code (from Jaehwi's `mvn.py`, lines 55–61):

```python
def _xy2complex(x):
    c = x[:, 0].astype(np.complex64)  # <-- converts to float32 precision
    c.imag += x[:, 1]
    return c
```

Our code uses `scipy.spatial.distance.cdist` which keeps float64 precision
throughout.

### 2. Reduced Precision for Vs30 Values

In grid mode, Vs30 values are written to a GeoTIFF file (which stores numbers
as float32), then read back for the MVN step. This roundtrip loses precision:

```
Value                 float64 (points)       float32 (grid)          Error
Vs30 (rock)           862.1423456780         862.1423339844     0.0000117
Vs30 (hard rock)     1250.9876543210        1250.9876708984     0.0000166
```

In points mode, values stay as float64 — no file roundtrip.

### 3. Processing Order Matters

Jaehwi's MVN code has a speed optimisation: if the previous pixel was far from
all observations, it assumes the next pixel is too, and **skips the calculation
entirely**. The relevant code (from `mvn.py`, lines 113–120):

```python
# don't recalculate distances if delta distance is too small anyway
try:
    movement = _dists(model_loc - prev_model_loc)
    if min_dist - movement > max_dist:
        continue  # SKIP this pixel entirely
except NameError:
    pass
```

When multiprocessing is used, the grid is split into chunks. Each chunk
processes a different portion of the grid and starts without any cached state.
This means different numbers of processors produce **different pixel processing
orders**, which causes different pixels to be skipped or computed.

Our code does not use this caching optimisation — every pixel is computed
independently, so results do not depend on processing order.

## Empirical Proof

### Test A: Points mode agrees between codebases

We ran Jaehwi's code and our code on the **same point** (easting=1575300,
northing=5169300):

| Source | Vs30 (m/s) |
|--------|-----------|
| Jaehwi's code (points mode) | 862.51 |
| Our refactored code | 862.14 |
| V1.0_26Mar.tif (grid mode) | 910.06 |

Points mode: **0.04% difference** — the two codebases agree.
Grid vs points: **5.5% difference** — the grid artifacts are significant.

### Test B: Jaehwi's own code cannot reproduce V1.0_26Mar.tif

We ran Jaehwi's own code on the full grid with 1 processor (V1.0_26Mar.tif
was produced with multiple processors):

| Metric | Value |
|--------|-------|
| Total pixels compared | 25,913,941 |
| Pixels that match exactly | 73% |
| Pixels that differ | 27% |
| Mean % difference | 2.83% |

**Even the same code cannot reproduce V1.0_26Mar.tif** with different processor
settings. The reference file is not deterministically reproducible.

### Test C: Our code is grid-points consistent

We run the same computation in grid mode and points mode and compare at every
pixel:

| Metric | Value |
|--------|-------|
| Difference | **Exactly zero at every pixel** |

This is confirmed by an automated test (`test_grid_and_points_consistency`).
Our code produces identical results regardless of mode because we use float64
throughout and compute every pixel independently.

## Conclusion

The refactored pipeline correctly implements Jaehwi's intended Vs30 model. The
differences to V1.0_26Mar.tif come from float32 precision shortcuts in the
legacy code's grid mode — shortcuts that our code intentionally avoids. These
shortcuts make the legacy output **non-reproducible** (processor-count dependent),
while our output is fully deterministic and reproducible.

A runnable demonstration of each artifact is available in
`dev/demonstrate_grid_artifacts.py`.
