# Foster 2019 Model Reproduction Comparison

Comparison of the reproduced Foster 2019 model against the published reference rasters.

## Files Compared

| File | Description |
|------|-------------|
| `/home/arr65/data/vs30/grid_models/foster_2019/15_eeri_35_4_suppl_3_es1_online.tif` | Published Vs30 (1 band, float32, nodata=-3.4e+38) |
| `/home/arr65/data/vs30/grid_models/foster_2019/15_eeri_35_4_suppl_5_es1_online.tif` | Published StdDev (1 band, float32, nodata=-3.4e+38) |
| `/home/arr65/data/vs30/grid_models/foster_2019_reproduced/combined_vs30.tif` | Reproduced (2 bands: Vs30 + StdDev, nodata=-32767) |

Both grids: 11264x16384 pixels, origin (1000000, 6338400), 100m spacing, EPSG:2193.

## Vs30 (Band 1)

```
Valid pixels:      25,909,493 / 184,549,376

Reference range:   [122.8423, 803.7499], mean=425.3998
Reproduced range:  [123.5832, 773.0162], mean=425.6214

Max abs diff:      2.680923e+02
Mean abs diff:     8.767391e-01
Median abs diff:   9.155273e-05
Std abs diff:      5.864140e+00
Max rel diff:      87.44%
Mean rel diff:     0.30%

% pixels |diff| > 1e-05   : 88.08%
% pixels |diff| > 0.001   : 16.16%
% pixels |diff| > 0.1     : 13.62%
% pixels |diff| > 1.0     : 8.37%
% pixels |diff| > 10.0    : 1.81%

Max diff at index 23651071: reference=521.1074, reproduced=253.0151
```

## Standard Deviation (Band 2)

```
Valid pixels:      25,909,493 / 184,549,376

Reference range:   [0.0801, 0.9239], mean=0.4862
Reproduced range:  [0.0670, 0.9240], mean=0.4857

Max abs diff:      4.851381e-01
Mean abs diff:     1.736444e-03
Median abs diff:   1.648068e-05
Std abs diff:      1.132851e-02
Max rel diff:      104.95%
Mean rel diff:     0.53%

% pixels |diff| > 1e-05   : 92.99%
% pixels |diff| > 0.001   : 9.35%
% pixels |diff| > 0.1     : 0.30%
% pixels |diff| > 1.0     : 0.00%
% pixels |diff| > 10.0    : 0.00%

Max diff at index 3648759: reference=0.7132, reproduced=0.2281
```

## Interpretation

- **Median diffs are negligible** (9e-5 m/s for Vs30, 2e-5 for StdDev), indicating the
  vast majority of pixels match almost exactly.
- **88-93% of pixels differ by >1e-5**: consistent with float32 (published) vs float64
  (reproduced) precision differences.
- **8.4% of Vs30 pixels differ by >1 m/s, 1.8% by >10 m/s**: these are concentrated
  near Vs30 observation sites where the MVN spatial adjustment is active. Likely sources:
  differences in the observation set, observation coordinates, or MVN numerical precision.
- **Max Vs30 diff of 268 m/s** (ref=521, repro=253): a localized outlier, possibly a pixel
  where an observation is included/excluded differently or where MVN conditioning diverges.
- **StdDev is tighter**: only 0.3% of pixels differ by >0.1, and none by >1.0.

## Reproduction Command

```bash
vs30 grid-custom \
    --grid-xmin 1000000 --grid-xmax 2126400 \
    --grid-ymin 4700000 --grid-ymax 6338400 \
    --grid-dx 100 --grid-dy 100 \
    --config foster_2019 \
    --include-intermediate \
    /home/arr65/data/vs30/grid_models/foster_2019_reproduced
```

## Date

Comparison performed 2026-03-28.
