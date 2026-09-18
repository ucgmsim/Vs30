# Layer-based SPT effective stress

The `add-jaehwi-effstress` implementation supports optional soil layers and
groundwater in `SPT`, while preserving the default calculations without layers.
It evaluates stress at the requested depth, including the existing 0.3048 m
penetration offset, and uses that stress in both Vs and its uncertainty.

Missing or empty layers use the original soil-type approximation. Missing
groundwater and borehole diameter default to 2 m and 150 mm. Explicit groundwater
is respected with or without layers. `energy_ratio` is a percentage, e.g. `75.0`.
The water unit weight is defined locally; `vs_calc` no longer imports `nzgd`.
CSV/web imports and older JSON payloads remain supported.

Layer inputs contain `layer_thickness_m`, `unsaturated_unit_weight_kN/m3`, and
`saturated_unit_weight_kN/m3`, ordered continuously downwards from the surface.
The groundwater interface is split automatically. Stress is interpolated between
the resulting layer bottoms; this is exact for constant unit weights within each
layer. Calculations leave the input DataFrame intact. An out-of-profile stress
query raises `ValueError`, so callers must provide coverage through the deepest
measurement plus the penetration offset.

## Reproduce the checks

From the branch checkout, with the Vs30 Python environment activated:

```bash
python -m pytest VsViewer/tests -q

PYTHONPATH=VsViewer python -m vs_calc.scripts.validate_spt_database \
  /home/arr65/data/nzgd/dev_extracted_cpt_and_scpt_data/uc_nzgd_v0p8p2_20260709_deduped.db \
  --unit-weights /home/arr65/src/nzgd/nzgd/resources/soil_type_unit_weights.csv \
  --output-dir VsViewer/docs/validation \
  --limit 25
```

The database is opened with SQLite `mode=ro` and `query_only`. Its size and
modification time are also checked before and after validation. No NZGD estimator
script is imported or run. The added VsViewer CI job runs the tests with only
NumPy, pandas, and pytest installed; it does not require the external database.

## Database interpretation

The schema was checked against `nzgd/db/orm.py` and the actual SQLite tables.
Records with extracted groundwater are selected first, followed by SPT ID.
Only measurements and soil logs from the same `spt_id` are combined.

- Use finite `ISPT_NVAL` where present, otherwise `ISPT_MAIN`. `ISPT_REP` is not
  used. Record disagreements between the two selected fields. Exclude invalid
  depths/counts, collapse identical measurements at the same depth, and reject
  conflicting duplicates. Zero N values are retained in the input but excluded
  from logarithmic Vs correlations.
- Fill gaps between logged soil intervals with the nearest interval's soil type.
  Switch at the gap midpoint; a tie belongs to the deeper interval. Extend the
  nearest end layer to the surface or deepest required stress depth. Where only
  layer tops are recorded, infer each bottom from the next top, extending the
  final layer through the required depth.
- Merge overlapping intervals of the same soil. Skip ambiguous soil mixtures or
  overlaps of different soils instead of selecting an arbitrary constituent.
- Use the provided NZGD unit-weight lookup table unchanged. Map soil types to
  correlation coefficients at the recorded measurement depths. As in the
  existing estimator, types outside Clay/Silt/Sand/Gravel use clay coefficients;
  Brandenberg also retains its existing clay fallback for gravel.
- Use reported groundwater, efficiency, and diameter where finite. Missing values
  use the NZGD configuration defaults: 2 m, 75%, and 150 mm respectively. This
  validation's explicit 75% efficiency differs from the library's historical
  automatic-hammer factor when `energy_ratio` is omitted.

Every record's assumptions and any skipped records are saved in
[spt_validation.json](validation/spt_validation.json).

## Results

- 83 automated tests passed, including legacy callers, layer boundaries, offset
  depths, uncertainty, invalid inputs, JSON round trips, and nearest-layer gaps.
- Default behavior without layers matched `master` in 192 stress/coefficient
  cases and eight SPT profiles, at relative tolerance `1e-14`.
- 25 real SPT reports supplied 323 measurements, including three zero N values.
  Eight reports supplied extracted groundwater; 17 used the 2 m default.
- Both SPT correlations (Brandenberg 2010, Kwak 2015) and both Vs30 conversions
  (Boore 2011, Boore 2004) produced 100 finite positive Vs30 estimates.
- Independent interval integration agreed with computed stresses to a maximum
  absolute error of `1.14e-13 kPa`. Separate evaluation of the correlation
  equations agreed in Vs to `2.85e-13 m/s` and in total log uncertainty to
  `1.12e-16`. N60 agreed with its unrounded formula within its 0.005 rounding
  allowance. Serialization round trips preserved each profile.

Examples using Boore 2011, in m/s:

| SPT ID | NZGD ID | Groundwater (m) | Brandenberg Vs30 | Kwak Vs30 |
| --- | --- | --- | --- | --- |
| 802 | 14862 | 1.9 | 287.49 | 284.13 |
| 4369 | 143789 | 2.7 | 408.15 | 417.51 |
| 4388 | 144062 | 4.0 | 270.20 | 269.95 |
| 5590 | 179163 | 1.5 | 289.75 | 299.34 |

The complete estimates and comparison with the soil-type stress approximation
are in [spt_validation.csv](validation/spt_validation.csv).

These checks establish computational agreement using real records and explicit
input assumptions; they do not compare predictions with measured Vs profiles.
The supplied table has saturated unit weights below unsaturated weights for
gravel and silt, and four selected reports contain an extracted groundwater
depth of 75 m. Those inputs are retained and recorded, not independently
verified. The existing Kwak uncertainty approximation is preserved; this change
corrects which stress it uses, without recalibrating the empirical model.
