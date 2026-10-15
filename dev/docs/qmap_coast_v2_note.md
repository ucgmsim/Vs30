# Vs30 v2 — alignment between qmap and coast

A note for v2 design, prompted by an issue that surfaced while running the
refactored Python pipeline at the full-NZ 100 m grid.

## Background

The pipeline's gap-fill stage classifies a pixel as fillable if all three
conditions hold:

1. The combined Vs30 is NaN.
2. The geology id (`gid`) is not water — i.e., `gid != 0`.
3. The pixel's NZTM coordinates fall inside the NZ coastline polygon.

Conditions 2 and 3 use **two independently digitised shapefiles**:
`qmap.shp` for `gid`, and `coast.shp` for the polygon test. They agree
about where NZ is to within ~99.99 %, but disagree at pixel-precision in
some places along the boundary.

## Observation

When the full-NZ pipeline runs:

- ~83 % of the bounding-box pixels have `gid = 255`
  (`RASTER_ID_NODATA_VALUE` — "no qmap polygon covered this pixel").
  Almost all of these are sea, well outside NZ.
- A small fraction of `gid = 255` pixels lie **inside** the coast
  polygon. On the 5000 m benchmark this is 18 pixels; on the 100 m
  full-grid it scales to a few hundred to a few thousand.

Those are pixels where the coast outline says "this is NZ" but qmap has
no polygon there. They are most often:

- tiny offshore islets that are inside the coast polygon but absent
  from qmap;
- urban / built-up areas where qmap has no geology classification;
- rasterisation slivers along the coast where the two shapefiles'
  digitised boundaries cross at slightly different places.

## What the current code does with them

It fills them. The gap-fill nearest-neighbour pass copies the Vs30 of
the closest pixel that *does* have a qmap classification. So the output
Vs30 at those pixels has provenance:

> "qmap doesn't tell us what rock is here, so we copied the Vs30 of the
> nearest pixel where qmap did."

That is a defensible choice — but right now it's an emergent property
of two shapefiles disagreeing, not a deliberate decision.

## The question for v2

**Which shapefile is the source of truth for "is this pixel land"?**

There are three coherent answers:

| Option | What it means | Effect on gap-fill |
|---|---|---|
| **A. qmap is canonical** | Fill all qmap holes inside NZ in v2 so `gid = 255` reliably means "outside NZ" | The expensive polygon test becomes redundant; gap-fill is just `gid != 0 and gid != 255` |
| **B. coast is canonical** | Rasterise `coast.shp` once and use it as the on-land mask; ignore qmap for gap-fill scope | Current behaviour, made explicit and self-documenting |
| **C. agree to disagree** | Never fill `gid = 255` — if qmap has no classification, leave NaN | Slightly more pixels stay NaN (the few hundred above); no nearest-neighbour fabrication where geology is unknown |

Each choice is scientifically valid; they differ in what they say about
the meaning of the output at the qmap-vs-coast disagreement set.

## Implementation upside (small, mention only if relevant)

Option A makes the gap-fill polygon test redundant, which would cut
gap-fill memory peak by ~3 GB and wall time from ~30 s to <1 s on the
full-NZ 100 m run. The savings are real but tiny next to MVN, which
dominates total runtime by ~50×. Worth knowing about if v2 is already
revisiting qmap; not worth a data pass on its own.

## Why this is worth raising

The current behaviour emerges from two independent shapefiles agreeing
99.99 % of the time. That's fragile: re-issuing either shapefile (which
v2 may well do) can silently shift the set of disagreement pixels.
Picking a source of truth in v2 makes the choice explicit and stable
across future data revisions, regardless of which option is chosen.
