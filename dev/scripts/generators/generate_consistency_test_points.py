"""
Generate the consistency test points CSV fixture.

Produces tests/fixtures/consistency_test_points.csv containing ~40 on-land
points used by test_grid_points_consistency.py. Run manually whenever the
point set needs updating:

    python dev/scripts/generators/generate_consistency_test_points.py

Deliberate points target specific geology categories, coastal zones, cities,
observation-sparse areas, and geology boundaries. Random points fill the
remaining coverage uniformly across the NZ domain.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from qcore import coordinates

from vs30 import category, constants, raster

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
OUTPUT_CSV = FIXTURES_DIR / "consistency_test_points.csv"

# Number of random points to generate (after land filtering)
N_RANDOM = 25


def _snap_to_pixel_center(easting: float, northing: float) -> tuple[float, float]:
    """Snap to the nearest grid node aligned with FULL_NZ_GRID_CONFIG,
    matching the snap logic in gapfill.create_local_grid_config.
    """
    nz = constants.FULL_NZ_GRID_CONFIG
    snap_e = nz.grid_xmin + round((easting - nz.grid_xmin) / nz.grid_dx) * nz.grid_dx
    snap_n = nz.grid_ymin + round((northing - nz.grid_ymin) / nz.grid_dy) * nz.grid_dy
    return snap_e, snap_n


def _nztm_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """Convert a single NZTM point to WGS84 (lat, lon)."""
    result = coordinates.nztm_to_wgs_depth(np.array([[northing, easting]]))
    return float(result[0, 0]), float(result[0, 1])


def _wgs84_to_nztm(lat: float, lon: float) -> tuple[float, float]:
    """Convert a single WGS84 point to NZTM (easting, northing)."""
    result = coordinates.wgs_depth_to_nztm(np.array([[lat, lon]]))
    return float(result[0, 1]), float(result[0, 0])


def _load_coastline() -> shapely.Geometry:
    """Load and union the NZ coastline polygon."""
    coastline_path = constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH
    raster.ensure_shapefile_extracted(coastline_path, "coast")
    coast_gdf = gpd.read_file(coastline_path)
    return coast_gdf.geometry.union_all()


def _geology_id_at(easting: float, northing: float) -> int:
    """Return the geology category ID at a single NZTM point."""
    ids = category.assign_to_category_geology(np.array([[easting, northing]]))
    return int(ids[0])


def _find_point_with_geology_id(
    target_gid: int,
    coast_union: shapely.Geometry,
    rng: np.random.Generator,
    max_attempts: int = 5000,
) -> tuple[float, float] | None:
    """Random-sample until we find an on-land point with the given geology ID."""
    nz = constants.FULL_NZ_GRID_CONFIG
    for _ in range(max_attempts):
        e = rng.uniform(nz.grid_xmin, nz.grid_xmax)
        n = rng.uniform(nz.grid_ymin, nz.grid_ymax)
        e, n = _snap_to_pixel_center(e, n)
        pt = shapely.Point(e, n)
        if not shapely.within(pt, coast_union):
            continue
        if _geology_id_at(e, n) == target_gid:
            return e, n
    return None


def generate_deliberate_points(
    coast_union: shapely.Geometry,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate deliberately chosen test points."""
    points = []

    # --- Major cities (WGS84 lat/lon -> NZTM -> snap) ---
    cities = {
        "Auckland": (-36.8485, 174.7633),
        "Wellington": (-41.2865, 174.7762),
        "Christchurch": (-43.5321, 172.6362),
        "Dunedin": (-45.8788, 170.5028),
        "Hamilton": (-37.7870, 175.2793),
    }
    for name, (lat, lon) in cities.items():
        e, n = _wgs84_to_nztm(lat, lon)
        e, n = _snap_to_pixel_center(e, n)
        lat_s, lon_s = _nztm_to_wgs84(e, n)
        points.append(
            {"name": name.lower(), "longitude": lon_s, "latitude": lat_s, "category": "city"}
        )

    # --- Rare geology categories ---
    rare_gids = {
        1: "peat",
        5: "lacustrine",
        9: "outwash",
        14: "volcanic",
    }
    for gid, description in rare_gids.items():
        result = _find_point_with_geology_id(gid, coast_union, rng)
        if result is not None:
            e, n = result
            lat, lon = _nztm_to_wgs84(e, n)
            points.append(
                {
                    "name": f"geology_gid{gid}_{description}",
                    "longitude": lon,
                    "latitude": lat,
                    "category": "rare_geology",
                }
            )
        else:
            print(f"WARNING: Could not find point for GID {gid} ({description})")

    # --- Coastal-sensitive geology (GID 4 alluvium, GID 10 flood plain near coast) ---
    coastal_gids = {4: "alluvium_coastal", 10: "flood_plain_coastal"}
    coast_boundary = coast_union.boundary
    for gid, description in coastal_gids.items():
        # Find a point with this GID that is within 5 km of coast
        nz = constants.FULL_NZ_GRID_CONFIG
        for _ in range(5000):
            e_r = rng.uniform(nz.grid_xmin, nz.grid_xmax)
            n_r = rng.uniform(nz.grid_ymin, nz.grid_ymax)
            e_r, n_r = _snap_to_pixel_center(e_r, n_r)
            pt = shapely.Point(e_r, n_r)
            if not shapely.within(pt, coast_union):
                continue
            if shapely.distance(pt, coast_boundary) > 5000:
                continue
            if _geology_id_at(e_r, n_r) == gid:
                lat, lon = _nztm_to_wgs84(e_r, n_r)
                points.append(
                    {
                        "name": f"geology_gid{gid}_{description}",
                        "longitude": lon,
                        "latitude": lat,
                        "category": "coastal_sensitive",
                    }
                )
                break
        else:
            print(f"WARNING: Could not find coastal point for GID {gid}")

    # --- Observation-sparse area (Fiordland / remote West Coast) ---
    # Pick a point in the southwest corner of the South Island
    fiordland_lat, fiordland_lon = -45.5, 167.0
    e, n = _wgs84_to_nztm(fiordland_lat, fiordland_lon)
    e, n = _snap_to_pixel_center(e, n)
    lat, lon = _nztm_to_wgs84(e, n)
    points.append(
        {
            "name": "fiordland_sparse",
            "longitude": lon,
            "latitude": lat,
            "category": "observation_sparse",
        }
    )

    # --- Near geology boundary ---
    # Find two adjacent pixels with different geology IDs
    nz = constants.FULL_NZ_GRID_CONFIG
    for _ in range(5000):
        e_r = rng.uniform(nz.grid_xmin + 200, nz.grid_xmax - 200)
        n_r = rng.uniform(nz.grid_ymin + 200, nz.grid_ymax - 200)
        e_r, n_r = _snap_to_pixel_center(e_r, n_r)
        pt = shapely.Point(e_r, n_r)
        if not shapely.within(pt, coast_union):
            continue
        gid_center = _geology_id_at(e_r, n_r)
        if gid_center == 0 or gid_center == constants.RASTER_ID_NODATA_VALUE:
            continue
        # Check the pixel 100m to the east
        gid_east = _geology_id_at(e_r + 100, n_r)
        if gid_east != gid_center and gid_east != 0 and gid_east != constants.RASTER_ID_NODATA_VALUE:
            lat, lon = _nztm_to_wgs84(e_r, n_r)
            points.append(
                {
                    "name": f"geology_boundary_gid{gid_center}_gid{gid_east}",
                    "longitude": lon,
                    "latitude": lat,
                    "category": "geology_boundary",
                }
            )
            break
    else:
        print("WARNING: Could not find geology boundary point")

    return points


def generate_random_points(
    n: int,
    coast_union: shapely.Geometry,
    rng: np.random.Generator,
    existing_eastings_northings: list[tuple[float, float]],
) -> list[dict]:
    """Generate random on-land points, avoiding duplicates with existing points."""
    nz = constants.FULL_NZ_GRID_CONFIG
    existing_set = set(existing_eastings_northings)
    points = []

    # Generate more candidates than needed, filter to on-land
    while len(points) < n:
        batch_size = n * 10
        eastings = rng.uniform(nz.grid_xmin, nz.grid_xmax, size=batch_size)
        northings = rng.uniform(nz.grid_ymin, nz.grid_ymax, size=batch_size)

        for e, n_coord in zip(eastings, northings):
            e, n_coord = _snap_to_pixel_center(e, n_coord)
            if (e, n_coord) in existing_set:
                continue
            pt = shapely.Point(e, n_coord)
            if not shapely.within(pt, coast_union):
                continue
            # Exclude water pixels (GID 0)
            gid = _geology_id_at(e, n_coord)
            if gid == 0 or gid == constants.RASTER_ID_NODATA_VALUE:
                continue
            lat, lon = _nztm_to_wgs84(e, n_coord)
            points.append(
                {
                    "name": f"random_{len(points):02d}",
                    "longitude": lon,
                    "latitude": lat,
                    "category": "random",
                }
            )
            existing_set.add((e, n_coord))
            if len(points) >= n:
                break

    return points


def main():
    print("Loading coastline...")
    coast_union = _load_coastline()

    rng = np.random.default_rng(42)

    print("Generating deliberate points...")
    deliberate = generate_deliberate_points(coast_union, rng)
    print(f"  Generated {len(deliberate)} deliberate points")

    # Collect NZTM coordinates of deliberate points to avoid duplicates
    existing_en = []
    for p in deliberate:
        e, n = _wgs84_to_nztm(p["latitude"], p["longitude"])
        existing_en.append(_snap_to_pixel_center(e, n))

    print(f"Generating {N_RANDOM} random points...")
    random_pts = generate_random_points(N_RANDOM, coast_union, rng, existing_en)
    print(f"  Generated {len(random_pts)} random points")

    all_points = deliberate + random_pts
    df = pd.DataFrame(all_points)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(df)} points to {OUTPUT_CSV}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
