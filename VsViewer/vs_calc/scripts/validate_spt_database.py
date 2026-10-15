"""Validate SPT correlations against NZGD SQLite records without changing the DB.

Run with ``PYTHONPATH=VsViewer python -m vs_calc.scripts.validate_spt_database
DATABASE --unit-weights CSV --output-dir DIRECTORY``. See
``VsViewer/docs/effective_stress_validation.md`` for data assumptions.
"""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from vs_calc import SPT, SPT_CORRELATIONS, VsProfile
from vs_calc.constants import SoilType


class InvalidRecord(ValueError):
    """A record cannot supply unambiguous inputs for this validation."""


def numeric_value(value, default):
    """Convert a report field, falling back only for absent/non-finite values."""
    value = pd.to_numeric(value, errors="coerce")
    return float(value) if pd.notna(value) and np.isfinite(value) else default


def complete_soil_profile(soils, minimum_depth):
    """Fill gaps with the nearest logged interval, switching at gap midpoints."""
    soils = soils.sort_values("top_depth_m").copy()
    missing_bottoms = int(soils["bottom_depth_m"].isna().sum())
    soils["bottom_depth_m"] = soils["bottom_depth_m"].fillna(
        soils["top_depth_m"]
        .shift(-1)
        .fillna(max(minimum_depth, soils["top_depth_m"].iloc[-1] + 0.01))
    )
    if (
        not np.isfinite(soils["bottom_depth_m"]).all()
        or (soils["bottom_depth_m"] <= soils["top_depth_m"]).any()
    ):
        raise InvalidRecord("non-positive or invalid logged layer thickness")

    intervals = []
    overlaps_merged = 0
    for top, bottom, soil in soils[
        ["top_depth_m", "bottom_depth_m", "soil_type"]
    ].itertuples(index=False, name=None):
        if intervals and top < intervals[-1][1] - 1e-8:
            if soil != intervals[-1][2]:
                raise InvalidRecord(
                    "overlapping logged intervals with different soil types"
                )
            intervals[-1][1] = max(intervals[-1][1], bottom)
            overlaps_merged += 1
        else:
            intervals.append([top, bottom, soil])

    # In each known gap [previous bottom, next top], the closest logged soil
    # changes halfway across. The logged intervals themselves retain their soil.
    boundaries = [0.0]
    gap_lengths = []
    for previous, current in zip(intervals, intervals[1:]):
        boundaries.append((previous[1] + current[0]) / 2)
        gap_lengths.append(max(0.0, current[0] - previous[1]))
    boundaries.append(max(minimum_depth, intervals[-1][1]))
    profile = pd.DataFrame(
        {
            "top_depth_m": boundaries[:-1],
            "bottom_depth_m": boundaries[1:],
            "soil_type": [interval[2] for interval in intervals],
        }
    )
    assumptions = {
        "surface_extension_m": float(intervals[0][0]),
        "bottom_extension_m": float(boundaries[-1] - intervals[-1][1]),
        "interval_gaps_filled": sum(length > 1e-8 for length in gap_lengths),
        "gap_depth_filled_m": float(sum(gap_lengths)),
        "overlapping_same_soil_intervals_merged": overlaps_merged,
        "layer_bottoms_inferred": missing_bottoms,
        "profile_bottom_m": float(boundaries[-1]),
    }
    return profile, assumptions


def load_record(conn, report, unit_weights):
    """Prepare one report and retain the assumptions used to complete its profile."""
    measurements = pd.read_sql_query(
        """SELECT depth_m, ISPT_MAIN, ISPT_NVAL FROM sptmeasurements
           WHERE spt_id = ? ORDER BY depth_m, spt_measurement_id""",
        conn,
        params=(report["spt_id"],),
    ).apply(pd.to_numeric, errors="coerce")
    # NVAL is the explicit N-value field. MAIN also contains PDF-extracted N.
    measurements["n"] = measurements["ISPT_NVAL"].where(
        np.isfinite(measurements["ISPT_NVAL"]), measurements["ISPT_MAIN"]
    )
    valid = (
        np.isfinite(measurements["depth_m"])
        & np.isfinite(measurements["n"])
        & (measurements["depth_m"] >= 0)
        & (measurements["n"] >= 0)
    )
    invalid_measurements = int((~valid).sum())
    measurements = measurements[valid].sort_values("depth_m")
    if (measurements.groupby("depth_m")["n"].nunique() > 1).any():
        raise InvalidRecord("conflicting N values at the same depth")
    duplicate_measurements = int(measurements.duplicated("depth_m").sum())
    measurements = measurements.drop_duplicates("depth_m")
    positive = measurements[measurements["n"] > 0]
    if len(positive) < 2 or positive["depth_m"].max() < 5:
        raise InvalidRecord(
            "fewer than two positive N values or insufficient depth for Vs30"
        )

    soils = pd.read_sql_query(
        """SELECT sm.soil_measurement_id, sm.top_depth_m, sm.bottom_depth_m,
                  st.value AS soil_type
           FROM soilmeasurements sm
           LEFT JOIN soilmeasurementsoiltype j USING (soil_measurement_id)
           LEFT JOIN soiltypes st ON st.id = j.soil_type_id
           WHERE sm.spt_id = ? ORDER BY sm.top_depth_m, sm.soil_measurement_id""",
        conn,
        params=(report["spt_id"],),
    )
    soils["top_depth_m"] = pd.to_numeric(soils["top_depth_m"], errors="coerce")
    soils["bottom_depth_m"] = pd.to_numeric(soils["bottom_depth_m"], errors="coerce")
    if (
        soils.empty
        or soils["soil_type"].isna().any()
        or not np.isfinite(soils["top_depth_m"]).all()
        or (soils["top_depth_m"] < 0).any()
    ):
        raise InvalidRecord("missing or invalid soil-layer data")
    if (soils.groupby("top_depth_m")["soil_type"].nunique() != 1).any():
        raise InvalidRecord("ambiguous soil types at a layer top")

    soils = soils.groupby("top_depth_m", as_index=False).agg(
        bottom_depth_m=("bottom_depth_m", "max"), soil_type=("soil_type", "first")
    )
    soils, layer_assumptions = complete_soil_profile(
        soils, float(measurements["depth_m"].max()) + 0.3048 + 0.01
    )
    layers = pd.DataFrame(
        {
            "layer_thickness_m": (
                soils["bottom_depth_m"] - soils["top_depth_m"]
            ).to_numpy()
        }
    )
    soil_names = soils["soil_type"].str.lower()
    for column in ("unsaturated_unit_weight_kN/m3", "saturated_unit_weight_kN/m3"):
        layers[column] = soil_names.map(unit_weights[column]).to_numpy()
    if not np.isfinite(layers.to_numpy()).all():
        raise InvalidRecord("soil type is missing from the unit-weight table")

    soil_at_measurements = (
        np.searchsorted(
            soils["top_depth_m"].to_numpy(),
            measurements["depth_m"].to_numpy(),
            side="right",
        )
        - 1
    )
    soil_types = np.array(
        [
            SoilType.__members__.get(name.title(), SoilType.Clay)
            for name in soils["soil_type"].iloc[soil_at_measurements]
        ]
    )
    groundwater = numeric_value(report["extracted_gwl_m"], 2.0)
    diameter = numeric_value(report["borehole_diameter"], 150.0)
    efficiency = numeric_value(report["efficiency"], 75.0)
    if groundwater < 0 or diameter <= 0 or efficiency <= 0:
        raise InvalidRecord("negative groundwater or non-positive diameter/efficiency")
    spt = SPT(
        name=str(report["spt_id"]),
        depth=measurements["depth_m"].to_numpy(),
        n=measurements["n"].to_numpy(),
        energy_ratio=efficiency,
        borehole_diameter=diameter,
        soil_type=soil_types,
        layers=layers,
        groundwater_level=groundwater,
    )
    assumptions = {
        "spt_id": int(report["spt_id"]),
        "nzgd_id": int(report["nzgd_id"]),
        "measurement_count": len(measurements),
        "nval_count": int(measurements["ISPT_NVAL"].notna().sum()),
        "nval_main_disagreements": int(
            (
                measurements["ISPT_NVAL"].notna()
                & measurements["ISPT_MAIN"].notna()
                & (measurements["ISPT_NVAL"] != measurements["ISPT_MAIN"])
            ).sum()
        ),
        "invalid_measurements_removed": invalid_measurements,
        "duplicate_measurements_removed": duplicate_measurements,
        "zero_n_count": int((measurements["n"] == 0).sum()),
        "groundwater_m": groundwater,
        "groundwater_assumed": not np.isfinite(
            numeric_value(report["extracted_gwl_m"], np.nan)
        ),
        "energy_ratio_percent": efficiency,
        "energy_ratio_assumed": not np.isfinite(
            numeric_value(report["efficiency"], np.nan)
        ),
        "borehole_diameter_mm": diameter,
        "diameter_assumed": not np.isfinite(
            numeric_value(report["borehole_diameter"], np.nan)
        ),
        **layer_assumptions,
        "soil_types": sorted(soils["soil_type"].unique().tolist()),
        "unit_weights_saturated_below_unsaturated": bool(
            (
                layers["saturated_unit_weight_kN/m3"]
                < layers["unsaturated_unit_weight_kN/m3"]
            ).any()
        ),
    }
    return spt, layers, assumptions


def reference_stress(depths, layers, groundwater):
    """Independently integrate submerged/dry interval lengths at each depth."""
    result = []
    for depth in depths:
        stress = 0.0
        top = 0.0
        for thickness, dry_weight, saturated_weight in layers.itertuples(
            index=False, name=None
        ):
            bottom = top + thickness
            dry_length = max(0.0, min(depth, bottom, groundwater) - top)
            wet_length = max(0.0, min(depth, bottom) - max(top, groundwater))
            stress += dry_length * dry_weight + wet_length * (saturated_weight - 9.81)
            top = bottom
        result.append(stress)
    return np.array(result)


def validate_record(spt, layers, assumptions):
    """Check stress, N60, Vs, sigma, serialization and both Vs30 extrapolations."""
    depths = spt.depth[spt.N60 > 0]
    expected_stress = reference_stress(depths + 0.3048, layers, spt.groundwater_level)
    rod_factor = np.select(
        [spt.depth < 3, spt.depth < 4, spt.depth < 6, spt.depth < 10],
        [0.75, 0.8, 0.85, 0.95],
        default=1.0,
    )
    diameter_factor = (
        1.0
        if 65 <= spt.borehole_diameter <= 115
        else 1.15
        if spt.borehole_diameter == 200
        else 1.05
    )
    expected_n60 = spt.N * (spt.energy_ratio / 60) * diameter_factor * rod_factor
    # N60 is rounded to two decimals by the legacy API. Compare with the
    # unrounded formula so binary halfway rounding cannot alter the oracle.
    np.testing.assert_allclose(spt.N60, expected_n60, atol=0.0050000001, rtol=0)
    restored = SPT.from_json(json.loads(json.dumps(spt.to_json())))
    legacy = SPT(
        spt.name,
        spt.depth,
        spt.N,
        energy_ratio=spt.energy_ratio,
        borehole_diameter=spt.borehole_diameter,
        soil_type=spt.soil_type,
        groundwater_level=spt.groundwater_level,
    )
    rows = []
    # Independent coefficient tables: (b0, b1, b2, tau).
    coefficients = {
        "brandenberg_2010": {
            SoilType.Sand: (4.045, 0.096, 0.236, 0.217),
            SoilType.Silt: (3.783, 0.178, 0.231, 0.227),
            SoilType.Clay: (3.996, 0.230, 0.164, 0.227),
        },
        "kwak_2015": {
            SoilType.Sand: (3.913, 0.167, 0.216, 0.217),
            SoilType.Silt: (3.879, 0.255, 0.168, 0.227),
            SoilType.Gravel: (3.840, 0.154, 0.285, 0.369),
            SoilType.Clay: (4.119, 0.209, 0.165, 0.227),
        },
    }
    for name, correlation in SPT_CORRELATIONS.items():
        vs, sd, returned_depths, stress = correlation(spt)
        np.testing.assert_allclose(stress, expected_stress, atol=1e-10, rtol=1e-12)
        np.testing.assert_array_equal(returned_depths, depths)
        expected_vs, expected_sd = [], []
        for n60, soil, stress_value in zip(
            spt.N60[spt.N60 > 0], spt.soil_type[spt.N60 > 0], expected_stress
        ):
            effective_soil = soil if soil in coefficients[name] else SoilType.Clay
            b0, b1, b2, tau = coefficients[name][effective_soil]
            intercept, gradient, above_200 = {
                SoilType.Sand: (0.57, 0.07, 0.2),
                SoilType.Silt: (0.31, 0.03, 0.15),
                SoilType.Gravel: (0.31, 0.03, 0.15),
                SoilType.Clay: (0.21, 0.01, 0.16),
            }[effective_soil]
            sigma = (
                intercept - gradient * np.log(stress_value)
                if stress_value <= 200
                else above_200
            )
            expected_vs.append(
                np.exp(b0 + b1 * np.log(n60) + b2 * np.log(stress_value))
            )
            expected_sd.append(np.hypot(tau, sigma))
        np.testing.assert_allclose(vs, expected_vs, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(sd, expected_sd, atol=1e-12, rtol=1e-12)
        for actual, round_trip in zip(
            (vs, sd, returned_depths, stress), correlation(restored)
        ):
            np.testing.assert_allclose(round_trip, actual, atol=1e-10, rtol=1e-12)
        for vs30_name, min_depth in (("boore_2011", 5), ("boore_2004", 10)):
            if depths[-1] < min_depth:
                continue
            profile = VsProfile.from_spt(spt, name, vs30_name)
            original_profile = VsProfile.from_spt(legacy, name, vs30_name)
            assert np.isfinite(profile.vs30) and profile.vs30 > 0
            assert np.isfinite(profile.vs30_sd) and profile.vs30_sd >= 0
            rows.append(
                {
                    "spt_id": assumptions["spt_id"],
                    "nzgd_id": assumptions["nzgd_id"],
                    "spt_correlation": name,
                    "vs30_correlation": vs30_name,
                    "positive_n_count": len(depths),
                    "max_measurement_depth_m": float(depths[-1]),
                    "groundwater_m": spt.groundwater_level,
                    "groundwater_assumed": assumptions["groundwater_assumed"],
                    "vs30_m_per_s": float(profile.vs30),
                    "vs30_legacy_soil_stress_m_per_s": float(original_profile.vs30),
                    "max_stress_error_kpa": float(
                        np.max(np.abs(stress - expected_stress))
                    ),
                    "max_vs_error_m_per_s": float(np.max(np.abs(vs - expected_vs))),
                    "max_sigma_error": float(np.max(np.abs(sd - expected_sd))),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--unit-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=25, help="Number of valid SPT reports to check"
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    weights = pd.read_csv(args.unit_weights)
    weights["soil_type"] = weights["soil_type"].str.lower()
    weights = weights.set_index("soil_type")
    before = args.database.stat()
    assumptions, skipped, rows = [], [], []
    with closing(
        sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        conn.execute("PRAGMA query_only = ON")
        conn.row_factory = sqlite3.Row
        reports = conn.execute("""
            SELECT r.* FROM sptreport r
            WHERE EXISTS (SELECT 1 FROM sptmeasurements m WHERE m.spt_id = r.spt_id)
              AND EXISTS (SELECT 1 FROM soilmeasurements s WHERE s.spt_id = r.spt_id)
            ORDER BY (r.extracted_gwl_m IS NOT NULL) DESC, r.spt_id
        """).fetchall()
        for report in reports:
            try:
                spt, layers, record_assumptions = load_record(conn, report, weights)
            except InvalidRecord as exc:
                skipped.append({"spt_id": report["spt_id"], "reason": str(exc)})
                continue
            rows.extend(validate_record(spt, layers, record_assumptions))
            assumptions.append(record_assumptions)
            if len(assumptions) >= args.limit:
                break
    if len(assumptions) < args.limit:
        raise RuntimeError(
            f"Only {len(assumptions)} eligible reports for requested {args.limit}"
        )
    after = args.database.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    results = pd.DataFrame(rows)
    summary = {
        "database": str(args.database.resolve()),
        "database_opened_read_only": True,
        "database_size_and_mtime_unchanged": True,
        "unit_weights": str(args.unit_weights.resolve()),
        "gap_fill_rule": "nearest logged interval; switch at gap midpoint",
        "reports_validated": len(assumptions),
        "spt_measurements": sum(record["measurement_count"] for record in assumptions),
        "positive_n_measurements": sum(
            record["measurement_count"] - record["zero_n_count"]
            for record in assumptions
        ),
        "reports_with_measured_groundwater": sum(
            not record["groundwater_assumed"] for record in assumptions
        ),
        "vs30_estimates": len(rows),
        "max_stress_error_kpa": float(results["max_stress_error_kpa"].max()),
        "max_vs_error_m_per_s": float(results["max_vs_error_m_per_s"].max()),
        "max_sigma_error": float(results["max_sigma_error"].max()),
        "assumptions": assumptions,
        "skipped_records": skipped,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "spt_validation.csv", index=False)
    (args.output_dir / "spt_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("assumptions", "skipped_records")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
