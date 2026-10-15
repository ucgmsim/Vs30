"""Regression tests for SPT construction and layer-based correlations."""

import json

import numpy as np
import pandas as pd
import pytest

from vs_calc import SPT, SPT_CORRELATIONS, VsProfile
from vs_calc.constants import SoilType
from vs_calc.spt_vs_correlations import (
    calculate_effective_stress,
    effective_stress_brandenberg,
    effective_stress_kwak,
)


@pytest.fixture
def layers():
    return pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 16.0, 19.0],
            "saturated_unit_weight_kN/m3": [20.0, 18.0, 21.0],
        }
    )


@pytest.mark.parametrize("layers", [None, pd.DataFrame()])
def test_legacy_spt_defaults_and_empty_layers(layers):
    spt = SPT("legacy", [1.0, 3.0, 5.0], [10, 20, 30], layers=layers)
    np.testing.assert_allclose(spt.N60, [6.3, 13.44, 21.42])
    assert spt.layers is None
    for correlation in SPT_CORRELATIONS.values():
        vs, sd, depth, stress = correlation(spt)
        np.testing.assert_allclose(stress, [20.8768, 42.686312, 59.066312])
        np.testing.assert_array_equal(depth, spt.depth)
        assert np.isfinite(vs).all() and (vs > 0).all()
        assert np.isfinite(sd).all() and (sd > 0).all()


@pytest.mark.parametrize("correlation", SPT_CORRELATIONS.values())
def test_groundwater_is_used_without_layers(correlation):
    spt = SPT("surface water", [1.0], [10], groundwater_level=0.0)
    np.testing.assert_allclose(correlation(spt)[3], [1.3048 * 8.19])


@pytest.mark.parametrize(
    "correlation", [effective_stress_brandenberg, effective_stress_kwak]
)
@pytest.mark.parametrize(
    "depth, expected",
    [
        (0.5, 9.0),
        (1.0, 18.0),
        (1.5, 26.0),
        (2.0, 30.095),
        (3.0, 38.285),
        (4.0, 49.475),
        (5.0, 60.665),
    ],
)
def test_stress_at_interfaces_and_inside_layers(layers, correlation, depth, expected):
    spt = SPT("layered", [0.5, 4.5], [10, 20], layers=layers, groundwater_level=1.5)
    stress, sigma, *_ = calculate_effective_stress(
        depth, SoilType.Clay, spt, correlation
    )
    assert stress == pytest.approx(expected)
    assert sigma == pytest.approx(0.21 - 0.01 * np.log(expected))


@pytest.mark.parametrize(
    "name, coefficients",
    [("brandenberg_2010", (3.996, 0.230, 0.164)), ("kwak_2015", (4.119, 0.209, 0.165))],
)
def test_correlations_use_penetration_offset_and_layer_sigma(
    layers, name, coefficients
):
    spt = SPT(
        "offset",
        [0.1952, 1.6952, 4.6952],
        [10, 20, 30],
        layers=layers,
        groundwater_level=1.5,
    )
    vs, sd, depth, stress = SPT_CORRELATIONS[name](spt)
    expected_stress = np.array([9.0, 30.095, 60.665])
    b0, b1, b2 = coefficients
    expected_vs = np.exp(
        b0 + b1 * np.log([6.3, 12.6, 21.42]) + b2 * np.log(expected_stress)
    )
    expected_sigma = 0.21 - 0.01 * np.log(expected_stress)
    np.testing.assert_allclose(stress, expected_stress)
    np.testing.assert_allclose(vs, expected_vs)
    np.testing.assert_allclose(sd, np.sqrt(0.227**2 + expected_sigma**2))
    np.testing.assert_array_equal(depth, spt.depth)


@pytest.mark.parametrize(
    "correlation", [effective_stress_brandenberg, effective_stress_kwak]
)
@pytest.mark.parametrize("stress", [100.0, 300.0])
@pytest.mark.parametrize("soil", list(SoilType))
def test_sigma_uses_supplied_stress_and_preserves_coefficients(
    correlation, stress, soil
):
    result = correlation(10.0, soil, effective_stress=stress)
    assert result[0] == stress
    effective_soil = (
        SoilType.Clay
        if correlation is effective_stress_brandenberg and soil == SoilType.Gravel
        else soil
    )
    intercept, gradient, high_stress_sigma = {
        SoilType.Clay: (0.21, 0.01, 0.16),
        SoilType.Sand: (0.57, 0.07, 0.2),
        SoilType.Silt: (0.31, 0.03, 0.15),
        SoilType.Gravel: (0.31, 0.03, 0.15),
    }[effective_soil]
    assert result[1] == pytest.approx(
        intercept - gradient * np.log(stress) if stress <= 200 else high_stress_sigma
    )
    assert result[2:] == correlation(10.0, soil)[2:]


def test_layer_stress_rejects_depth_beyond_profile(layers):
    spt = SPT("too deep", [5.0], [10], layers=layers)
    with pytest.raises(ValueError, match="outside the layer profile"):
        SPT_CORRELATIONS["kwak_2015"](spt)


@pytest.mark.parametrize("layered", [False, True])
def test_json_round_trip(layers, layered):
    spt = SPT(
        "round trip",
        [0.5, 4.5],
        [10, 20],
        layers=layers if layered else None,
        groundwater_level=1.5,
    )
    restored = SPT.from_json(json.loads(json.dumps(spt.to_json())))
    for correlation in SPT_CORRELATIONS.values():
        for expected, actual in zip(correlation(spt), correlation(restored)):
            np.testing.assert_allclose(actual, expected)


def test_old_json_without_optional_fields():
    payload = SPT("old client", [1.0, 3.0], [10, 20]).to_json()
    for name in ("layers", "groundwater_level", "N60"):
        payload.pop(name)
    payload["borehole_diameter"] = None
    restored = SPT.from_json(payload)
    np.testing.assert_allclose(restored.N60, [6.3, 13.44])


def test_existing_csv_and_web_upload_paths(tmp_path):
    path = tmp_path / "measurements.csv"
    path.write_text("Depth,NValue,Soil\n1.0,10,Sand\n3.0,20,Clay\n")
    from_file = SPT.from_file(str(path))
    from_form = SPT.from_byte_stream_form(
        path.name,
        path.read_bytes(),
        {"hammerType": "", "boreholeDiameter": "", "energyRatio": "", "soilType": ""},
    )
    np.testing.assert_allclose(from_file.N60, from_form.N60)
    assert from_form.to_json()["soil_type"] == ["Sand", "Clay"]


def test_energy_ratio_uses_percent():
    spt = SPT("efficiency", [1.0], [10], energy_ratio=75.0, borehole_diameter=None)
    np.testing.assert_allclose(spt.N60, [9.84])


def test_end_to_end_vs30_from_layered_spt():
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [35.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    spt = SPT("deep", [1.0, 10.0, 20.0, 30.0], [10, 20, 30, 40], layers=layers)
    for name in SPT_CORRELATIONS:
        profile = VsProfile.from_spt(spt, name)
        assert np.isfinite(profile.vs30) and profile.vs30 > 0
        assert np.isfinite(profile.vs30_sd)
