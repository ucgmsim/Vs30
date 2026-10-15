"""Tests for the provisional Vs30 uncertainty."""

import numpy as np
import pytest
from vs_calc import VsProfile, calculate_weighted_vs30, constants


def constant_profile(max_depth, vs30_correlation="boore_2011", vs=250.0, name="site"):
    """Constant-velocity profile sampled every 0.5 m down to max_depth."""
    depth = np.arange(0.5, max_depth + 0.25, 0.5)
    return VsProfile(
        name,
        np.full(depth.shape, vs),
        np.zeros(depth.shape),
        depth,
        vs30_correlation=vs30_correlation,
    )


def test_30m_profile_has_only_the_profile_sigma():
    profile = constant_profile(30)
    assert profile.vs30 == pytest.approx(250.0)
    assert profile.vs30_sd == pytest.approx(constants.PROFILE_SIGMA_LN_VS30)


def test_boore_2011_sigma_at_5m():
    assert constant_profile(5).vs30_sd == pytest.approx(
        np.sqrt(0.2**2 + (np.log(10) * 0.119) ** 2)
    )


def test_boore_2004_sigma_at_10m():
    assert constant_profile(10, "boore_2004").vs30_sd == pytest.approx(
        np.sqrt(0.2**2 + (np.log(10) * 0.07126) ** 2)
    )


@pytest.mark.parametrize(
    "vs30_correlation,shallowest", [("boore_2011", 5), ("boore_2004", 10)]
)
def test_sigma_decreases_with_depth(vs30_correlation, shallowest):
    sigmas = [
        constant_profile(depth, vs30_correlation).vs30_sd
        for depth in range(shallowest, 31)
    ]
    assert (np.diff(sigmas) < 0).all()


def test_sigma_uses_the_integer_depth_below_a_fractional_maximum():
    depth = np.append(np.arange(0.5, 25.6, 0.5), 25.97)
    profile = VsProfile(
        "site",
        np.full(depth.shape, 250.0),
        np.zeros(depth.shape),
        depth,
        vs30_correlation="boore_2011",
    )
    assert profile.max_depth == 25
    assert profile.vs30_sd == pytest.approx(constant_profile(25).vs30_sd)


@pytest.mark.parametrize(
    "vs30_correlation,max_depth", [("boore_2011", 4), ("boore_2004", 9)]
)
def test_profiles_too_shallow_for_the_relation_still_raise(vs30_correlation, max_depth):
    with pytest.raises(IndexError, match="not deep enough"):
        constant_profile(max_depth, vs30_correlation).vs30_sd


def test_weighted_sigma_of_identical_profiles_is_the_profile_sigma():
    profiles = [
        constant_profile(30, None, name="a"),
        constant_profile(30, None, name="b"),
    ]
    _, sigma = calculate_weighted_vs30(profiles, {"a": 0.5, "b": 0.5}, {}, {}, {})
    assert sigma == pytest.approx(constants.PROFILE_SIGMA_LN_VS30)


def test_weighted_sigma_adds_the_spread_of_ln_vs30():
    profiles = [
        constant_profile(30, None, vs=200.0, name="a"),
        constant_profile(30, None, vs=400.0, name="b"),
    ]
    _, sigma = calculate_weighted_vs30(profiles, {"a": 0.5, "b": 0.5}, {}, {}, {})
    assert sigma == pytest.approx(np.sqrt(0.2**2 + (np.log(2) / 2) ** 2))


@pytest.mark.parametrize(
    "vs30_correlation,expected_vs30",
    [("boore_2011", 312.1085397086086), ("boore_2004", 253.6328922676476)],
)
def test_central_vs30_is_unchanged(vs30_correlation, expected_vs30):
    depth = np.arange(0.5, 12.5, 0.5)
    profile = VsProfile(
        "site",
        150 + 10 * depth,
        np.zeros(depth.shape),
        depth,
        vs30_correlation=vs30_correlation,
    )
    assert profile.vs30 == pytest.approx(expected_vs30, rel=1e-12)
