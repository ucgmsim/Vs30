"""
Tests for the VS30 correlations module.

Tests cover:
- matern: Matérn correlation against R gstat reference values
"""

import numpy as np
import pytest

from vs30 import correlations


class TestMaternCorrelationFunction:
    """Tests for the Matérn correlation function."""

    def test_kappa_half_matches_exponential(self):
        """Matérn with kappa=0.5 and no nugget should match exponential.

        This is a mathematical identity: Matérn(κ=0.5) = exp(-d/range).
        Verifies our implementation against the known analytical result.
        """
        distances = np.array([100.0, 500.0, 1000.0, 5000.0])
        range_m = 993.0
        result = correlations.matern(
            distances,
            range_m=range_m,
            kappa=0.5,
        )
        expected = np.exp(-distances / range_m)
        np.testing.assert_allclose(result, expected, rtol=0.05)

    def test_matches_r_gstat_variogram_line(self):
        """Matérn correlation must match R gstat variogramLine output.

        Reference values generated with R 4.5.2 and gstat using:
            vgm(psill=0.15, model="Mat", range=20e3, nugget=0.05, kappa=0.9)
        Then divided by psill to obtain the correlation values used in the
        Worden et al. MVN formulation (see vs30/correlations.py docstring).
        """
        distances = np.array(
            [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 20000.0, 50000.0, 100000.0]
        )
        r_gstat_reference = np.array(
            [
                1.0,
                1.0,
                0.9999974,
                0.9998576,
                0.9933186,
                0.8000460,
                0.5647182,
                0.1637082,
                0.01697042,
            ]
        )
        result = correlations.matern(
            distances,
            range_m=20000.0,
            kappa=0.9,
        )
        np.testing.assert_allclose(result, r_gstat_reference, atol=1e-6)

    def test_zero_distance_gives_unit_correlation(self):
        """At zero distance, correlation must be ≈1.

        R gstat treats the nugget as contributing to per-point variance, not
        to the correlation function itself (Worden et al. Eq. 7), so the
        correlation-at-zero must be 1 regardless of variogram nugget. The
        correlation function does not take a nugget argument; this test
        documents the contract.
        """
        result = correlations.matern(
            np.array([0.0]),
            range_m=20000.0,
            kappa=0.9,
        )
        assert result[0] == pytest.approx(1.0, abs=1e-6)
