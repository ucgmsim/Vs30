"""The batch optimization must preserve the existing CPT stress integration."""

import numpy as np

from vs_calc import CPT


def test_unit_weight_computed_once_and_stress_unchanged():
    class CountingCPT(CPT):
        calls = 0

        @property
        def gamma(self):
            self.calls += 1
            return super().gamma

    depth = np.array([0.5, 1.0, 1.7, 2.0, 5.0, 12.0])
    cpt = CountingCPT(
        "regression",
        depth,
        np.array([2.0, 3.0, 4.0, 5.0, 3.0, 6.0]),
        np.array([0.03, 0.05, 0.08, 0.07, 0.02, 0.05]),
        np.array([0.005, 0.01, 0.04, 0.06, 0.07, 0.13]),
        ground_water_level=1.0,
    )
    gamma = CPT.gamma.fget(cpt)
    expected_total = np.zeros(len(depth))
    expected_total[0] = gamma[0] * depth[0]
    for index in range(1, len(depth)):
        expected_total[index] = (
            gamma[index] * (depth[index] - depth[index - 1]) + expected_total[index - 1]
        )
    expected_pore = np.zeros(len(depth))
    expected_pore[1:] = 0.00981 * np.maximum(depth[1:] - 1.0, 0)
    _, effective, _, _, total = cpt.calc_cpt_params()
    assert cpt.calls == 1
    np.testing.assert_array_equal(total, expected_total)
    np.testing.assert_array_equal(effective, expected_total - expected_pore)
