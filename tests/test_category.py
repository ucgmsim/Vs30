"""Tests for the VS30 category module."""

import numpy as np
import pandas as pd
import pytest

from vs30 import category, constants


class TestUpdateWithIndependentData:
    """Tests for the update_with_independent_data function."""

    @pytest.fixture
    def sample_categorical_model(self):
        """Create sample categorical model DataFrame."""
        return pd.DataFrame(
            {
                constants.STANDARD_ID_COLUMN: [1, 2, 3],
                constants.COL_MEAN: [200.0, 300.0, 400.0],
                constants.COL_STDV: [0.5, 0.4, 0.3],
            }
        )

    @pytest.fixture
    def sample_observations(self):
        """Create sample observations DataFrame."""
        return pd.DataFrame(
            {
                constants.STANDARD_ID_COLUMN: [1, 1, 2],
                constants.ObservationColumn.VS30: [210.0, 195.0, 320.0],
                constants.ObservationColumn.UNCERTAINTY: [0.2, 0.2, 0.15],
            }
        )

    def test_basic_update(self, sample_categorical_model, sample_observations):
        """Test basic Bayesian update with observations."""
        result = category.update_with_independent_data(
            sample_categorical_model,
            sample_observations,
        )

        assert constants.COL_POSTERIOR_MEAN_INDEPENDENT in result.columns
        assert constants.COL_POSTERIOR_STDV_INDEPENDENT in result.columns
        assert constants.COL_POSTERIOR_NOBS_INDEPENDENT in result.columns

        # Category 1 has 2 observations on top of N_PRIOR=3, so n=5.
        cat1 = result[result[constants.STANDARD_ID_COLUMN] == 1].iloc[0]
        assert cat1[constants.COL_POSTERIOR_NOBS_INDEPENDENT] == constants.N_PRIOR + 2

        # Category 3 has no observations, so it keeps its prior.
        cat3 = result[result[constants.STANDARD_ID_COLUMN] == 3].iloc[0]
        assert cat3[constants.COL_POSTERIOR_MEAN_INDEPENDENT] == 400.0


class TestUpdateWithClusteredData:
    """Tests for update_with_clustered_data and compute_cluster_weighted_mean_and_stddev."""

    @pytest.fixture
    def prior_df(self):
        """Two-category prior model for cluster-update tests."""
        return pd.DataFrame(
            {
                constants.STANDARD_ID_COLUMN: [1, 2],
                constants.COL_MEAN: [300.0, 400.0],
                constants.COL_STDV: [0.5, 0.4],
            }
        )

    def test_one_cluster_plus_one_unclustered(self, prior_df):
        """Two clusters + one unclustered point: effective_n = 3."""
        sites_df = pd.DataFrame(
            {
                constants.STANDARD_ID_COLUMN: [1] * 6,
                constants.ObservationColumn.VS30: [200.0, 220.0, 240.0, 300.0, 320.0, 400.0],
                constants.ObservationColumn.CLUSTER: [0, 0, 0, 1, 1, -1],
            }
        )

        result_df = category.update_with_clustered_data(prior_df, sites_df)

        cat1 = result_df[result_df[constants.STANDARD_ID_COLUMN] == 1].iloc[0]
        cat2 = result_df[result_df[constants.STANDARD_ID_COLUMN] == 2].iloc[0]

        # Hand-computed expected mean for category 1:
        # weighted_log_sum = (log200 + log220 + log240) / 3
        #                 + (log300 + log320) / 2
        #                 + log400
        # log_geo_mean = weighted_log_sum / effective_n  (effective_n = 3)
        # mean = exp(log_geo_mean)
        log_sum = (
            (np.log(200) + np.log(220) + np.log(240)) / 3
            + (np.log(300) + np.log(320)) / 2
            + np.log(400)
        )
        expected_mean = float(np.exp(log_sum / 3))
        assert cat1[constants.COL_POSTERIOR_MEAN_CLUSTERED] == pytest.approx(
            expected_mean, rel=1e-6
        )
        # Category 2 has no observations: posterior should equal prior.
        assert cat2[constants.COL_POSTERIOR_MEAN_CLUSTERED] == pytest.approx(400.0)

    def test_all_in_one_cluster(self, prior_df):
        """Single cluster: posterior mean is the geometric mean of cluster members."""
        sites_df = pd.DataFrame(
            {
                constants.STANDARD_ID_COLUMN: [1, 1, 1, 1],
                constants.ObservationColumn.VS30: [200.0, 220.0, 240.0, 260.0],
                constants.ObservationColumn.CLUSTER: [0, 0, 0, 0],
            }
        )

        result_df = category.update_with_clustered_data(prior_df, sites_df)
        cat1 = result_df[result_df[constants.STANDARD_ID_COLUMN] == 1].iloc[0]

        # Cluster mean is geometric mean of [200, 220, 240, 260].
        expected_mean = float(np.exp(np.mean(np.log([200.0, 220.0, 240.0, 260.0]))))
        assert cat1[constants.COL_POSTERIOR_MEAN_CLUSTERED] == pytest.approx(
            expected_mean, rel=1e-6
        )
        # Stddev formula is complex for the single-cluster case; just check it's finite and non-negative.
        stddev = cat1[constants.COL_POSTERIOR_STDV_CLUSTERED]
        assert np.isfinite(stddev)
        assert stddev >= 0.0

    def test_unclustered_only(self, prior_df):
        """Unclustered only: posterior mean is the plain geometric mean of the points."""
        sites_df = pd.DataFrame(
            {
                constants.STANDARD_ID_COLUMN: [1, 1, 1],
                constants.ObservationColumn.VS30: [200.0, 220.0, 240.0],
                constants.ObservationColumn.CLUSTER: [-1, -1, -1],
            }
        )

        result_df = category.update_with_clustered_data(prior_df, sites_df)
        cat1 = result_df[result_df[constants.STANDARD_ID_COLUMN] == 1].iloc[0]

        expected_mean = float(np.exp(np.mean(np.log([200.0, 220.0, 240.0]))))
        assert cat1[constants.COL_POSTERIOR_MEAN_CLUSTERED] == pytest.approx(
            expected_mean, rel=1e-6
        )

    def test_only_nodata_category_skips(self, prior_df):
        """Sites whose only category is RASTER_ID_NODATA_VALUE should be skipped."""
        sites_df = pd.DataFrame(
            {
                constants.STANDARD_ID_COLUMN: [
                    constants.RASTER_ID_NODATA_VALUE,
                    constants.RASTER_ID_NODATA_VALUE,
                ],
                constants.ObservationColumn.VS30: [200.0, 220.0],
                constants.ObservationColumn.CLUSTER: [-1, -1],
            }
        )

        result_df = category.update_with_clustered_data(prior_df, sites_df)

        # Both categories keep their priors.
        assert result_df.loc[
            result_df[constants.STANDARD_ID_COLUMN] == 1,
            constants.COL_POSTERIOR_MEAN_CLUSTERED,
        ].iloc[0] == pytest.approx(300.0)
        assert result_df.loc[
            result_df[constants.STANDARD_ID_COLUMN] == 2,
            constants.COL_POSTERIOR_MEAN_CLUSTERED,
        ].iloc[0] == pytest.approx(400.0)

    def test_compute_cluster_weighted_mean_and_stddev_basic(self):
        """Direct test of the per-category weighted mean/stddev helper."""
        category_sites = pd.DataFrame(
            {
                constants.ObservationColumn.VS30: [200.0, 220.0, 240.0, 400.0],
                constants.ObservationColumn.CLUSTER: [0, 0, 0, -1],
            }
        )
        cluster_counts = category_sites[
            constants.ObservationColumn.CLUSTER
        ].value_counts()
        # 1 cluster + 1 unclustered point -> effective_n = 2.
        effective_n = 2

        mean, stddev = category.compute_cluster_weighted_mean_and_stddev(
            category_sites, cluster_counts, effective_n
        )

        # Expected: exp((mean(log[200,220,240]) + log400) / 2)
        log_sum = np.mean(np.log([200.0, 220.0, 240.0])) + np.log(400.0)
        expected_mean = float(np.exp(log_sum / 2))
        assert mean == pytest.approx(expected_mean, rel=1e-6)
        assert np.isfinite(stddev)
        assert stddev > 0.0
