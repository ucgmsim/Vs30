"""
Tests for the VS30 parallel processing module.

Tests cover:
- resolve_n_proc function
- single_threaded_blas context manager
- Parallel processing infrastructure
"""

import multiprocessing as mp
import os

import numpy as np
import pytest

from vs30.parallel import (
    resolve_n_proc,
    single_threaded_blas,
)


class TestResolveNProc:
    """Tests for the resolve_n_proc function."""

    def test_none_returns_one(self):
        """Test that None returns 1 (single process)."""
        assert resolve_n_proc(None) == 1

    def test_one_returns_one(self):
        """Test that 1 returns 1."""
        assert resolve_n_proc(1) == 1

    def test_minus_one_returns_cpu_count(self):
        """Test that -1 returns CPU count."""
        expected = mp.cpu_count()
        assert resolve_n_proc(-1) == expected

    def test_explicit_value_returned(self):
        """Test that explicit values are returned (up to CPU count)."""
        # Small value should be returned as-is
        assert resolve_n_proc(2) == min(2, mp.cpu_count())
        assert resolve_n_proc(4) == min(4, mp.cpu_count())

    def test_large_value_capped_at_cpu_count(self):
        """Test that values larger than CPU count are capped."""
        large_value = mp.cpu_count() + 100
        assert resolve_n_proc(large_value) == mp.cpu_count()

    def test_zero_raises_error(self):
        """Test that 0 raises ValueError."""
        with pytest.raises(ValueError):
            resolve_n_proc(0)

    def test_negative_less_than_minus_one_raises(self):
        """Test that values < -1 raise ValueError."""
        with pytest.raises(ValueError):
            resolve_n_proc(-2)

        with pytest.raises(ValueError):
            resolve_n_proc(-100)


class TestSingleThreadedBlas:
    """Tests for the single_threaded_blas context manager."""

    def test_context_manager_runs(self):
        """Test that context manager runs without error."""
        with single_threaded_blas():
            # Just verify it executes
            result = np.dot(np.array([1, 2, 3]), np.array([4, 5, 6]))
            assert result == 32

    def test_numpy_operations_work(self):
        """Test that numpy operations work inside context."""
        with single_threaded_blas():
            # Matrix multiplication (uses BLAS)
            a = np.random.rand(100, 100)
            b = np.random.rand(100, 100)
            c = np.dot(a, b)

            assert c.shape == (100, 100)

    def test_context_manager_is_reentrant(self):
        """Test that context manager can be nested."""
        with single_threaded_blas():
            with single_threaded_blas():
                result = np.sum(np.array([1, 2, 3]))
                assert result == 6

    def test_eigenvalue_computation(self):
        """Test that eigenvalue computation works (BLAS-heavy)."""
        with single_threaded_blas():
            # Symmetric matrix for real eigenvalues
            a = np.random.rand(50, 50)
            a = (a + a.T) / 2  # Make symmetric

            eigenvalues, eigenvectors = np.linalg.eig(a)

            assert len(eigenvalues) == 50
            assert eigenvectors.shape == (50, 50)

    def test_linear_solve(self):
        """Test that linear system solving works."""
        with single_threaded_blas():
            # Solve Ax = b
            A = np.random.rand(20, 20)
            A = A + np.eye(20) * 10  # Make well-conditioned
            b = np.random.rand(20)

            x = np.linalg.solve(A, b)

            # Verify solution
            np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=10)


class TestParallelProcessingInfrastructure:
    """Tests for parallel processing infrastructure."""

    def test_spawn_context_available(self):
        """Test that spawn context is available."""
        from vs30.parallel import _spawn_context

        assert _spawn_context is not None
        # Should be able to create a Pool (just test creation, don't start)
        assert hasattr(_spawn_context, 'Pool')

    def test_multiprocessing_cpu_count(self):
        """Test that CPU count is available and reasonable."""
        cpu_count = mp.cpu_count()

        assert cpu_count >= 1
        assert cpu_count <= 1024  # Reasonable upper bound
