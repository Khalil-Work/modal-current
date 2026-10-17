"""Tests of the shared input-validation helpers."""

import numpy as np
import pytest

from modal_current.validation import (
    validate_finite_scalar,
    validate_positive_scalar,
    validate_snapshot_matrix,
    validate_time_vector,
)


class TestSnapshotMatrix:
    def test_accepts_valid_matrix(self):
        data = np.random.default_rng(0).normal(size=(4, 6))
        out = validate_snapshot_matrix(data, min_samples=2)
        assert out.shape == (4, 6)
        assert out.dtype == np.float64

    def test_accepts_minimum_sample_count(self):
        assert validate_snapshot_matrix(np.zeros((3, 1)), min_samples=1).shape == (3, 1)

    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError, match="2-D"):
            validate_snapshot_matrix(np.zeros(10))

    def test_rejects_three_dimensional_input(self):
        with pytest.raises(ValueError, match="2-D"):
            validate_snapshot_matrix(np.zeros((2, 3, 4)))

    def test_rejects_empty_matrix(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_snapshot_matrix(np.empty((0, 3)))

    def test_rejects_nan(self):
        data = np.zeros((4, 6))
        data[2, 3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            validate_snapshot_matrix(data)

    def test_rejects_inf(self):
        data = np.zeros((4, 6))
        data[0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            validate_snapshot_matrix(data)

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            validate_snapshot_matrix(np.zeros((4, 1)), min_samples=2)

    def test_coerces_integer_input(self):
        out = validate_snapshot_matrix(np.ones((2, 3), dtype=int))
        assert out.dtype == np.float64


class TestPositiveScalar:
    def test_accepts_positive(self):
        assert validate_positive_scalar(0.025, "dt") == 0.025
        assert validate_positive_scalar(1, "dt") == 1.0
        assert validate_positive_scalar(1e-12, "dt") == 1e-12

    @pytest.mark.parametrize("bad", [0.0, -0.05, float("nan"), float("inf")])
    def test_rejects_invalid_values(self, bad):
        with pytest.raises(ValueError):
            validate_positive_scalar(bad, "dt")

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="real scalar"):
            validate_positive_scalar("quarter", "dt")


class TestFiniteScalar:
    def test_accepts_any_finite(self):
        assert validate_finite_scalar(-0.05, "gamma") == -0.05
        assert validate_finite_scalar(0.0, "phase") == 0.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError, match="finite"):
            validate_finite_scalar(bad, "gamma")

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="real scalar"):
            validate_finite_scalar(None, "gamma")


class TestTimeVector:
    def test_accepts_valid_vector(self):
        out = validate_time_vector([0.0, 0.1, 0.2])
        assert out.shape == (3,)

    def test_rejects_two_dimensional(self):
        with pytest.raises(ValueError, match="1-D"):
            validate_time_vector(np.zeros((2, 2)))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_time_vector([])

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="finite"):
            validate_time_vector([0.0, np.nan])
