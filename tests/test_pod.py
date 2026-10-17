"""POD verification tests (SYNTHETIC demonstration data).

With noise-free data built from ``r`` separable modes the centered snapshot
matrix has rank exactly ``r``: the whole trajectory lives in the
``r``-dimensional span of the analytic shapes.
"""

import numpy as np
import pytest

from modal_current.pod import compute_pod, numerical_rank


def test_energy_fractions_sum_to_one(snapshots):
    result = compute_pod(snapshots)
    np.testing.assert_allclose(result.energy_fractions.sum(), 1.0, rtol=0, atol=1e-12)


def test_noise_free_rank_equals_number_of_modes(snapshots):
    result = compute_pod(snapshots)
    assert result.rank == 3
    # the next singular value sits at machine-precision roundoff level
    s = result.singular_values
    assert s[3] < 1e-12 * s[0]
    assert numerical_rank(s, snapshots.shape) == 3


def test_pod_basis_reproduces_the_analytic_subspace(field, snapshots):
    result = compute_pod(snapshots)
    for mode in field.modes:
        projected = result.spatial_modes.T @ mode.shape
        capture = float(np.linalg.norm(projected) ** 2 / np.linalg.norm(mode.shape) ** 2)
        assert capture > 0.99


def test_pod_modes_are_orthonormal(snapshots):
    result = compute_pod(snapshots)
    gram = result.spatial_modes.T @ result.spatial_modes
    np.testing.assert_allclose(gram, np.eye(result.rank), atol=1e-12)


def test_full_rank_reconstruction_is_roundoff_accurate(snapshots):
    result = compute_pod(snapshots)
    rel = np.linalg.norm(result.reconstruct() - snapshots) / np.linalg.norm(snapshots)
    assert rel < 1e-12


def test_truncated_reconstruction_error_equals_discarded_energy(snapshots):
    result = compute_pod(snapshots)
    centered_norm2 = np.linalg.norm(snapshots - result.mean[:, np.newaxis]) ** 2
    error2 = np.linalg.norm(result.reconstruct(rank=2) - snapshots) ** 2
    np.testing.assert_allclose(
        error2 / centered_norm2,
        1.0 - result.energy_fractions[:2].sum(),
        rtol=1e-10,
        atol=1e-14,
    )


def test_temporal_coefficients_are_uncorrelated_with_norms_sigma(snapshots):
    result = compute_pod(snapshots)
    gram = result.temporal_coefficients @ result.temporal_coefficients.T
    np.testing.assert_allclose(
        gram, np.diag(result.singular_values[: result.rank] ** 2), rtol=1e-10, atol=1e-12
    )


def test_mean_is_the_temporal_average(snapshots):
    result = compute_pod(snapshots)
    np.testing.assert_allclose(result.mean, snapshots.mean(axis=1), rtol=0, atol=0)


def test_constant_data_has_zero_rank_and_perfect_mean_reconstruction():
    data = np.full((16, 5), 3.25)
    result = compute_pod(data)
    assert result.rank == 0
    assert np.all(result.energy_fractions == 0.0)
    np.testing.assert_allclose(result.reconstruct(), data)


def test_numerical_rank_threshold_behaviour():
    s = np.array([1.0, 1e-3, 3e-17, 2e-18])
    assert numerical_rank(s, (1000, 400)) == 2
    assert numerical_rank(np.array([0.0, 0.0]), (4, 2)) == 0


class TestPODValidation:
    def test_rejects_one_dimensional_data(self, snapshots):
        with pytest.raises(ValueError):
            compute_pod(snapshots.ravel())

    def test_rejects_non_finite_data(self, snapshots):
        with pytest.raises(ValueError, match="finite"):
            compute_pod(snapshots + np.nan)

    def test_rejects_empty_data(self):
        with pytest.raises(ValueError):
            compute_pod(np.empty((4, 0)))

    def test_rejects_invalid_ranks(self, snapshots):
        with pytest.raises(ValueError, match="positive integer"):
            compute_pod(snapshots, rank=0)
        with pytest.raises(ValueError, match="positive integer"):
            compute_pod(snapshots, rank=2.5)
        with pytest.raises(ValueError, match="<="):
            compute_pod(snapshots, rank=10_000)
