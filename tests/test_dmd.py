"""Exact-DMD verification tests (SYNTHETIC demonstration data).

Because the field is a sum of separable *standing* modes, exact DMD is run
on delay-embedded (Hankel) snapshots with ``delays=2``; the plain variant
(``delays=1``) is covered by a dedicated test documenting why the embedding
is required (see ``docs/model.md``).
"""

import numpy as np
import pytest

from modal_current.dmd import compute_dmd, hankel_embed, match_eigenvalues
from modal_current.synthetic import default_times, generate_snapshots


def test_hankel_embed_structure(snapshots):
    n_x, n_t = snapshots.shape
    embedded = hankel_embed(snapshots, 2)
    assert embedded.shape == (2 * n_x, n_t - 1)
    np.testing.assert_array_equal(embedded[:n_x, :], snapshots[:, :-1])
    np.testing.assert_array_equal(embedded[n_x:, :], snapshots[:, 1:])


def test_hankel_embed_rejects_excessive_delays(snapshots):
    with pytest.raises(ValueError, match="delays"):
        hankel_embed(snapshots[:, :2], 2)
    with pytest.raises(ValueError, match="integer"):
        hankel_embed(snapshots, 0)


def test_dmd_result_fields(snapshots):
    result = compute_dmd(snapshots, dt=0.05, delays=2)
    assert result.rank == 6  # 3 modes -> 3 conjugate pairs
    assert result.n_delays == 2
    assert result.n_snapshots == snapshots.shape[1]
    assert result.eigenvalues.shape == (6,)
    assert result.modes.shape == (snapshots.shape[0], 6)
    assert result.amplitudes.shape == (6,)


def test_recovers_known_frequencies_and_growth(snapshots, analytic_modes):
    growth, frequency = analytic_modes
    result = compute_dmd(snapshots, dt=0.05, delays=2)
    idx = match_eigenvalues(result, growth, frequency)
    for k, i in enumerate(idx):
        omega_err = abs(result.frequencies[i] - frequency[k]) / abs(frequency[k])
        gamma_err = abs(result.growth_rates[i] - growth[k])
        assert omega_err < 1e-10, (k, result.frequencies[i])
        assert gamma_err < 1e-10, (k, result.growth_rates[i])


def test_eigenvalues_form_conjugate_pairs(snapshots):
    result = compute_dmd(snapshots, dt=0.05, delays=2)
    for mu in result.eigenvalues:
        assert np.abs(result.eigenvalues - np.conj(mu)).min() < 1e-12


def test_training_reconstruction_is_roundoff_accurate(snapshots, times):
    result = compute_dmd(snapshots, dt=0.05, delays=2)
    rel = np.linalg.norm(result.reconstruct(times) - snapshots) / np.linalg.norm(snapshots)
    assert rel < 1e-9
    np.testing.assert_allclose(
        result.reconstruct_training(), result.reconstruct(times), rtol=1e-10, atol=1e-14
    )


def test_extrapolation_beyond_the_training_window(field):
    train_times = default_times(dt=0.05, t_end=6.0)
    result = compute_dmd(field.evaluate(train_times), dt=0.05, delays=2)
    held_out = 6.0 + 0.05 * np.arange(1, 41)  # 40 steps past the window
    truth = field.evaluate(held_out)
    pred = result.reconstruct(held_out)
    rel = np.linalg.norm(pred - truth) / np.linalg.norm(truth)
    assert rel < 1e-8


def test_noise_robustness_band(field):
    """Documented small noise (1% of field std, seed 0): recovered
    frequencies stay within a 1% band and growth rates within 0.03 absolute."""
    times = default_times(dt=0.05, t_end=6.0)
    noisy = generate_snapshots(field, times, noise_level=0.01, seed=0)
    result = compute_dmd(noisy, dt=0.05, rank=6, delays=2)
    growth = [-0.05, 0.02, -0.12]
    frequency = [2.0, 5.0, 9.5]
    idx = match_eigenvalues(result, growth, frequency)
    for k, i in enumerate(idx):
        omega_err = abs(result.frequencies[i] - frequency[k]) / abs(frequency[k])
        gamma_err = abs(result.growth_rates[i] - growth[k])
        assert omega_err < 0.01, (k, result.frequencies[i])
        assert gamma_err < 0.03, (k, result.growth_rates[i])


def test_plain_dmd_without_embedding_is_biased_for_standing_modes(
    snapshots, analytic_modes
):
    """Plain exact DMD (delays=1) cannot represent standing oscillations.

    The trajectory of K separable standing modes spans only K spatial
    dimensions while the dynamics need 2K state dimensions, so the linear
    fit is inconsistent; this test documents that failure honestly and is
    the reason the verification uses ``delays=2``.
    """
    growth, frequency = analytic_modes
    result = compute_dmd(snapshots, dt=0.05, delays=1)
    assert result.rank == 3  # only the spatial span is seen
    idx = match_eigenvalues(result, growth, frequency, positive_only=False)
    worst = max(
        abs(result.frequencies[i] - frequency[k]) / abs(frequency[k])
        for k, i in enumerate(idx)
    )
    assert worst > 1e-2
    rel = np.linalg.norm(result.reconstruct(times_of(snapshots)) - snapshots) / np.linalg.norm(snapshots)
    assert rel > 0.5


def times_of(snapshots):
    return 0.05 * np.arange(snapshots.shape[1])


class TestDMDValidation:
    def test_rejects_nonpositive_dt(self, snapshots):
        with pytest.raises(ValueError, match="dt"):
            compute_dmd(snapshots, dt=0.0)
        with pytest.raises(ValueError, match="dt"):
            compute_dmd(snapshots, dt=-0.05)
        with pytest.raises(ValueError, match="dt"):
            compute_dmd(snapshots, dt=float("nan"))

    def test_rejects_single_snapshot(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_dmd(np.ones((8, 1)), dt=0.05)

    def test_rejects_non_finite_data(self, snapshots):
        with pytest.raises(ValueError, match="finite"):
            compute_dmd(snapshots + np.inf, dt=0.05)

    def test_rejects_invalid_rank(self, snapshots):
        with pytest.raises(ValueError, match="positive integer"):
            compute_dmd(snapshots, dt=0.05, delays=2, rank=0)
        with pytest.raises(ValueError, match="<="):
            compute_dmd(snapshots, dt=0.05, delays=2, rank=10_000)

    def test_rejects_invalid_delays(self, snapshots):
        with pytest.raises(ValueError, match="delays"):
            compute_dmd(snapshots, dt=0.05, delays=0)
        with pytest.raises(ValueError, match="delays"):
            compute_dmd(snapshots, dt=0.05, delays=1.5)

    def test_rejects_non_finite_t0(self, snapshots):
        with pytest.raises(ValueError, match="t0"):
            compute_dmd(snapshots, dt=0.05, t0=float("inf"))


def test_match_eigenvalues_validates_inputs(snapshots):
    result = compute_dmd(snapshots, dt=0.05, delays=2)
    with pytest.raises(ValueError, match="equal length"):
        match_eigenvalues(result, [-0.05], [2.0, 5.0])
