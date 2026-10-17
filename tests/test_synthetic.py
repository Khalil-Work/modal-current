"""Tests of the synthetic field generator (SYNTHETIC demonstration data)."""

import numpy as np
import pytest

from modal_current.synthetic import (
    DEFAULT_MODE_PARAMETERS,
    ModeSpec,
    SyntheticField,
    default_grid,
    default_times,
    generate_snapshots,
    make_default_field,
    sine_shape,
)


def test_default_mode_parameters_are_the_documented_constants():
    assert [p["name"] for p in DEFAULT_MODE_PARAMETERS] == ["mode-1", "mode-2", "mode-3"]
    assert [p["amplitude"] for p in DEFAULT_MODE_PARAMETERS] == [1.0, 0.6, 0.35]
    assert [p["growth"] for p in DEFAULT_MODE_PARAMETERS] == [-0.05, 0.02, -0.12]
    assert [p["frequency"] for p in DEFAULT_MODE_PARAMETERS] == [2.0, 5.0, 9.5]
    assert [p["phase"] for p in DEFAULT_MODE_PARAMETERS] == [0.30, 1.10, 2.00]


def test_default_grid_is_cell_centred_on_the_unit_interval():
    grid = default_grid(n_points=512)
    assert grid.shape == (512,)
    assert grid.min() > 0.0 and grid.max() < 1.0
    np.testing.assert_allclose(np.diff(grid), 1.0 / 512)


def test_default_grid_rejects_bad_point_counts():
    with pytest.raises(ValueError):
        default_grid(n_points=1)
    with pytest.raises(ValueError):
        default_grid(n_points=7.5)


def test_default_times_are_uniform():
    times = default_times(dt=0.025, t_end=10.0)
    assert times.size == 401
    np.testing.assert_allclose(np.diff(times), 0.025)


def test_default_times_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        default_times(dt=0.0, t_end=1.0)


def test_evaluate_matches_the_closed_form_definition(field, times):
    u = field.evaluate(times)
    manual = np.zeros_like(u)
    for mode in field.modes:
        manual += (
            mode.amplitude
            * np.exp(mode.growth * times)[np.newaxis, :]
            * np.cos(mode.frequency * times + mode.phase)[np.newaxis, :]
            * mode.shape[:, np.newaxis]
        )
    np.testing.assert_allclose(u, manual, rtol=1e-13, atol=0.0)


def test_evaluate_shape_and_modal_contributions(field, times):
    u = field.evaluate(times)
    assert u.shape == (field.n_gridpoints, times.size)
    contributions = field.modal_contributions(times)
    assert contributions.shape == (field.n_modes, field.n_gridpoints, times.size)
    np.testing.assert_allclose(contributions.sum(axis=0), u, rtol=1e-12, atol=1e-15)


def test_generate_snapshots_noise_free_equals_analytic(field, times):
    np.testing.assert_array_equal(generate_snapshots(field, times), field.evaluate(times))


def test_noise_is_reproducible_and_seed_dependent(field, times):
    first = generate_snapshots(field, times, noise_level=0.01, seed=0)
    same = generate_snapshots(field, times, noise_level=0.01, seed=0)
    other = generate_snapshots(field, times, noise_level=0.01, seed=1)
    assert np.array_equal(first, same)
    assert not np.array_equal(first, other)


def test_noise_level_scales_with_field_std(field, times):
    """Added noise std must equal noise_level times the field std (~1%)."""
    clean = field.evaluate(times)
    noisy = generate_snapshots(field, times, noise_level=0.01, seed=0)
    ratio = float(np.std(noisy - clean) / np.std(clean))
    assert 0.008 < ratio < 0.012


def test_sine_shape_values_and_validation():
    grid = default_grid(n_points=64)
    shape = sine_shape(grid, 2)
    np.testing.assert_allclose(shape, np.sin(2.0 * np.pi * grid), rtol=0, atol=0)
    with pytest.raises(ValueError):
        sine_shape(grid, 0.0)
    with pytest.raises(ValueError):
        sine_shape(grid, -1)


class TestModeSpecValidation:
    def test_rejects_nonpositive_amplitude(self):
        with pytest.raises(ValueError, match="amplitude"):
            ModeSpec("m", 0.0, -0.05, 2.0, 0.3, np.ones(8))

    def test_rejects_nonpositive_frequency(self):
        with pytest.raises(ValueError, match="frequency"):
            ModeSpec("m", 1.0, -0.05, 0.0, 0.3, np.ones(8))

    def test_rejects_nonfinite_growth(self):
        with pytest.raises(ValueError, match="growth"):
            ModeSpec("m", 1.0, float("nan"), 2.0, 0.3, np.ones(8))

    def test_rejects_two_dimensional_shape(self):
        with pytest.raises(ValueError, match="1-D"):
            ModeSpec("m", 1.0, -0.05, 2.0, 0.3, np.ones((2, 8)))

    def test_rejects_zero_shape(self):
        with pytest.raises(ValueError, match="zero"):
            ModeSpec("m", 1.0, -0.05, 2.0, 0.3, np.zeros(8))


class TestSyntheticFieldValidation:
    def test_rejects_shape_grid_length_mismatch(self):
        field_ok = make_default_field(n_points=16)
        bad_mode = ModeSpec("bad", 1.0, 0.0, 1.0, 0.0, np.ones(field_ok.n_gridpoints + 1))
        with pytest.raises(ValueError, match="same length as the grid"):
            SyntheticField(grid=field_ok.grid, modes=field_ok.modes + (bad_mode,))

    def test_rejects_empty_mode_list(self):
        with pytest.raises(ValueError, match="at least one mode"):
            SyntheticField(grid=default_grid(8), modes=())

    def test_rejects_non_mode_entries(self):
        with pytest.raises(TypeError):
            SyntheticField(grid=default_grid(8), modes=("not-a-mode",))


def test_generate_snapshots_rejects_bad_arguments(field, times):
    with pytest.raises(ValueError, match="noise_level"):
        generate_snapshots(field, times, noise_level=-0.01)
    with pytest.raises(ValueError, match="noise_level"):
        generate_snapshots(field, times, noise_level=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        generate_snapshots(field, times, noise_level=0.01, seed="not-a-seed")


def test_evaluate_rejects_bad_times(field):
    with pytest.raises(ValueError, match="1-D"):
        field.evaluate(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        field.evaluate([0.0, np.nan, 0.1])
