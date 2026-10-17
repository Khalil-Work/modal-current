"""Shared fixtures: a small synthetic problem that keeps the suite fast.

The default verification configuration (512 grid points, 401 snapshots) is
exercised by ``examples/run_verification.py``; the unit tests use a reduced
but analytically identical configuration (128 points, dt = 0.05, 121
snapshots).
"""

import pytest

from modal_current.synthetic import default_times, make_default_field


@pytest.fixture(scope="session")
def field():
    """Documented three-mode synthetic field on a 128-point grid."""
    return make_default_field(n_points=128)


@pytest.fixture(scope="session")
def times():
    """Uniform time grid: dt = 0.05 over [0, 6] (121 snapshots)."""
    return default_times(dt=0.05, t_end=6.0)


@pytest.fixture(scope="session")
def snapshots(field, times):
    """Noise-free synthetic snapshot matrix (SYNTHETIC demonstration data)."""
    return field.evaluate(times)


@pytest.fixture(scope="session")
def analytic_modes():
    from modal_current.synthetic import DEFAULT_MODE_PARAMETERS

    growth = [float(p["growth"]) for p in DEFAULT_MODE_PARAMETERS]
    frequency = [float(p["frequency"]) for p in DEFAULT_MODE_PARAMETERS]
    return growth, frequency
