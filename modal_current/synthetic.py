"""Synthetic spatiotemporal field with documented, known modal content.

Definition (all quantities nondimensional)::

    u(x, t) = sum_k  a_k * exp(gamma_k * t) * cos(omega_k * t + phi_k) * shape_k(x)

* ``x`` lives on a cell-centred grid covering the open interval (0, 1);
* ``t`` starts at 0 and advances in uniform steps ``dt``;
* ``a_k``     : mode amplitude;
* ``gamma_k`` : mode growth rate (negative = decaying, positive = growing);
* ``omega_k`` : angular frequency;
* ``phi_k``   : phase in radians;
* ``shape_k`` : spatial shape evaluated on the grid.

The default configuration is a three-mode field with distinct frequencies,
two decaying modes (one strongly, one mildly) and one slightly growing mode,
built from orthogonal sine shapes ``sin(k * pi * x)``; see
``DEFAULT_MODE_PARAMETERS`` for the exact constants.

SYNTHETIC DATA NOTICE
---------------------
Every array produced by this module is synthetic: it is constructed from the
documented formula above and involves no measurement, no simulation of a
physical system, and no real-world flow.  Results derived from it must be
labelled as synthetic demonstration data wherever they appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .validation import (
    validate_finite_scalar,
    validate_positive_scalar,
    validate_time_vector,
)

__all__ = [
    "DEFAULT_MODE_PARAMETERS",
    "ModeSpec",
    "SyntheticField",
    "default_grid",
    "default_times",
    "generate_snapshots",
    "make_default_field",
    "sine_shape",
]

#: Documented parameters of the default synthetic field (all nondimensional).
DEFAULT_MODE_PARAMETERS: tuple[dict[str, float | int | str], ...] = (
    {
        "name": "mode-1",
        "amplitude": 1.0,
        "growth": -0.05,
        "frequency": 2.0,
        "phase": 0.30,
        "wavenumber": 1,
    },
    {
        "name": "mode-2",
        "amplitude": 0.6,
        "growth": 0.02,
        "frequency": 5.0,
        "phase": 1.10,
        "wavenumber": 2,
    },
    {
        "name": "mode-3",
        "amplitude": 0.35,
        "growth": -0.12,
        "frequency": 9.5,
        "phase": 2.00,
        "wavenumber": 3,
    },
)


@dataclass(frozen=True)
class ModeSpec:
    """One mode of the synthetic field: documented parameters plus a shape."""

    name: str
    amplitude: float
    growth: float
    frequency: float
    phase: float
    shape: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ModeSpec.name must be a non-empty string")
        object.__setattr__(
            self, "amplitude", validate_positive_scalar(self.amplitude, "ModeSpec.amplitude")
        )
        object.__setattr__(
            self, "growth", validate_finite_scalar(self.growth, "ModeSpec.growth")
        )
        object.__setattr__(
            self, "frequency", validate_positive_scalar(self.frequency, "ModeSpec.frequency")
        )
        object.__setattr__(self, "phase", validate_finite_scalar(self.phase, "ModeSpec.phase"))
        shape = np.asarray(self.shape, dtype=float)
        if shape.ndim != 1 or shape.size < 1:
            raise ValueError("ModeSpec.shape must be a non-empty 1-D array")
        if not np.all(np.isfinite(shape)):
            raise ValueError("ModeSpec.shape must contain only finite values")
        if not np.any(shape != 0.0):
            raise ValueError("ModeSpec.shape must not be identically zero")
        object.__setattr__(self, "shape", shape)

    def temporal(self, times) -> np.ndarray:
        """Temporal factor ``a * exp(gamma t) * cos(omega t + phi)``."""
        t = validate_time_vector(times, name="times")
        return (
            self.amplitude
            * np.exp(self.growth * t)
            * np.cos(self.frequency * t + self.phase)
        )


@dataclass(frozen=True)
class SyntheticField:
    """A fully documented synthetic field on a fixed spatial grid."""

    grid: np.ndarray
    modes: tuple[ModeSpec, ...]

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=float)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("SyntheticField.grid must be a 1-D array with at least 2 points")
        if not np.all(np.isfinite(grid)):
            raise ValueError("SyntheticField.grid must contain only finite values")
        object.__setattr__(self, "grid", grid)
        modes_in = (
            (self.modes,) if isinstance(self.modes, ModeSpec) else tuple(self.modes)
        )
        if len(modes_in) == 0:
            raise ValueError("SyntheticField needs at least one mode")
        for mode in modes_in:
            if not isinstance(mode, ModeSpec):
                raise TypeError(
                    f"SyntheticField.modes entries must be ModeSpec; got {type(mode).__name__}"
                )
            if mode.shape.shape != grid.shape:
                raise ValueError(
                    "each mode shape must have the same length as the grid "
                    f"({grid.size}); got {mode.shape.size} for mode {mode.name!r}"
                )
        object.__setattr__(self, "modes", modes_in)

    @property
    def n_gridpoints(self) -> int:
        return self.grid.size

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def evaluate(self, times) -> np.ndarray:
        """Snapshot matrix ``u(x, t_j)`` of shape ``(n_gridpoints, n_times)``."""
        t = validate_time_vector(times, name="times")
        u = np.zeros((self.n_gridpoints, t.size), dtype=float)
        for mode in self.modes:
            u += mode.temporal(t)[np.newaxis, :] * mode.shape[:, np.newaxis]
        return u

    def modal_contributions(self, times) -> np.ndarray:
        """Per-mode fields, shape ``(n_modes, n_gridpoints, n_times)``."""
        t = validate_time_vector(times, name="times")
        return np.stack(
            [
                mode.temporal(t)[np.newaxis, :] * mode.shape[:, np.newaxis]
                for mode in self.modes
            ]
        )


def sine_shape(grid, wavenumber: float) -> np.ndarray:
    """Spatial shape ``sin(wavenumber * pi * x)`` evaluated on ``grid``."""
    grid = np.asarray(grid, dtype=float)
    k = validate_positive_scalar(wavenumber, "sine_shape wavenumber")
    return np.sin(k * np.pi * grid)


def default_grid(n_points: int = 512) -> np.ndarray:
    """Cell-centred grid on (0, 1) with ``n_points`` points.

    Cell centring makes integer-wavenumber sines exactly orthogonal under the
    discrete inner product, which keeps the verification of the POD subspace
    clean.
    """
    if not isinstance(n_points, (int, np.integer)) or int(n_points) < 2:
        raise ValueError(f"default_grid n_points must be an integer >= 2; got {n_points!r}")
    n = int(n_points)
    return (np.arange(n) + 0.5) / n


def default_times(dt: float = 0.025, t_end: float = 10.0) -> np.ndarray:
    """Uniform time grid ``t_j = j * dt`` covering ``[0, t_end]``."""
    dt = validate_positive_scalar(dt, "default_times dt")
    t_end = validate_finite_scalar(t_end, "default_times t_end")
    if t_end < dt:
        raise ValueError("default_times t_end must be at least one dt")
    n_steps = int(np.floor(t_end / dt + 1e-9))
    return dt * np.arange(n_steps + 1)


def make_default_field(
    n_points: int = 512,
    mode_parameters: Sequence[dict] | None = None,
) -> SyntheticField:
    """Build the documented default three-mode synthetic field."""
    grid = default_grid(n_points)
    params = DEFAULT_MODE_PARAMETERS if mode_parameters is None else tuple(mode_parameters)
    modes = tuple(
        ModeSpec(
            name=str(p["name"]),
            amplitude=float(p["amplitude"]),
            growth=float(p["growth"]),
            frequency=float(p["frequency"]),
            phase=float(p["phase"]),
            shape=sine_shape(grid, float(p["wavenumber"])),
        )
        for p in params
    )
    return SyntheticField(grid=grid, modes=modes)


def generate_snapshots(
    field: SyntheticField,
    times=None,
    noise_level: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Evaluate ``field`` and optionally add Gaussian noise (SYNTHETIC data).

    ``noise_level`` is relative to the standard deviation of the noise-free
    field over the whole snapshot matrix.  The generator is
    ``numpy.random.default_rng(seed)``, so noisy data is fully reproducible.
    """
    if times is None:
        times = default_times()
    u = field.evaluate(times)
    level = validate_finite_scalar(noise_level, "generate_snapshots noise_level")
    if level < 0.0:
        raise ValueError(f"noise_level must be >= 0; got {level!r}")
    if level > 0.0:
        try:
            s = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"seed must be an integer; got {seed!r}") from exc
        rng = np.random.default_rng(s)
        sigma = level * float(np.std(u))
        u = u + sigma * rng.standard_normal(u.shape)
    return u
