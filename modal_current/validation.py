"""Shared input-validation helpers for modal_current.

These helpers implement the package-wide input contract:

* snapshot matrices must be finite 2-D arrays with snapshots stored as
  columns and must provide a minimum number of samples;
* scalar parameters such as the sampling interval ``dt`` must be finite and
  strictly positive;
* time vectors must be finite 1-D arrays.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "validate_finite_scalar",
    "validate_positive_scalar",
    "validate_snapshot_matrix",
    "validate_time_vector",
]


def validate_snapshot_matrix(data, name: str = "data", min_samples: int = 2) -> np.ndarray:
    """Return ``data`` as a 2-D float array of snapshots (columns) or raise.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, is empty, contains non-finite values, or has
        fewer than ``min_samples`` columns.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array with snapshots as columns; got ndim={arr.ndim}"
        )
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN or Inf)")
    if arr.shape[1] < min_samples:
        raise ValueError(
            f"{name} needs at least {min_samples} snapshot(s) (columns); got {arr.shape[1]}"
        )
    return arr


def validate_positive_scalar(value, name: str) -> float:
    """Return ``value`` as a finite float that is strictly positive, or raise."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar; got {value!r}") from exc
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive; got {v!r}")
    return v


def validate_finite_scalar(value, name: str) -> float:
    """Return ``value`` as a finite float, or raise."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar; got {value!r}") from exc
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite; got {v!r}")
    return v


def validate_time_vector(times, name: str = "times") -> np.ndarray:
    """Return ``times`` as a finite 1-D float array or raise."""
    arr = np.asarray(times, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array of time instants; got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one time instant")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN or Inf)")
    return arr
