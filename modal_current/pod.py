"""Proper Orthogonal Decomposition (POD) via the SVD of the centered data matrix.

Given snapshots ``D`` in ``R^{n x m}`` (``n`` grid points, ``m`` time
samples, snapshots as columns):

1. subtract the temporal mean ``d_bar`` (one mean value per grid point)::
       D_c = D - d_bar * 1^T
2. compute the economy SVD of the centered matrix::
       D_c = U S V^T
3. POD modes are the columns of ``U``; temporal coefficients are
   ``U^T D_c``; the energy fraction of mode ``i`` is
   ``S_i^2 / sum_j S_j^2`` computed over the full spectrum.

For ``m << n`` the economy SVD of the tall ``n x m`` centered matrix is
numerically the *method of snapshots* (the ``m x m`` Gram matrix
``D_c^T D_c = V S^2 V^T`` is never formed explicitly, but the decomposition
returned is the same).

For the synthetic standing-mode field with ``K`` separable modes the
centered snapshot matrix has rank exactly ``K`` (each mode occupies a single
spatial direction), so ``K`` nonzero singular values are expected in the
noise-free case at the machine-precision threshold — while DMD resolves
``2 K`` eigenvalues (one conjugate pair per mode); see :mod:`modal_current.dmd`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation import validate_snapshot_matrix

__all__ = ["PODResult", "compute_pod", "numerical_rank"]


def numerical_rank(singular_values, matrix_shape: tuple[int, int]) -> int:
    """Number of singular values above the machine-precision threshold.

    The threshold is the standard rank tolerance
    ``s_max * eps * max(n_rows, n_cols)`` used by numerical-rank routines.
    """
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0 or s[0] <= 0.0:
        return 0
    tol = s[0] * np.finfo(float).eps * max(matrix_shape)
    return int(np.count_nonzero(s > tol))


@dataclass
class PODResult:
    """POD of a snapshot matrix (centered SVD)."""

    mean: np.ndarray                    # (n,) temporal-mean field
    spatial_modes: np.ndarray           # (n, r) retained POD modes (columns)
    singular_values: np.ndarray         # (p,) FULL spectrum, p = min(n, m)
    energy_fractions: np.ndarray        # (p,) full-spectrum fractions, sum = 1
    temporal_coefficients: np.ndarray   # (r, m) coefficients of the centered field

    @property
    def rank(self) -> int:
        """Number of retained POD modes."""
        return self.spatial_modes.shape[1]

    def energy_capture(self, rank: int | None = None) -> float:
        """Energy fraction captured by the first ``rank`` POD modes."""
        r = self.rank if rank is None else int(rank)
        if r < 0 or r > self.energy_fractions.size:
            raise ValueError(
                f"rank must be in [0, {self.energy_fractions.size}]; got {rank!r}"
            )
        return float(self.energy_fractions[:r].sum())

    def reconstruct(self, rank: int | None = None) -> np.ndarray:
        """Rebuild the snapshots from the first ``rank`` modes (mean added back)."""
        r = self.rank if rank is None else int(rank)
        if r < 0 or r > self.rank:
            raise ValueError(f"rank must be in [0, {self.rank}]; got {rank!r}")
        return self.mean[:, np.newaxis] + self.spatial_modes[:, :r] @ self.temporal_coefficients[:r, :]


def compute_pod(data, rank: int | None = None) -> PODResult:
    """POD of ``data`` (snapshots as columns), retaining ``rank`` modes.

    Parameters
    ----------
    data : (n, m) array_like
        Snapshot matrix, one column per time sample.
    rank : int, optional
        Number of modes to retain.  ``None`` (default) keeps every singular
        value above the machine-precision threshold
        ``s_max * eps * max(n, m)``.

    Raises
    ------
    ValueError
        If ``data`` is not a finite 2-D matrix with at least one snapshot,
        or ``rank`` is not a positive integer within range.
    """
    data = validate_snapshot_matrix(data, name="data", min_samples=1)
    mean = data.mean(axis=1)
    centered = data - mean[:, np.newaxis]
    u_mat, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
    if rank is None:
        r = numerical_rank(s_vals, centered.shape)
    else:
        if not isinstance(rank, (int, np.integer)) or int(rank) < 1:
            raise ValueError(f"rank must be a positive integer; got {rank!r}")
        if int(rank) > s_vals.size:
            raise ValueError(f"rank must be <= {s_vals.size}; got {int(rank)}")
        r = int(rank)
    total_energy = float(np.sum(s_vals**2))
    if total_energy > 0.0:
        energy = s_vals**2 / total_energy
    else:
        energy = np.zeros_like(s_vals)
    temporal = u_mat[:, :r].T @ centered
    return PODResult(
        mean=mean,
        spatial_modes=u_mat[:, :r],
        singular_values=s_vals,
        energy_fractions=energy,
        temporal_coefficients=temporal,
    )
