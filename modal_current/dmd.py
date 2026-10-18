"""Exact Dynamic Mode Decomposition with optional delay (Hankel) embedding.

Variant implemented
-------------------
This module implements **exact DMD** — the eigenvector variant of Tu et al.,
"Dynamic mode decomposition: Theory and applications" — built on the
truncated SVD of the snapshot matrix and the pseudoinverse of the truncated
basis.  For snapshots ``D`` (snapshots as columns) and an embedding depth
``d = delays``:

1. (only when ``delays = d > 1``) build the Hankel / delay-embedded matrix
   with columns ``z_j = (u_j, u_{j+1}, ..., u_{j+d-1})``;
2. split the (possibly embedded) matrix into ``X = H[:, :-1]`` and
   ``Y = H[:, 1:]``;
3. compute the truncated SVD ``X ~= U_r S_r V_r^H`` (rank ``r`` taken from
   the machine-precision threshold unless given explicitly);
4. form the projected operator using the pseudoinverse of the truncated
   basis, ``A~ = U_r^H Y V_r S_r^{-1}`` (an ``r x r`` matrix);
5. eigendecompose ``A~ w_k = mu_k w_k`` (discrete eigenvalues);
6. recover the *exact*-DMD modes ``phi_k = (1/mu_k) Y V_r S_r^{-1} w_k``
   (with the projected fallback ``U_r w_k`` for zero/numerically singular
   eigenvalues relative to the projected operator's scale);
7. compute amplitudes ``b = Phi^+ z_0`` by least squares against the
   embedded first snapshot;
8. map to continuous eigenvalues ``gamma_k + i omega_k = log(mu_k) / dt``.

Reconstruction and prediction in physical space use the first block of each
embedded mode together with the amplitudes::

    u(t) ~= Re( sum_k  b_k phi_k exp((gamma_k + i omega_k) (t - t0)) )

Why a delay embedding is used for this project's field
------------------------------------------------------
A sum of ``K`` separable standing modes ``shape_k(x) f_k(t)`` has a snapshot
matrix of rank ``K`` only: the entire trajectory lives in the
``K``-dimensional span of the shapes.  On the raw snapshots the exact shift
relation ``u_{j+1} = A u_j`` is then not consistent (a standing oscillation
needs two state dimensions per mode), so plain exact DMD with ``delays=1``
returns biased eigenvalues for this kind of data.  Stacking ``d >= 2``
time-delayed copies lifts the trajectory into a space where the linear shift
relation holds exactly; each analytic ``mu_k = exp((gamma_k + i omega_k) dt)``
is then recovered to machine precision.  This is the standard time-delay /
Hankel-DMD construction; see ``docs/model.md`` for the full derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pod import numerical_rank
from .validation import (
    validate_finite_scalar,
    validate_positive_scalar,
    validate_snapshot_matrix,
    validate_time_vector,
)

__all__ = ["DMDResult", "compute_dmd", "hankel_embed", "match_eigenvalues"]


def hankel_embed(data: np.ndarray, delays: int) -> np.ndarray:
    """Delay-embed ``data`` (n x m): stack ``delays`` shifted copies.

    Returns an array of shape ``(delays * n, m - delays + 1)`` whose column
    ``j`` is the delay vector ``(u_j, u_{j+1}, ..., u_{j + delays - 1})``.
    """
    if not isinstance(delays, (int, np.integer)) or int(delays) < 1:
        raise ValueError(f"delays must be an integer >= 1; got {delays!r}")
    delays = int(delays)
    n_t = data.shape[1]
    if delays > n_t - 1:
        raise ValueError(
            f"delays ({delays}) must leave at least 2 embedded snapshots; "
            f"data has {n_t} snapshots"
        )
    columns = n_t - delays + 1
    return np.vstack([data[:, j : j + columns] for j in range(delays)])


@dataclass
class DMDResult:
    """Exact-DMD fit of a snapshot sequence (see module docstring)."""

    eigenvalues: np.ndarray      # (r,) discrete eigenvalues mu_k
    modes: np.ndarray            # (n, r) physical-space modes phi_k
    amplitudes: np.ndarray       # (r,) modal amplitudes b_k
    dt: float
    t0: float
    n_snapshots: int
    n_delays: int
    rank: int
    singular_values: np.ndarray  # (p,) full spectrum of the embedded X

    @property
    def continuous_eigenvalues(self) -> np.ndarray:
        """Principal complex log(mu)/dt; zero eigenvalues raise ValueError.

        Negative real eigenvalues have Nyquist angular frequency pi/dt,
        not NaN. No finite continuous-time generator exists for mu=0.
        """
        if np.any(self.eigenvalues == 0):
            raise ValueError("zero eigenvalues have no finite continuous-time logarithm; use reconstruct_training()")
        return np.log(self.eigenvalues.astype(complex)) / self.dt

    @property
    def growth_rates(self) -> np.ndarray:
        """``gamma_k = Re(log(mu_k)) / dt``."""
        return self.continuous_eigenvalues.real

    @property
    def frequencies(self) -> np.ndarray:
        """``omega_k = Im(log(mu_k)) / dt``."""
        return self.continuous_eigenvalues.imag

    def reconstruct(self, times) -> np.ndarray:
        """Principal-log continuous reconstruction at finite real times.

        Fits containing a zero eigenvalue require discrete reconstruction
        via reconstruct_training(); their continuous/backward dynamics
        are undefined and raise explicitly instead of producing NaNs.
        """
        t = validate_time_vector(times, name="times")
        expo = np.exp(np.outer(self.continuous_eigenvalues, t - self.t0))
        weighted = self.modes * self.amplitudes[np.newaxis, :]
        return (weighted @ expo).real

    def reconstruct_training(self) -> np.ndarray:
        """Discrete training reconstruction using mu**j, including zero modes.

        At j=0 every mode has multiplier 1 (including 0**0); a zero mode
        vanishes at j>=1. No continuous logarithm is evaluated.
        """
        powers = self.eigenvalues.astype(complex)[:, np.newaxis] ** np.arange(self.n_snapshots)
        weighted = self.modes * self.amplitudes[np.newaxis, :]
        return (weighted @ powers).real


def compute_dmd(
    data,
    dt: float,
    rank: int | None = None,
    delays: int = 1,
    t0: float = 0.0,
) -> DMDResult:
    """Exact DMD of ``data`` (snapshots as columns); see module docstring.

    Parameters
    ----------
    data : (n, m) array_like
        Snapshot matrix, one column per uniformly sampled time instant.
    dt : float
        Sampling interval (strictly positive, same time units as ``t0``).
    rank : int, optional
        Truncation rank of the SVD of (the embedded) ``X``. ``None``
        uses the machine-precision numerical rank. Explicit ranks above
        that threshold raise before division by singular values.
    delays : int, default 1
        Number of stacked time-delayed copies (Hankel embedding depth).
        ``delays=1`` is plain exact DMD on the raw snapshots; ``delays >= 2``
        is required to recover the eigenvalues of separable standing-mode
        data such as this project's synthetic field.
    t0 : float
        Time instant of the first snapshot.

    Raises
    ------
    ValueError
        If ``data`` is not a finite 2-D matrix with at least 2 snapshots,
        ``dt`` is not strictly positive, or ``rank``/``delays`` are invalid.
    """
    data = validate_snapshot_matrix(data, name="data", min_samples=2)
    dt = validate_positive_scalar(dt, "dt")
    t0 = validate_finite_scalar(t0, "t0")
    if not isinstance(delays, (int, np.integer)) or int(delays) < 1:
        raise ValueError(f"delays must be an integer >= 1; got {delays!r}")
    delays = int(delays)
    n_x = data.shape[0]
    embedded = hankel_embed(data, delays) if delays > 1 else data
    x_mat = embedded[:, :-1]
    y_mat = embedded[:, 1:]
    u_mat, s_vals, vh = np.linalg.svd(x_mat, full_matrices=False)
    rank_available = numerical_rank(s_vals, x_mat.shape)
    if rank is None:
        r = rank_available
        if r == 0:
            raise ValueError("data has zero numerical rank; cannot compute DMD")
    else:
        if not isinstance(rank, (int, np.integer)) or int(rank) < 1:
            raise ValueError(f"rank must be a positive integer; got {rank!r}")
        if int(rank) > s_vals.size:
            raise ValueError(f"rank must be <= {s_vals.size}; got {int(rank)}")
        r = int(rank)
        if r > rank_available:
            raise ValueError(f"requested rank {r} exceeds numerical rank {rank_available}; reduce rank")
    u_r = u_mat[:, :r]
    s_r = s_vals[:r]
    v_r = vh[:r].conj().T
    a_tilde = (u_r.conj().T @ y_mat @ v_r) / s_r[np.newaxis, :]
    eigenvalues, w = np.linalg.eig(a_tilde)
    pre = ((y_mat @ v_r) / s_r[np.newaxis, :]) @ w
    # Dividing by an eigenvalue at the operator's roundoff scale can destroy
    # an otherwise valid impulse mode. Use its projected eigenvector there;
    # retain the fitted eigenvalue itself (no eigenvalue clipping).
    zero_mu = np.abs(eigenvalues) <= (
        np.finfo(float).eps * max(a_tilde.shape) * np.linalg.norm(a_tilde, ord=2)
    )
    modes_emb = np.empty_like(pre)
    modes_emb[:, ~zero_mu] = pre[:, ~zero_mu] / eigenvalues[np.newaxis, ~zero_mu]
    if np.any(zero_mu):
        modes_emb[:, zero_mu] = u_r @ w[:, zero_mu]
    amplitudes = np.linalg.lstsq(modes_emb, embedded[:, 0], rcond=None)[0]
    modes_phys = modes_emb[:n_x] if delays > 1 else modes_emb
    return DMDResult(
        eigenvalues=eigenvalues,
        modes=modes_phys,
        amplitudes=amplitudes,
        dt=dt,
        t0=t0,
        n_snapshots=data.shape[1],
        n_delays=delays,
        rank=r,
        singular_values=s_vals,
    )


def match_eigenvalues(
    result: DMDResult,
    growth_rates,
    frequencies,
    positive_only: bool = True,
) -> np.ndarray:
    """Index of the recovered eigenvalue closest to each analytic mode.

    Parameters
    ----------
    result : DMDResult
        Output of :func:`compute_dmd`.
    growth_rates, frequencies : (K,) array_like
        Analytic ``gamma_k`` and ``omega_k`` of the synthetic modes.
    positive_only : bool, default True
        Restrict matching to the positive-frequency branch (one eigenvalue
        per conjugate pair).

    Returns
    -------
    (K,) ndarray of int
        Indices into ``result.eigenvalues`` of the best matches.
    """
    g = np.asarray(growth_rates, dtype=float)
    w = np.asarray(frequencies, dtype=float)
    if g.ndim != 1 or w.ndim != 1 or g.size != w.size:
        raise ValueError("growth_rates and frequencies must be 1-D arrays of equal length")
    candidates = np.arange(result.eigenvalues.size)
    if positive_only:
        candidates = candidates[result.frequencies > 0]
        if candidates.size == 0:
            raise ValueError("no positive-frequency eigenvalues available to match")
    indices = []
    for g_k, w_k in zip(g, w):
        dist = np.hypot(result.growth_rates[candidates] - g_k, result.frequencies[candidates] - w_k)
        indices.append(int(candidates[np.argmin(dist)]))
    return np.array(indices, dtype=int)
