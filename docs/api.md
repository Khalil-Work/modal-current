# API reference

All functions take and return NumPy arrays with **snapshots as columns**
(shape `(n_gridpoints, n_times)`). All data is synthetic demonstration data.

## Synthetic field — `modal_current.synthetic`

### `DEFAULT_MODE_PARAMETERS`
Tuple of dicts with the documented three-mode configuration
(`name`, `amplitude`, `growth`, `frequency`, `phase`, `wavenumber`).

### `ModeSpec(name, amplitude, growth, frequency, phase, shape)`
One mode of the synthetic field.

* `name: str` — label; `amplitude: float > 0`; `growth: float` (any finite);
  `frequency: float > 0`; `phase: float` (radians); `shape: (n,) ndarray`.
* `.temporal(times) -> (m,) ndarray` — the scalar factor
  `a * exp(gamma t) * cos(omega t + phi)`.

### `SyntheticField(grid, modes)`
* `.evaluate(times) -> (n, m) ndarray` — snapshot matrix.
* `.modal_contributions(times) -> (K, n, m) ndarray` — per-mode fields.
* `.n_gridpoints`, `.n_modes` properties.
* Validates that every mode shape matches the grid length.

### `sine_shape(grid, wavenumber)`
`sin(wavenumber * pi * x)` on the grid; `wavenumber` must be positive.

### `default_grid(n_points=512)`
Cell-centred grid `(j + 0.5)/n_points` on (0, 1).

### `default_times(dt=0.025, t_end=10.0)`
Uniform `t_j = j * dt` covering `[0, t_end]` (`dt > 0` required).

### `make_default_field(n_points=512, mode_parameters=None)`
Builds the documented default field (or a custom one from a sequence of
parameter dicts with the same keys).

### `generate_snapshots(field, times=None, noise_level=0.0, seed=0)`
Evaluates the field and adds Gaussian noise with standard deviation
`noise_level * std(u_clean)` drawn from `default_rng(seed)`.
Reproducible for a fixed `seed`.

## POD — `modal_current.pod`

### `compute_pod(data, rank=None) -> PODResult`
SVD of the temporally centered snapshot matrix.

* `data: (n, m)` finite 2-D array, at least 1 snapshot;
* `rank=None` keeps every singular value above
  `s_max * eps * max(n, m)`; otherwise a positive integer `<= min(n, m)`.

### `PODResult`
* `.mean (n,)`, `.spatial_modes (n, r)`, `.temporal_coefficients (r, m)`
* `.singular_values (p,)`, `.energy_fractions (p,)` — full spectrum, the
  fractions sum to 1
* `.rank` — number of retained modes
* `.energy_capture(rank=None) -> float`
* `.reconstruct(rank=None) -> (n, m)` — mean plus retained modes

### `numerical_rank(singular_values, matrix_shape)`
Count of singular values above the machine-precision threshold.

## DMD — `modal_current.dmd`

### `compute_dmd(data, dt, rank=None, delays=1, t0=0.0) -> DMDResult`
Exact DMD (Tu et al.) with optional Hankel delay embedding; see
[`model.md`](model.md) for the precise construction.

* `data: (n, m)` finite 2-D array, at least 2 snapshots;
* `dt > 0` — uniform sampling interval;
* `rank=None` uses the numerical rank of (the embedded) `X`;
* `delays` — embedding depth; use `delays >= 2` for standing-mode data such
  as this project's field;
* `t0` — time of the first snapshot.

### `DMDResult`
* `.eigenvalues (r,)` — discrete `mu_k`
* `.growth_rates (r,)`, `.frequencies (r,)` — `Re/Im(log(mu)/dt)`
* `.continuous_eigenvalues (r,)` — `log(mu)/dt`
* `.modes (n, r)` — physical-space DMD modes
* `.amplitudes (r,)`, `.rank`, `.n_delays`, `.n_snapshots`,
  `.singular_values`
* `.reconstruct(times) -> (n, m)` — reconstruction / extrapolation
* `.reconstruct_training() -> (n, m)`

### `hankel_embed(data, delays)`
Delay-embedding matrix of shape `(delays * n, m - delays + 1)`.

### `match_eigenvalues(result, growth_rates, frequencies, positive_only=True)`
Index of the recovered eigenvalue closest to each analytic
`(gamma_k, omega_k)`; `positive_only=True` matches one eigenvalue per
conjugate pair.

## Validation — `modal_current.validation`

* `validate_snapshot_matrix(data, name, min_samples)` — finite 2-D check
* `validate_positive_scalar(value, name)` / `validate_finite_scalar(value, name)`
* `validate_time_vector(times, name)`

All raise `ValueError` with a descriptive message on bad input.

## Minimal example

```python
import numpy as np
from modal_current import compute_dmd, compute_pod, default_times, make_default_field, match_eigenvalues

field = make_default_field()                    # documented 3-mode SYNTHETIC field
times = default_times(dt=0.025, t_end=10.0)
u = field.evaluate(times)                       # (512, 401) snapshots

pod = compute_pod(u)                            # centered SVD
print("POD rank:", pod.rank)
print("energy fractions:", np.round(pod.energy_fractions[:3], 6))

dmd = compute_dmd(u, dt=0.025, delays=2)        # exact DMD, delay-embedded
idx = match_eigenvalues(dmd, [-0.05, 0.02, -0.12], [2.0, 5.0, 9.5])
print("recovered omega:", np.round(dmd.frequencies[idx], 9))
print("recovered gamma:", np.round(dmd.growth_rates[idx], 9))
```
