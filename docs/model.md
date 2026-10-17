# Model and methods

**Modal Current** decomposes a *synthetic* spatiotemporal field whose modal
content is known by construction, so decomposition code can be verified
against ground truth. Everything on this page is stated in **nondimensional
units**: `x` lives on the open interval (0, 1), `t` covers `[0, t_end]`, and
amplitudes, growth rates and frequencies carry no physical units. All data in
this project is **SYNTHETIC demonstration data** — it is not a measurement or
simulation of any physical flow, and no turbulence or other flow-physics
claim is made anywhere.

## 1. The synthetic field

The field is the sum of `K` separable standing modes,

```
u(x, t) = sum_{k=1..K}  a_k * exp(gamma_k * t) * cos(omega_k * t + phi_k) * shape_k(x)
```

with the documented default configuration (`K = 3`):

| mode     | amplitude `a_k` | growth `gamma_k` | frequency `omega_k` | phase `phi_k` | shape `shape_k(x)` |
|----------|-----------------|------------------|---------------------|---------------|--------------------|
| `mode-1` | 1.00            | -0.05 (decaying) | 2.0                 | 0.30 rad      | `sin(1*pi*x)`      |
| `mode-2` | 0.60            | +0.02 (growing)  | 5.0                 | 1.10 rad      | `sin(2*pi*x)`      |
| `mode-3` | 0.35            | -0.12 (decaying) | 9.5                 | 2.00 rad      | `sin(3*pi*x)`      |

Grid and sampling used throughout the verification:

* spatial grid: 512 **cell-centred** points `x_j = (j + 0.5)/512` on (0, 1).
  Cell centring makes integer-wavenumber sines exactly orthogonal under the
  discrete inner product (sums of roots of unity vanish), which keeps the POD
  subspace check clean;
* time grid: uniform `t_j = j * dt` with `dt = 0.025`, `t_end = 10`
  (401 snapshots);
* optional noise: additive Gaussian with standard deviation
  `noise_level * std(u_clean)` (`noise_level = 0.01`, i.e. 1%), drawn from
  `numpy.random.default_rng(seed)` with a fixed `seed = 0`, so every noisy
  result is reproducible.

Each real mode corresponds to one **conjugate pair** of continuous
eigenvalues `lambda_k = gamma_k +/- i omega_k`, i.e. `2K = 6` complex
exponentials in total — a bookkeeping fact that matters below.

## 2. POD — SVD of the centered data matrix

Snapshots are stored as columns of `D` in `R^{n x m}` (`n` grid points,
`m` time samples). POD proceeds as:

1. **center** — subtract the temporal mean `d_bar` (one value per grid
   point): `D_c = D - d_bar * 1^T`;
2. **decompose** — economy SVD `D_c = U S V^T`;
3. **interpret** — columns of `U` are the POD (spatial) modes, rows of
   `U^T D_c` are the temporal coefficients, and the energy fraction of mode
   `i` is `S_i^2 / sum_j S_j^2` over the *full* spectrum (so the fractions
   sum to 1 by construction, at roundoff).

For `m << n` the economy SVD of the tall `n x m` centered matrix is the
*method of snapshots*: it never forms the `n x n` covariance, and is
numerically equivalent to the eigendecomposition of the `m x m` Gram matrix
`D_c^T D_c = V S^2 V^T`.

A structural fact used by the tests: because each mode is separable
(`shape_k(x)` times one scalar temporal signal), the whole trajectory lives
in the `K`-dimensional span of the shapes, so the noise-free centered matrix
has **rank exactly K** — one nonzero singular value per mode (here 3, at a
machine-precision threshold), *not* `2K`. POD sees spatial directions; it
does not split oscillation quadratures.

## 3. DMD — exact variant, step by step

The implemented variant is **exact DMD** (Tu et al., *Dynamic mode
decomposition: Theory and applications*), i.e. the SVD-based construction
whose eigenvectors are reconstructed with the pseudoinverse of the truncated
basis rather than read off as `U w`. Applied to a snapshot matrix (snapshots
as columns):

1. **(embedding, `delays = d`)** optionally replace the snapshots by the
   Hankel / delay-embedded matrix with columns
   `z_j = (u_j, u_{j+1}, ..., u_{j+d-1})` — see Section 3.1 for why this is
   needed here;
2. **split** `X = H[:, :-1]`, `Y = H[:, 1:]`;
3. **truncate** `X ~= U_r S_r V_r^H` (rank `r` = machine-precision numerical
   rank by default, `s_i > s_1 * eps * max(shape)`);
4. **projected operator** `A~ = U_r^H Y V_r S_r^{-1}` — an `r x r` matrix
   built with the pseudoinverse of the truncated basis;
5. **eigendecompose** `A~ w_k = mu_k w_k` (discrete eigenvalues);
6. **exact-DMD modes** `phi_k = (1/mu_k) * Y V_r S_r^{-1} w_k`, with the
   projected fallback `U_r w_k` if `mu_k = 0`;
7. **amplitudes** `b = Phi^+ z_0` — least-squares fit of the (embedded)
   first snapshot;
8. **continuous spectrum** `gamma_k + i*omega_k = log(mu_k) / dt`, i.e.
   `omega = Im(log(mu))/dt` and `gamma = Re(log(mu))/dt`.

Reconstruction / prediction in physical space uses the first block of each
embedded mode (the physical part of `phi_k`) with the amplitudes:

```
u(t) ~= Re( sum_k  b_k * phi_k * exp((gamma_k + i*omega_k) * (t - t0)) )
```

### 3.1 Why a delay embedding is required for this field

A sum of `K` separable *standing* modes has a snapshot matrix of rank `K`
only. The exact shift relation `u_{j+1} = A u_j` is then **not consistent**
on the raw snapshots: a standing oscillation `e^{gamma t} cos(omega t + phi)
* shape(x)` needs two state dimensions per mode (cosine and sine
quadrature), but only one spatial direction exists per mode. Concretely, one
can construct a linear relation among the columns of `X` that is *not*
preserved by the shift to `Y` (take a real coefficient vector `alpha` with
`sum_j alpha_j mu_k^j = i * conj(w_k)` for one mode and zero for the
others; then `sum_j alpha_j u_j = 0` while `sum_j alpha_j u_{j+1} != 0`).
Plain exact DMD (`delays = 1`) therefore fits an inconsistent least-squares
problem and returns biased eigenvalues — on this project's field it returns
only real eigenvalues (recovered frequencies 0, 100% error) and a training
reconstruction error of 119%.

Stacking `d >= 2` time-delayed copies lifts the trajectory into a
`K*d`-dimensional space in which the shift relation holds exactly: the
embedded eigenvectors `v_k (x) (shape_k, mu_k shape_k, ..., mu_k^{d-1}
shape_k)` are linearly independent for distinct `mu_k` (Vandermonde
argument), so the delay-embedded operator has exactly the `2K` analytic
eigenvalues `mu_k = exp((gamma_k +/- i omega_k) dt)`. This is the standard
time-delay / Hankel-DMD construction; the verification uses `d = 2`, the
minimal depth for standing modes.

### 3.2 POD rank 3 vs DMD rank 6

For the same noise-free data, POD returns 3 modes (the span of the shapes)
and DMD returns 6 eigenvalues (3 conjugate pairs). Both are correct: POD
answers "which spatial directions carry the data", DMD answers "which
exponential spatiotemporal components generate the data".

## 4. Notation summary

| symbol | meaning |
|--------|---------|
| `D`    | snapshot matrix, `n x m`, snapshots as columns |
| `d_bar`| temporal mean field |
| `U S V^T` | (centered, truncated) SVD |
| `mu_k` | discrete DMD eigenvalue |
| `gamma_k`, `omega_k` | continuous growth rate and frequency, `log(mu_k)/dt` |
| `phi_k` | DMD spatial mode (physical block of the embedded eigenvector) |
| `b_k`  | DMD amplitude |
| `d` (`delays`) | Hankel embedding depth |

## References (by title)

* *Dynamic mode decomposition of numerical and experimental data* (Schmid)
* *Dynamic mode decomposition: Theory and applications* (Tu, Rowley,
  Luchtenburg, Brunton, Kutz)
* *Model reduction for flow analysis* (Rowley & Dawson)
