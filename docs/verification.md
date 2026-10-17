# Verification: what each test establishes — and what it does not

All verification uses the documented **SYNTHETIC** three-mode field
(`docs/model.md`); there is no physical data anywhere in this project.
Measured numbers below come from the actual runs recorded in
[`results/run_summary.json`](results/run_summary.json) (verification
configuration: 512 grid points, 401 snapshots, dt = 0.025) and from the
pytest suite (reduced configuration: 128 points, 121 snapshots, dt = 0.05;
75 tests, all passing).

## Test inventory

### Synthetic generator (`tests/test_synthetic.py`)

| test | establishes |
|------|-------------|
| default parameters equal the documented constants | the "known ground truth" cannot drift from the docs |
| grid/times construction | cell-centred (0,1) grid; exactly uniform `j*dt` time base |
| `evaluate` matches the closed-form definition | the generator implements the stated formula |
| modal contributions sum to the total | decomposition of the data into per-mode fields is consistent |
| noise-free generation equals the analytic field | no hidden contamination of clean data |
| noise reproducibility (same seed equal, different seed different) | noisy experiments are repeatable |
| noise std = `noise_level` x field std (measured 0.00998 for 1%) | the documented noise model is what is applied |
| validation tests | bad modes / grids / noise levels / seeds / time vectors raise `ValueError` |

### POD (`tests/test_pod.py`)

| test | establishes | measured |
|------|-------------|----------|
| energy fractions sum to 1 | normalization is exact | `sum = 1.0` (0.0 difference) |
| noise-free rank equals number of modes | exactly 3 nonzero singular values at machine threshold; 4th SV at 7.9e-16 of the 1st | rank = 3 |
| POD basis reproduces the analytic subspace | projection of each `sin(k pi x)` onto the basis captures > 99% of its energy | capture = 1.0 (±1e-15) |
| POD modes orthonormal | `U^T U = I` | to 1e-15 |
| full-rank reconstruction roundoff | rank-3 POD rebuilds the training data | rel. error 7.4e-16 |
| truncated error = discarded energy | `||D - D_r||^2 / ||D_c||^2 = 1 - sum(f_i)` | identity to 1e-10 |
| temporal coefficients uncorrelated | `A A^T = diag(S_r^2)` (POD decorrelation property) | to 1e-10 |
| constant data edge case | zero rank, energy fractions 0, mean-only reconstruction exact | — |
| validation | non-finite / wrong-shape data, invalid ranks raise | — |

### DMD (`tests/test_dmd.py`)

| test | establishes | measured |
|------|-------------|----------|
| Hankel embed structure | stacked matrix equals shifted copies of the data | exact |
| rank = 6 = 2K | delay-embedded data resolves both quadratures of each mode | rank = 6 |
| **known frequencies/growth recovered** | matched eigenvalues reproduce `omega_k`, `gamma_k` (assert < 1e-10; spec target 1e-6) | max omega rel. err 2.0e-14; max abs growth err 4.7e-14 |
| conjugate pairs | real data must yield conjugate spectra | pairs exact to < 1e-12 |
| training reconstruction | full-rank exact DMD rebuilds the snapshots (assert < 1e-9) | 4.5e-13 |
| held-out extrapolation | model trained on [0, 6] predicts 40 steps (2 time units) beyond the window (assert < 1e-8) | 5.5e-13 (train 0–8 case: 81 held-out snapshots over 2.02 units, 5.5e-13) |
| noise robustness band | 1% Gaussian noise (seed 0), rank 6: frequencies within a 1% band, growth within 0.03 absolute (asserted) | max omega rel. err 6.6e-4 (0.07%); max abs growth err 1.0e-2 |
| plain DMD (`delays=1`) is biased | documents honestly why the embedding is used: recovered frequencies all 0 (100% error), reconstruction error 119% | see below |
| validation | bad `dt` (0, negative, NaN), single snapshot, invalid rank/delays/t0 raise | — |

## Key measured results (verification configuration)

| mode | omega analytic | omega recovered (noise-free) | rel. error | gamma analytic | gamma recovered | abs. error |
|------|----------------|------------------------------|------------|----------------|-----------------|------------|
| mode-1 | 2.0 | 1.99999999999996 | 1.9e-14 | -0.05 | -0.050000000000031 | 3.1e-14 |
| mode-2 | 5.0 | 4.99999999999990 | 2.0e-14 | +0.02 | +0.019999999999953 | 4.7e-14 |
| mode-3 | 9.5 | 9.49999999999992 | 8.2e-15 | -0.12 | -0.119999999999977 | 2.3e-14 |

Additional aggregates:

* POD energy fractions: 0.5616 / 0.3965 / 0.0420 (sum exactly 1).
* DMD training-window reconstruction relative error: 4.5e-13.
* Extrapolation (train on first 320 of 401 snapshots, predict the remaining
  81 over 2.02 time units): relative error 5.5e-13.
* Noise (1% of field std, seed 0, rank 6, delays 2): omega relative errors
  6.6e-4 / 2.2e-5 / 2.1e-4; growth absolute errors 1.0e-2 / 1.9e-3 / 7.8e-3.
* Plain exact DMD without embedding (`delays=1`): max omega relative error
  1.0 (all recovered frequencies zero) and training reconstruction relative
  error 1.197 — the documented motivation for the Hankel embedding
  (`docs/model.md`, Section 3.1).
* Total verification-script runtime: ~22 s (single workstation run).

## Limits of these claims

1. **SYNTHETIC data only.** Every number above describes recovery of a
   field constructed from the documented formula. Nothing here validates
   POD/DMD against experimental or simulation data, and no turbulence,
   coherent-structure or other flow-physics claim is made.
2. **DMD is a linear fit.** Exact DMD assumes `u_{j+1} = A u_j` (after
   embedding). The synthetic field satisfies this exactly by construction,
   which is why recovery reaches machine precision. Real data rarely does;
   the same code applied to nonlinear data inherits DMD's well-known
   linearity bias.
3. **Delay embedding changes the operator.** With `delays=2` the fitted
   operator acts on delay coordinates, not on the physical field directly.
   For this project's separable standing modes that is necessary (Section
   3.1 of `model.md`); for other data (e.g. already-full-rank traveling
   waves) `delays=1` may be appropriate.
4. **Noise sensitivity.** Growth rates are markedly more noise-sensitive
   than frequencies: at the documented 1% noise the frequencies stay within
   ~0.07% but growth-rate errors reach |Δgamma| ≈ 1.0e-2 (comparable to the
   smallest rate 0.02). The 1%-band / 0.03-absolute assertions are honest
   bounds for this fixed seed and configuration, not universal guarantees.
5. **Rank selection matters under noise.** The noise cases pass `rank=6`
   (the known `2K`) explicitly. Automatic numerical-rank selection is only
   meaningful for noise-free roundoff-limited data.
6. **Centering.** POD uses the temporally centered matrix (standard). DMD
   is applied to the *uncentered* snapshots: subtracting a temporal mean
   makes the shift relation affine (`y_{j+1} = A y_j + w`) and would
   introduce a spurious `mu = 1` component for this data.
