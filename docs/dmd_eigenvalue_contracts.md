# DMD eigenvalue and numerical-rank contracts

This repair addresses three reproducible core failures independently of
the separate results/presentation study.

## Negative real eigenvalues

A sequence u_j=(-0.8)^j has discrete eigenvalue -0.8. Its principal
complex logarithm is log(0.8)+iπ, so continuous growth is log(0.8)/dt
and the sampled angular frequency has magnitude π/dt (Nyquist). Taking
the real-valued NumPy logarithm returned NaN. The complex logarithm now
keeps the fitted mode finite, and both discrete and sampled continuous
reconstruction reproduce the alternating decay.

The principal branch lies in the usual sampled frequency band; discrete
data cannot determine which aliased continuous frequency generated it.
The sign at the negative real branch cut depends on signed-zero convention.

## Zero eigenvalues

The impulse [1,0,0,...] has a zero discrete eigenvalue. There is no finite
continuous logarithm. `reconstruct_training()` now evaluates integer
powers directly: multiplier 1 at step zero, and 0 on later steps. It
preserves the impulse instead of evaluating exp(log(0)*0) as NaN.

`continuous_eigenvalues`, growth/frequency properties and continuous
`reconstruct(times)` explicitly raise ValueError for a zero mode. This
includes backwards and off-grid requests; no continuous extension is
invented. Tiny fitted nonzero eigenvalues retain their ordinary complex
logarithm; no zero-eigenvalue clipping is introduced. Mode construction
uses projected eigenvectors when abs(mu)<=eps*rank*norm(A_tilde,2), avoiding
division by a numerically singular eigenvalue. This also preserves an
impulse mixed with a decaying mode when an exact zero is fitted as a tiny
roundoff value. Fitted eigenvalues themselves are retained.

## Rank-deficient data

Explicit truncation rank must not exceed `numerical_rank(singular_values,
X.shape)`, the same machine-precision cutoff used by automatic rank
selection. Oversized rank is rejected before inversion, with the requested
and available ranks in the error message. Noisy data can still use any
truncation at or below its numerical rank.

## Reproduction

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Regressions use a scalar alternating decay, a scalar impulse, a mixed
impulse/decay sequence and three exactly rank-deficient matrices. These
are hand-solvable numerical anchors, not experimental flow validation.
The existing synthetic POD/DMD suite remains in place.

[Existing calendar metadata disclosure](CALENDAR_HISTORY.md).
