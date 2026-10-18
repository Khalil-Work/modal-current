#!/usr/bin/env python
"""End-to-end verification run for Modal Current (SYNTHETIC demonstration data).

Generates the documented synthetic field, computes POD and exact DMD (with
delay embedding), measures recovery of the known frequencies and growth
rates, reconstruction and extrapolation errors, and noise robustness, then
writes into ``docs/results/``:

* ``pod_spectrum.png``                 singular-value spectrum + energy fractions
* ``dmd_eigenvalues.png``              eigenvalue plane (unit circle) + (gamma, omega) plane
* ``reconstruction_error.png``         reconstruction error vs rank + probe time series
* ``frequency_growth_recovery.csv``    recovery table (noise-free and noisy)
* ``modal_current_demo.gif``           animated SYNTHETIC field evolution
* ``run_summary.json``                 every measured number, machine-readable

Usage::

    python examples/run_verification.py
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import PIL
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from modal_current import (  # noqa: E402
    DEFAULT_MODE_PARAMETERS,
    compute_dmd,
    compute_pod,
    default_times,
    generate_snapshots,
    make_default_field,
    match_eigenvalues,
)

RESULTS_DIR = REPO / "docs" / "results"

# --- documented run configuration (nondimensional) -------------------------
N_POINTS = 512            # cell-centred grid points on (0, 1)
DT = 0.025                # uniform sampling interval
T_END = 10.0              # training window covers [0, 10]
NOISE_LEVEL = 0.01        # Gaussian noise, 1% of the noise-free field std
NOISE_SEED = 0            # fixed seed -> fully reproducible
DELAYS = 2                # Hankel embedding depth for exact DMD
TRAIN_FRACTION = 0.8      # extrapolation exercise trains on the first 80%
GIF_FRAMES = 60           # real animation frames

SYNTHETIC_NOTE = "SYNTHETIC demonstration data"

COLOR_TOTAL = "#0F4C81"
COLOR_MODES = ["#E4572E", "#17A398", "#7678ED"]
COLOR_REF = "#8A8A8A"


def relative_error(recovered: float, analytic: float) -> float:
    return float(abs(recovered - analytic) / abs(analytic))


def pod_spectrum_figure(pod, path: Path) -> None:
    indices = np.arange(1, 9)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

    ax_left.semilogy(indices, pod.singular_values[:8], "o-", color=COLOR_TOTAL, lw=1.5, ms=5)
    ax_left.axvline(pod.rank + 0.5, color=COLOR_REF, ls="--", lw=1.1)
    ax_left.annotate(
        f"numerical rank = {pod.rank}",
        xy=(pod.rank + 0.5, pod.singular_values[0] * 0.5),
        fontsize=9,
        color="#444444",
    )
    ax_left.set_xlabel("POD mode index")
    ax_left.set_ylabel("Singular value")
    ax_left.set_title("Singular values", fontsize=10)

    ax_right.bar(
        indices,
        pod.energy_fractions[:8],
        color=COLOR_TOTAL,
        alpha=0.85,
        label="per mode",
    )
    ax_right.plot(
        indices,
        np.cumsum(pod.energy_fractions[:8]),
        "o-",
        color=COLOR_MODES[0],
        ms=4,
        lw=1.5,
        label="cumulative",
    )
    ax_right.set_xlabel("POD mode index")
    ax_right.set_ylabel("Energy fraction")
    ax_right.set_ylim(0.0, 1.05)
    ax_right.legend(frameon=False, fontsize=8)
    ax_right.set_title("POD energy fractions (sum = 1)", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def dmd_eigenvalue_figure(dmd, idx, growth, freq, names, dt, path: Path) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.2))

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax_left.plot(np.cos(theta), np.sin(theta), ls="--", color=COLOR_REF, lw=1.1, label="unit circle")
    positive_branch = dmd.frequencies > 0
    ax_left.plot(
        dmd.eigenvalues[~positive_branch].real,
        dmd.eigenvalues[~positive_branch].imag,
        "o",
        color=COLOR_TOTAL,
        ms=4,
        alpha=0.6,
        label="recovered (conj. branch)",
    )
    for k in range(len(names)):
        mu_analytic = np.exp((growth[k] + 1j * freq[k]) * dt)
        ax_left.plot(
            dmd.eigenvalues[idx[k]].real,
            dmd.eigenvalues[idx[k]].imag,
            "o",
            color=COLOR_MODES[k],
            ms=8,
            mfc="none",
            mec=COLOR_MODES[k],
            mew=1.6,
        )
        ax_left.plot(
            mu_analytic.real,
            mu_analytic.imag,
            marker="*",
            ls="none",
            ms=13,
            color=COLOR_MODES[k],
            label=f"{names[k]} analytic",
        )
    ax_left.set_xlabel("Re(mu)")
    ax_left.set_ylabel("Im(mu)")
    ax_left.set_aspect("equal")
    ax_left.set_ylim(-1.25, 1.25)
    ax_left.set_xlim(-1.25, 1.25)
    ax_left.legend(frameon=False, fontsize=7, loc="upper right")
    ax_left.set_title("Discrete DMD eigenvalues", fontsize=10)

    for k in range(len(names)):
        rec_gamma = dmd.growth_rates[idx[k]]
        rec_omega = dmd.frequencies[idx[k]]
        ax_right.plot(
            [growth[k], rec_gamma],
            [freq[k], rec_omega],
            ls=":",
            color=COLOR_REF,
            lw=1.0,
        )
        ax_right.plot(
            growth[k], freq[k], marker="*", ls="none", ms=13, color=COLOR_MODES[k]
        )
        ax_right.plot(
            rec_gamma,
            rec_omega,
            "o",
            ms=8,
            mfc="none",
            mec=COLOR_MODES[k],
            mew=1.6,
        )
        ax_right.annotate(names[k], (growth[k], freq[k]), fontsize=8, xytext=(4, 4),
                          textcoords="offset points", color="#444444")
    ax_right.axvline(0.0, color=COLOR_REF, ls="--", lw=1.0)
    ax_right.annotate("neutral growth", xy=(0.0, 9.9), fontsize=7, color="#666666",
                      xytext=(4, 0), textcoords="offset points")
    ax_right.set_xlabel("Growth rate gamma (per unit time)")
    ax_right.set_ylabel("Frequency omega (per unit time)")
    ax_right.set_title("Recovered (o) vs analytic (*)", fontsize=10)

    fig.suptitle(f"DMD eigenstructure — {SYNTHETIC_NOTE}", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def reconstruction_figure(u_clean, times, grid, pod8, dmd, dmd_rank_errors,
                          sweep_ranks, path: Path) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))
    norm_u = np.linalg.norm(u_clean)

    pod_errors = [
        float(np.linalg.norm(pod8.reconstruct(rank=int(r)) - u_clean) / norm_u) for r in sweep_ranks
    ]
    floor = max(min(pod_errors + list(dmd_rank_errors)), 1e-17)
    ax_left.semilogy(sweep_ranks, pod_errors, "o-", color=COLOR_TOTAL, lw=1.5, ms=5, label="POD")
    ax_left.semilogy(
        sweep_ranks, dmd_rank_errors, "s-", color=COLOR_MODES[0], lw=1.5, ms=5,
        label="exact DMD (delays=2)",
    )
    ax_left.set_ylim(floor * 0.1, 10.0)
    ax_left.set_xlabel("retained rank r")
    ax_left.set_ylabel("relative reconstruction error")
    ax_left.legend(frameon=False, fontsize=8)
    ax_left.set_title("Training-window reconstruction error", fontsize=10)

    jx = int(np.argmin(np.abs(grid - 0.35)))
    dmd_recon = dmd.reconstruct(times)
    ax_right.plot(times, u_clean[jx], color="#111111", lw=2.0, label="analytic u(x*, t)")
    ax_right.plot(
        times, dmd_recon[jx], color=COLOR_MODES[0], lw=1.2, ls="--",
        label="exact DMD (rank 6)",
    )
    ax_right.plot(
        times, pod8.reconstruct(rank=2)[jx], color=COLOR_MODES[1], lw=1.2, ls=":",
        label="POD rank 2",
    )
    ax_right.set_xlabel("t (nondimensional)")
    ax_right.set_ylabel(f"u(x* = {grid[jx]:.3f}, t)")
    ax_right.legend(frameon=False, fontsize=8)
    ax_right.set_title("Reconstruction at a probe point", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    t_wall = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    field = make_default_field(n_points=N_POINTS)
    grid = field.grid
    times = default_times(dt=DT, t_end=T_END)
    u_clean = field.evaluate(times)
    growth = [float(p["growth"]) for p in DEFAULT_MODE_PARAMETERS]
    freq = [float(p["frequency"]) for p in DEFAULT_MODE_PARAMETERS]
    names = [str(p["name"]) for p in DEFAULT_MODE_PARAMETERS]
    amplitudes = [float(p["amplitude"]) for p in DEFAULT_MODE_PARAMETERS]

    summary: dict = {
        "project": "modal-current",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": (
            "SYNTHETIC demonstration data generated by modal_current.synthetic "
            "(documented formula, no physical measurement)"
        ),
        "configuration": {
            "n_gridpoints": N_POINTS,
            "dt": DT,
            "t_end": T_END,
            "n_snapshots": int(times.size),
            "noise_level": NOISE_LEVEL,
            "noise_seed": NOISE_SEED,
            "dmd_delays": DELAYS,
            "modes": [
                {
                    "name": names[k],
                    "amplitude": amplitudes[k],
                    "growth": growth[k],
                    "frequency": freq[k],
                    "phase": float(DEFAULT_MODE_PARAMETERS[k]["phase"]),
                    "wavenumber": int(DEFAULT_MODE_PARAMETERS[k]["wavenumber"]),
                }
                for k in range(len(names))
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
        },
    }

    # ---------------- POD ---------------------------------------------------
    t0 = time.perf_counter()
    pod = compute_pod(u_clean)
    pod8 = compute_pod(u_clean, rank=8)
    pod_rel = float(
        np.linalg.norm(pod.reconstruct() - u_clean) / np.linalg.norm(u_clean)
    )
    capture = [
        float(
            np.linalg.norm(pod.spatial_modes.T @ mode.shape) ** 2
            / np.linalg.norm(mode.shape) ** 2
        )
        for mode in field.modes
    ]
    summary["pod"] = {
        "method": "economy SVD of the temporally centered snapshot matrix (method of snapshots)",
        "numerical_rank": int(pod.rank),
        "expected_rank": field.n_modes,
        "singular_values_first8": [float(v) for v in pod.singular_values[:8]],
        "energy_fractions_first8": [float(v) for v in pod.energy_fractions[:8]],
        "energy_fraction_sum": float(pod.energy_fractions.sum()),
        "subspace_capture_per_analytic_mode": capture,
        "full_rank_reconstruction_rel_error": pod_rel,
    }
    pod_spectrum_figure(pod, RESULTS_DIR / "pod_spectrum.png")
    summary.setdefault("runtime_seconds", {})["pod_and_spectrum"] = time.perf_counter() - t0

    # ---------------- DMD ---------------------------------------------------
    t0 = time.perf_counter()
    dmd_plain = compute_dmd(u_clean, dt=DT, delays=1)
    plain_idx = match_eigenvalues(dmd_plain, growth, freq, positive_only=False)
    plain_omega_errors = [
        relative_error(dmd_plain.frequencies[i], freq[k]) for k, i in enumerate(plain_idx)
    ]
    plain_rel = float(
        np.linalg.norm(dmd_plain.reconstruct_training() - u_clean)
        / np.linalg.norm(u_clean)
    )

    dmd = compute_dmd(u_clean, dt=DT, delays=DELAYS)
    idx = match_eigenvalues(dmd, growth, freq)
    rec_freq = [float(dmd.frequencies[i]) for i in idx]
    rec_growth = [float(dmd.growth_rates[i]) for i in idx]
    freq_errors = [relative_error(rec_freq[k], freq[k]) for k in range(3)]
    growth_errors_abs = [abs(rec_growth[k] - growth[k]) for k in range(3)]
    growth_errors_rel = [
        growth_errors_abs[k] / abs(growth[k]) if growth[k] != 0.0 else float("nan")
        for k in range(3)
    ]
    dmd_rel = float(
        np.linalg.norm(dmd.reconstruct(times) - u_clean) / np.linalg.norm(u_clean)
    )

    # rank sweep for the error-vs-rank curve: honor the numerical rank
    # (the guard added with the core fixes rejects oversized requests).
    dmd_probe = compute_dmd(u_clean, dt=DT, delays=DELAYS)
    sweep_ranks = np.arange(1, min(8, dmd_probe.rank) + 1)
    dmd_rank_errors = [
        float(
            np.linalg.norm(
                compute_dmd(u_clean, dt=DT, rank=int(r), delays=DELAYS).reconstruct(times)
                - u_clean
            )
            / np.linalg.norm(u_clean)
        )
        for r in sweep_ranks
    ]

    # extrapolation exercise
    n_train = int(np.floor(TRAIN_FRACTION * times.size))
    dmd_train = compute_dmd(u_clean[:, :n_train], dt=DT, delays=DELAYS)
    t_ext = times[n_train:]
    pred = dmd_train.reconstruct(t_ext)
    truth = field.evaluate(t_ext)
    ext_rel = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))

    summary["dmd"] = {
        "variant": (
            "exact DMD (Tu et al. 2013): truncated SVD of X, projected operator "
            "A~ = U_r^H Y V_r S_r^{-1}, exact modes phi = (1/mu) Y V_r S_r^{-1} w, "
            "amplitudes by least squares on the first embedded snapshot; "
            f"applied to a delay-embedded (Hankel) matrix with delays={DELAYS}"
        ),
        "plain_dmd_delays1": {
            "max_omega_rel_error": max(plain_omega_errors),
            "training_reconstruction_rel_error": plain_rel,
            "note": (
                "plain exact DMD without delay embedding is inconsistent for "
                "separable standing modes; motivates delays=2 (docs/model.md)"
            ),
        },
        "noise_free": {
            "rank": int(dmd.rank),
            "recovered_frequencies": rec_freq,
            "recovered_growth_rates": rec_growth,
            "omega_rel_errors": freq_errors,
            "max_omega_rel_error": max(freq_errors),
            "growth_abs_errors": growth_errors_abs,
            "max_growth_abs_error": max(growth_errors_abs),
            "growth_rel_errors": growth_errors_rel,
            "training_reconstruction_rel_error": dmd_rel,
        },
        "extrapolation": {
            "train_snapshots": int(n_train),
            "train_window": [float(times[0]), float(times[n_train - 1])],
            "held_out_snapshots": int(times.size - n_train),
            "horizon_time_units": float(times[-1] - times[n_train - 1]),
            "rel_error": ext_rel,
        },
    }
    dmd_eigenvalue_figure(
        dmd, idx, growth, freq, names, DT, RESULTS_DIR / "dmd_eigenvalues.png"
    )
    reconstruction_figure(
        u_clean,
        times,
        grid,
        pod8,
        dmd,
        dmd_rank_errors,
        sweep_ranks,
        RESULTS_DIR / "reconstruction_error.png",
    )
    summary["runtime_seconds"]["dmd_and_figures"] = time.perf_counter() - t0

    # ---------------- noise robustness --------------------------------------
    t0 = time.perf_counter()
    u_noisy = generate_snapshots(field, times, noise_level=NOISE_LEVEL, seed=NOISE_SEED)
    dmd_noisy = compute_dmd(u_noisy, dt=DT, rank=2 * field.n_modes, delays=DELAYS)
    idx_n = match_eigenvalues(dmd_noisy, growth, freq)
    noisy_freq_errors = [
        relative_error(dmd_noisy.frequencies[i], freq[k]) for k, i in enumerate(idx_n)
    ]
    noisy_growth_abs = [
        float(abs(dmd_noisy.growth_rates[i] - growth[k])) for k, i in enumerate(idx_n)
    ]
    summary["dmd"]["noise_1pct"] = {
        "noise_level": NOISE_LEVEL,
        "seed": NOISE_SEED,
        "rank": int(2 * field.n_modes),
        "recovered_frequencies": [float(dmd_noisy.frequencies[i]) for i in idx_n],
        "recovered_growth_rates": [float(dmd_noisy.growth_rates[i]) for i in idx_n],
        "omega_rel_errors": noisy_freq_errors,
        "max_omega_rel_error": max(noisy_freq_errors),
        "growth_abs_errors": noisy_growth_abs,
        "max_growth_abs_error": max(noisy_growth_abs),
    }
    summary["runtime_seconds"]["noise_case"] = time.perf_counter() - t0

    # ---------------- recovery table (CSV) -----------------------------------
    csv_path = RESULTS_DIR / "frequency_growth_recovery.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "mode",
                "amplitude",
                "omega_analytic",
                "omega_recovered",
                "omega_rel_error",
                "growth_analytic",
                "growth_recovered",
                "growth_abs_error",
                "growth_rel_error",
            ]
        )
        for k in range(3):
            writer.writerow(
                [
                    "noise-free",
                    names[k],
                    f"{amplitudes[k]:.6g}",
                    f"{freq[k]:.12g}",
                    f"{rec_freq[k]:.12g}",
                    f"{freq_errors[k]:.6e}",
                    f"{growth[k]:.12g}",
                    f"{rec_growth[k]:.12g}",
                    f"{growth_errors_abs[k]:.6e}",
                    f"{growth_errors_rel[k]:.6e}",
                ]
            )
        for k in range(3):
            writer.writerow(
                [
                    f"noisy {NOISE_LEVEL:.0%} (seed {NOISE_SEED})",
                    names[k],
                    f"{amplitudes[k]:.6g}",
                    f"{freq[k]:.12g}",
                    f"{dmd_noisy.frequencies[idx_n[k]]:.12g}",
                    f"{noisy_freq_errors[k]:.6e}",
                    f"{growth[k]:.12g}",
                    f"{dmd_noisy.growth_rates[idx_n[k]]:.12g}",
                    f"{noisy_growth_abs[k]:.6e}",
                    f"{noisy_growth_abs[k] / abs(growth[k]):.6e}",
                ]
            )

    # ---------------- animated GIF -------------------------------------------
    t0 = time.perf_counter()
    contributions = field.modal_contributions(times)
    frame_indices = np.unique(
        np.round(np.linspace(0, times.size - 1, GIF_FRAMES)).astype(int)
    )
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    line_total, = ax.plot(
        grid, u_clean[:, 0], color=COLOR_TOTAL, lw=2.2, label="u(x, t) total"
    )
    mode_lines = []
    for k, mode in enumerate(field.modes):
        line, = ax.plot(
            grid,
            contributions[k, :, 0],
            color=COLOR_MODES[k],
            lw=1.0,
            alpha=0.9,
            label=f"{mode.name} contribution",
        )
        mode_lines.append(line)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-2.3, 2.3)
    ax.set_xlabel("x (nondimensional)")
    ax.set_ylabel("u (nondimensional)")
    title = ax.set_title("")
    ax.legend(loc="lower left", fontsize=7, ncol=4, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.13)

    def update(frame_number: int):
        j = int(frame_indices[frame_number])
        line_total.set_ydata(u_clean[:, j])
        for k, line in enumerate(mode_lines):
            line.set_ydata(contributions[k, :, j])
        # Two short lines keep the title inside the canvas; the shared
        # SYNTHETIC-data caption lives in the figure caption above.
        title.set_text(
            f"u(x, t) at t = {times[j]:.2f} — {SYNTHETIC_NOTE}"
        )
        return (line_total, *mode_lines, title)

    anim = FuncAnimation(
        fig, update, frames=frame_indices.size, interval=80, blit=False
    )
    gif_path = RESULTS_DIR / "modal_current_demo.gif"
    anim.save(gif_path, writer=PillowWriter(fps=12.5))
    plt.close(fig)
    with Image.open(gif_path) as image:
        gif_frames = int(image.n_frames)
        gif_size = (int(image.size[0]), int(image.size[1]))
    if gif_frames < 30:
        raise RuntimeError(f"GIF has only {gif_frames} frames (< 30)")
    summary["gif"] = {
        "path": str(gif_path),
        "frames": gif_frames,
        "width_px": gif_size[0],
        "height_px": gif_size[1],
        "fps": 12.5,
        "content": "evolution of the SYNTHETIC field u(x, t) with per-mode contributions",
    }
    summary["runtime_seconds"]["gif"] = time.perf_counter() - t0

    # ---------------- finish --------------------------------------------------
    summary["runtime_seconds"]["total"] = time.perf_counter() - t_wall
    with open(RESULTS_DIR / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Modal Current verification run (SYNTHETIC demonstration data)")
    print(f"  configuration: nx={N_POINTS}  dt={DT}  t_end={T_END}  n={times.size}")
    print(f"  POD: rank={pod.rank}  energy fractions="
          f"{[round(v, 6) for v in pod.energy_fractions[:3]]}")
    print(f"  POD: subspace capture per mode = {[round(c, 12) for c in capture]}")
    print(f"  POD: full-rank reconstruction rel err = {pod_rel:.3e}")
    print(f"  DMD plain (delays=1): max omega rel err = {max(plain_omega_errors):.3e} "
          f"(documents the need for delay embedding)")
    for k in range(3):
        print(
            f"  DMD {names[k]}: omega {rec_freq[k]:.10f} (rel err {freq_errors[k]:.2e})  "
            f"gamma {rec_growth[k]:+.10f} (abs err {growth_errors_abs[k]:.2e})"
        )
    print(f"  DMD training reconstruction rel err = {dmd_rel:.3e}")
    print(
        f"  extrapolation: {summary['dmd']['extrapolation']['held_out_snapshots']} held-out "
        f"snapshots over {summary['dmd']['extrapolation']['horizon_time_units']:.2f} time "
        f"units, rel err = {ext_rel:.3e}"
    )
    print(
        f"  noise {NOISE_LEVEL:.0%}: max omega rel err = "
        f"{summary['dmd']['noise_1pct']['max_omega_rel_error']:.3e}, "
        f"max |gamma err| = {summary['dmd']['noise_1pct']['max_growth_abs_error']:.3e}"
    )
    print(f"  GIF: {gif_frames} frames, {gif_size[0]}x{gif_size[1]} px -> {gif_path}")
    print(f"  total runtime {summary['runtime_seconds']['total']:.1f} s")
    print(f"  results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
