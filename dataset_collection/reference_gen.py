"""
reference_gen.py
================
Smooth, band-limited [a_cmd, delta_cmd] reference sequences for closed-loop
data collection, sampled via a random sum-of-sines surrogate for a Gaussian
process. The paper draws position from a GP prior (their eq. 25) and back-solves
control via differential flatness — that path is unavailable for a nonholonomic
bicycle whose curvature is bounded, so we sample control DIRECTLY instead.

Two consequences follow:

  * Every reference is dynamically feasible BY CONSTRUCTION — the resulting
    (x, y, theta, v) trajectory is exactly what the bicycle model produces when
    driven with those controls, so the tracker never has to fight an infeasible
    curvature demand and every rollout yields clean training data.

  * The tracker's job flips from "reach this waypoint" to "follow this control
    profile with tracker-specific perturbations/optimization on top of it,"
    which is exactly what widens the action distribution the JEPA sees (MPC
    smooths and re-optimizes, MPPI adds sampling noise, the union covers both
    smooth and jittery command patterns — the paper's NMPC + MPPI diversity
    argument, adapted).

The surrogate is a truncated Fourier sum with random periods/phases/weights.
It matches a GP with a periodic kernel in its dominant-frequency content but
skips the O(n^3) posterior sampling — noticeable at 10k+ rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Command bounds. Copied from taxi_controller_mppi.py so the reference is always
# realizable regardless of the sampled dynamics (which may narrow the effective
# bounds further per-rollout; that's fine — the tracker will clip in real time).
A_MIN, A_MAX = -4.0, 1.5
DELTA_LIM = 0.5


@dataclass
class ReferenceConfig:
    """Configuration for one GP-style reference sequence.

    n_steps    : how many timesteps the reference spans (10s at 0.1s = 100).
    dt         : timestep [s]; must match the controller / Unity Fixed Timestep.
    n_harm     : number of sinusoidal components per channel. 2-4 gives smoothly
                 varying control with clear direction changes; higher hardens
                 the spectrum toward noise.
    a_scale    : amplitude of the acceleration channel [m/s^2]. Kept BELOW the
                 A_MIN/A_MAX magnitudes so most samples land in-bounds and only
                 the peaks clip — clipping is fine (it exposes the model to
                 saturation) but shouldn't be the norm.
    d_scale    : amplitude of the steering channel [rad].
    period_lo  : shortest period sampled [s]. Below ~2 s the resulting command
                 changes faster than the actuator can track (MAX_STEER_RATE
                 caps the effective delta), which just wastes samples on
                 pre-actuator noise.
    period_hi  : longest period [s]. Should be a good fraction of n_steps*dt so
                 the trajectory sees at least one full oscillation.
    a_bias     : constant added to a_cmd. Slight positive bias gives net forward
                 motion so rollouts explore forward speed, not just oscillate
                 around v=0.
    """
    n_steps: int = 100
    dt: float = 0.1
    n_harm: int = 3
    a_scale: float = 1.6
    d_scale: float = 0.30
    period_lo: float = 2.5
    period_hi: float = 12.0
    a_bias: float = 0.4


def _one_channel(rng: np.random.Generator, t: np.ndarray, cfg: ReferenceConfig,
                  scale: float, bias: float = 0.0) -> np.ndarray:
    """One 1-D band-limited random signal on the grid `t`.

    Sum of `cfg.n_harm` sines with periods drawn log-uniformly (favours low
    frequencies mildly, which matches how real driving commands are distributed),
    random phases, and softmax-normalised weights so the amplitude stays close
    to `scale` regardless of n_harm.
    """
    log_lo, log_hi = np.log(cfg.period_lo), np.log(cfg.period_hi)
    periods = np.exp(rng.uniform(log_lo, log_hi, cfg.n_harm))
    phases = rng.uniform(0.0, 2 * np.pi, cfg.n_harm)
    raw_w = rng.uniform(0.3, 1.0, cfg.n_harm)
    weights = raw_w / raw_w.sum()
    sig = np.zeros_like(t)
    for w, p, ph in zip(weights, periods, phases):
        sig += w * np.sin(2 * np.pi * t / p + ph)
    return bias + scale * sig


def gp_control_sequence(rng: np.random.Generator,
                          cfg: Optional[ReferenceConfig] = None
                          ) -> np.ndarray:
    """(n_steps, 2) reference control sequence [a_cmd, delta_cmd].

    Values are clipped to the actuator box so downstream trackers never receive
    something they'd have to reject; this makes the reference itself a legal
    control sequence, and the resulting rollout is guaranteed feasible.
    """
    cfg = cfg or ReferenceConfig()
    t = np.arange(cfg.n_steps) * cfg.dt
    a_cmd = _one_channel(rng, t, cfg, cfg.a_scale, bias=cfg.a_bias)
    delta_cmd = _one_channel(rng, t, cfg, cfg.d_scale, bias=0.0)
    a_cmd = np.clip(a_cmd, A_MIN, A_MAX)
    delta_cmd = np.clip(delta_cmd, -DELTA_LIM, DELTA_LIM)
    return np.stack([a_cmd, delta_cmd], axis=1)


def batch_references(rng: np.random.Generator, n: int,
                      cfg: Optional[ReferenceConfig] = None) -> np.ndarray:
    """(n, n_steps, 2) — convenience wrapper for pre-generating a batch."""
    cfg = cfg or ReferenceConfig()
    out = np.empty((n, cfg.n_steps, 2), dtype=np.float32)
    for i in range(n):
        out[i] = gp_control_sequence(rng, cfg)
    return out


if __name__ == "__main__":
    # Sanity: bounds respected, and the resulting spectrum looks band-limited.
    rng = np.random.default_rng(0)
    refs = batch_references(rng, n=64)
    assert refs.shape == (64, 100, 2)
    assert refs[..., 0].min() >= A_MIN - 1e-6
    assert refs[..., 0].max() <= A_MAX + 1e-6
    assert refs[..., 1].min() >= -DELTA_LIM - 1e-6
    assert refs[..., 1].max() <= DELTA_LIM + 1e-6

    # Mean and std over the batch, per channel — should show meaningful variance
    # per timestep, not a degenerate constant.
    a = refs[..., 0]
    d = refs[..., 1]
    print(f"a_cmd:   mean={a.mean():+.3f}  std={a.std():.3f}  "
          f"range=[{a.min():+.2f}, {a.max():+.2f}]")
    print(f"delta:   mean={d.mean():+.3f}  std={d.std():.3f}  "
          f"range=[{d.min():+.2f}, {d.max():+.2f}]")

    # Quick per-rollout autocorrelation check: lag-1 correlation should be high
    # (smooth reference), lag-20 should be lower (band-limited, not constant).
    def _autocorr(x, lag):
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    for i in range(3):
        r1 = _autocorr(refs[i, :, 0], 1)
        r20 = _autocorr(refs[i, :, 0], 20)
        print(f"  rollout {i}: a_cmd autocorr lag1={r1:+.3f}  lag20={r20:+.3f}")