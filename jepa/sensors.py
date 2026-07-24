"""
sensors.py
==========
Synthesises a NOISY, REDUNDANT sensor suite from the clean recorded state, to
create the regime where latent modelling is supposed to pay off (implements
options 1 + 2 from the analysis: observation noise + redundant sensors).

Why synthesise in Python instead of re-collecting in Unity
----------------------------------------------------------
Every sensor below is a deterministic function of quantities the dataset already
records — speed v, steering delta, acceleration accel, and yaw rate omega =
dtheta/DT — plus noise. So computing them here is EXACTLY equivalent to Unity
computing and publishing them, without a rebuild or a re-collection. The only
thing this cannot capture is a sensor that depends on something outside the
recorded state; none of these do.

The design intent
------------------
The 10 channels below all encode the SAME 3-4 true degrees of freedom (four
wheel speeds are all ~v; the gyro is v/L*tan(delta); lateral accel is v*omega).
The true dynamics therefore live on a low-dimensional manifold, but the
OBSERVATION is high-dimensional, redundant, and noisy. That is the setting a
JEPA is built for: it can fuse the redundancy and average out the noise into a
compact latent, whereas a direct predictor must reconcile ten disagreeing noisy
channels at every step. A per-rollout bias adds a *structured* nuisance (a
constant offset the model can infer from context and ignore), on top of the
i.i.d. per-step noise it can only average away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import DT, IDX_V, IDX_DELTA, IDX_ACCEL


# Channel layout — 10 redundant sensors over the 3-4 true DOF.
SENSOR_NAMES: List[str] = [
    "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr",   # 4x noisy speed
    "accel_long",                                      # longitudinal accel
    "accel_lat",                                       # lateral accel ~ v*omega
    "gyro_yaw",                                        # yaw rate omega
    "steer_sensor",                                    # steering angle
    "motor_current",                                   # ~ accel (thrust proxy)
    "pitot_speed",                                     # a second, differently-noisy speed
]
N_OBS = len(SENSOR_NAMES)


@dataclass
class SensorConfig:
    """Noise model. `iid_scale` and `bias_scale` are FRACTIONS of each channel's
    own standard deviation, so one setting applies sensibly across channels with
    very different units (m/s vs rad/s vs m/s^2)."""
    iid_scale: float = 0.08     # per-step i.i.d. noise (averageable)
    bias_scale: float = 0.05    # per-rollout constant offset (inferable nuisance)
    seed: int = 0


def _clean_channels(v: np.ndarray, delta: np.ndarray, accel: np.ndarray,
                    omega: np.ndarray) -> np.ndarray:
    """The noise-free value of each sensor, (N, N_OBS)."""
    return np.stack([
        v, v, v, v,            # four wheel speeds
        accel,                  # longitudinal accel
        v * omega,              # lateral accel
        omega,                  # gyro
        delta,                  # steering sensor
        accel,                  # motor current proxy
        v,                      # pitot speed
    ], axis=1).astype(np.float32)


def synthesize(states: np.ndarray, dtheta: np.ndarray,
               rollout_of: np.ndarray, cfg: SensorConfig) -> np.ndarray:
    """Build the (N, N_OBS) noisy observation array for a whole split.

    states     : (N, 6) recorded clean states
    dtheta     : (N,)   per-transition heading change (yaw rate = dtheta/DT)
    rollout_of : (N,)   rollout index per transition (for per-rollout bias)
    """
    v     = states[:, IDX_V]
    delta = states[:, IDX_DELTA]
    accel = states[:, IDX_ACCEL]
    omega = dtheta / DT

    clean = _clean_channels(v, delta, accel, omega)          # (N, N_OBS)
    ch_std = np.maximum(clean.std(axis=0, keepdims=True), 1e-3)

    rng = np.random.default_rng(cfg.seed)
    # Per-rollout bias: one constant offset per (rollout, channel).
    n_rollouts = int(rollout_of.max()) + 1
    bias_per_rollout = rng.normal(0.0, 1.0, size=(n_rollouts, N_OBS)).astype(np.float32)
    bias = bias_per_rollout[rollout_of] * (cfg.bias_scale * ch_std)

    iid = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32) * (cfg.iid_scale * ch_std)
    return clean + bias + iid
