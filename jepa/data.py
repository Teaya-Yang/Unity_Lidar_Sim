"""
data.py
=======
Turns a packed split (`packed/<cond>/<split>.npz`) into context/horizon windows.

The one invariant this file exists to protect
---------------------------------------------
A window must NEVER cross a rollout boundary. Two rollouts were generated with
different eta, so a window spanning both describes a plant that does not exist,
and the context encoder would be trained to identify a chimera. `rollout_offsets`
(written by split_and_pack.py) marks the boundaries and every index produced here
is checked against them.

The state factorisation
-----------------------
The recorded state is [x, y, theta, v, delta_actual, accel_actual], but the
dynamics are translation- and rotation-invariant: how the taxi evolves depends
only on (v, delta, accel) and the commands, never on where it happens to be on
the map. So we split:

  CORE  = [v, delta_actual, accel_actual]   — 3-D, what the network predicts
  POSE  = [x, y, theta]                     — integrated afterwards

The observation (OBS)
---------------------
What the encoders actually consume. In the default (plain) mode OBS == CORE, so
nothing changes. In `enriched` mode OBS becomes a noisy, redundant 10-channel
sensor suite (see sensors.py) that encodes the same few DOF — the regime where
latent modelling is meant to help. Crucially the model still PREDICTS the clean
CORE (via the decoder), so every metric stays in clean physical units and the
A/B/C comparison is unaffected; only the input becomes high-dimensional and noisy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DT, IDX_TH, CORE_SLICE, DataConfig
from .dynamics import wrap_angle


# Plain-mode context feature layout:  [v, delta, accel, a_cmd, delta_cmd, dtheta]
N_CTX_FEAT = 6
N_CORE = 3          # [v, delta, accel]
N_PRED = 4          # [dv, ddelta, daccel, dtheta] — what the decoder emits


@dataclass
class Normalizer:
    """Feature standardisation. Fitted on condition-A train ONLY and then frozen.

    `obs_*` are only populated in enriched mode; in plain mode OBS == CORE so the
    core stats are reused and these stay None (keeps checkpoints backward
    compatible — the fields simply do not appear)."""
    ctx_mean: np.ndarray
    ctx_std: np.ndarray
    core_mean: np.ndarray
    core_std: np.ndarray
    pred_mean: np.ndarray
    pred_std: np.ndarray
    obs_mean: Optional[np.ndarray] = None
    obs_std: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, List[float]]:
        return {k: np.asarray(v).tolist()
                for k, v in self.__dict__.items() if v is not None}

    @staticmethod
    def from_dict(d: Dict) -> "Normalizer":
        return Normalizer(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


class RolloutData:
    """One packed split, with per-transition arrays and rollout bookkeeping."""

    def __init__(self, path: str, dcfg: Optional[DataConfig] = None):
        self.dcfg = dcfg or DataConfig()
        d = np.load(path, allow_pickle=True)
        self.x       = d["x"].astype(np.float32)         # (N, 6) state at t
        self.a       = d["a"].astype(np.float32)         # (N, 2) action at t
        self.x_next  = d["x_next"].astype(np.float32)    # (N, 6) state at t+1
        self.offsets = d["rollout_offsets"].astype(np.int64)
        self.n_rollouts = len(self.offsets) - 1

        self.eta      = [json.loads(s) for s in d["eta_json"]]
        self.controller = np.asarray(d["controller"])
        self.terminal   = np.asarray(d["terminal_reason"])

        self.dtheta = wrap_angle(self.x_next[:, IDX_TH] - self.x[:, IDX_TH]).astype(np.float32)

        # transition index -> rollout index (eta lookup / per-rollout slicing / bias)
        self.rollout_of = np.zeros(len(self.x), dtype=np.int64)
        for r in range(self.n_rollouts):
            self.rollout_of[self.offsets[r]:self.offsets[r + 1]] = r

        # Clean core state — always the decode target and the evaluation state.
        self.core = self.x[:, CORE_SLICE].astype(np.float32)              # (N, 3)

        # Observation — what the encoders consume.
        if self.dcfg.enriched:
            from .sensors import synthesize, SensorConfig, N_OBS
            scfg = SensorConfig(self.dcfg.sensor_iid_scale,
                                self.dcfg.sensor_bias_scale, self.dcfg.sensor_seed)
            self.obs = synthesize(self.x, self.dtheta, self.rollout_of, scfg)  # (N, N_OBS)
            self.obs_dim = N_OBS
            # Context = [obs, a_cmd, delta_cmd]. The gyro channel already carries
            # yaw rate, so no separate dtheta feature is needed here.
            self.ctx_feat = np.concatenate([self.obs, self.a], axis=1).astype(np.float32)
        else:
            self.obs = self.core
            self.obs_dim = N_CORE
            self.ctx_feat = np.concatenate([
                self.core, self.a, self.dtheta[:, None]], axis=1).astype(np.float32)

        self.ctx_dim = self.ctx_feat.shape[1]

        # What the decoder predicts: per-step increments (clean).
        self.pred_target = np.concatenate([
            (self.x_next[:, CORE_SLICE] - self.x[:, CORE_SLICE]),
            self.dtheta[:, None]], axis=1).astype(np.float32)             # (N, 4)

    # ── Window index construction ────────────────────────────────────────────
    def valid_starts(self, K: int, H: int, min_len: int = 0) -> np.ndarray:
        starts = []
        for r in range(self.n_rollouts):
            lo, hi = self.offsets[r], self.offsets[r + 1]
            if (hi - lo) < max(min_len, K + H + 1):
                continue
            starts.append(np.arange(lo + K, hi - H, dtype=np.int64))
        if not starts:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(starts)

    # ── Batch assembly ───────────────────────────────────────────────────────
    def gather(self, t: np.ndarray, K: int, H: int) -> Dict[str, np.ndarray]:
        """Build a batch from window start indices `t`.

        Returns, with B = len(t):
          ctx        (B, K, ctx_dim)   context features over [t-K, t)
          ctx_state  (B, K, 6)         raw states over [t-K, t)   (LS baseline)
          ctx_act    (B, K, 2)         actions over [t-K, t)
          ctx_next   (B, K, 6)         next-states over [t-K, t)
          obs0       (B, obs_dim)      OBSERVATION at t   (encoder input)
          obs_tgt    (B, H, obs_dim)   observations at t+1..t+H (EMA target input)
          core0      (B, 3)            clean core at t
          pose0/state0 (B,3)/(B,6)     pose / full state at t
          acts       (B, H, 2)
          core_tgt   (B, H, 3)         clean core at t+1..t+H (decode target)
          pred_tgt   (B, H, 4)         clean increments over [t, t+H)
          state_tgt  (B, H, 6)         clean full states t+1..t+H (evaluation)
          rollout    (B,)
        """
        ctx_idx = t[:, None] - K + np.arange(K)[None, :]      # (B, K)
        fut_idx = t[:, None] + np.arange(H)[None, :]           # (B, H)
        return dict(
            ctx=self.ctx_feat[ctx_idx],
            ctx_state=self.x[ctx_idx],
            ctx_act=self.a[ctx_idx],
            ctx_next=self.x_next[ctx_idx],
            obs0=self.obs[t],
            obs_tgt=self.obs[fut_idx],
            core0=self.core[t],
            pose0=self.x[t][:, :3],
            state0=self.x[t],
            acts=self.a[fut_idx],
            core_tgt=self.x_next[fut_idx][:, :, CORE_SLICE],
            pred_tgt=self.pred_target[fut_idx],
            state_tgt=self.x_next[fut_idx],
            rollout=self.rollout_of[t],
        )

    def eta_array(self, rollout_idx: np.ndarray, key: str,
                  default: float = 0.0) -> np.ndarray:
        return np.array([float(self.eta[r].get(key, default)) for r in rollout_idx])


# ── Loading helpers ──────────────────────────────────────────────────────────
def load_split(packed_root: str, condition: str, split: str,
               dcfg: Optional[DataConfig] = None) -> RolloutData:
    path = os.path.join(packed_root, condition, f"{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run split_and_pack.py for condition {condition} first.")
    return RolloutData(path, dcfg)


def fit_normalizer(data: RolloutData, K: int, H: int,
                   n_sample: int = 50000, seed: int = 0) -> Normalizer:
    """Fit standardisation stats on a random subsample of condition-A train."""
    rng = np.random.default_rng(seed)
    starts = data.valid_starts(K, H)
    if len(starts) > n_sample:
        starts = rng.choice(starts, n_sample, replace=False)
    b = data.gather(starts, K, H)

    def _ms(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = arr.mean(axis=0).astype(np.float32)
        s = arr.std(axis=0).astype(np.float32)
        return m, np.maximum(s, 1e-6)

    ctx_m, ctx_s   = _ms(b["ctx"].reshape(-1, data.ctx_dim))
    core_m, core_s = _ms(b["core0"])
    pred_m, pred_s = _ms(b["pred_tgt"].reshape(-1, N_PRED))
    norm = Normalizer(ctx_m, ctx_s, core_m, core_s, pred_m, pred_s)
    if data.dcfg.enriched:
        norm.obs_mean, norm.obs_std = _ms(b["obs0"])
    return norm


class WindowSampler:
    """Uniform i.i.d. sampling of valid windows."""

    def __init__(self, data: RolloutData, K: int, H: int, seed: int = 0,
                 min_len: int = 0):
        self.data = data
        self.K, self.H = K, H
        self.starts = data.valid_starts(K, H, min_len)
        if len(self.starts) == 0:
            raise ValueError(
                f"no valid windows for K={K}, H={H} — rollouts are too short.")
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.starts)

    def batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        t = self.rng.choice(self.starts, batch_size, replace=True)
        return self.data.gather(t, self.K, self.H)

    def all_windows(self, stride: int = 1, max_n: Optional[int] = None
                    ) -> Dict[str, np.ndarray]:
        s = self.starts[::stride]
        if max_n is not None and len(s) > max_n:
            s = s[:max_n]
        return self.data.gather(s, self.K, self.H)
