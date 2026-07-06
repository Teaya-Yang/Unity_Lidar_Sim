"""
value_net.py
============
Belief-conditioned terminal value function V(s, b) for the MPPI planner.
Self-contained NumPy module: MLP core + feature encoders + inference wrapper.

The network predicts the discounted COST-TO-GO (lower = better) from a
situation described ego-relatively:

    ego block (4):            [v, d_cross, theta_e, goal_remaining]
    per obstacle × K_OBS (8): [fwd_d, lat_d, closing, cross_v,
                               b_straight, b_left, b_right, valid]

    → FEAT_DIM = 4 + 3·8 = 28 features, one scalar out.

The SAME situation encoding is produced from two sources:
  * features_from_obs()   — a logged 20-D observation (training data), and
  * terminal_features()   — the K rollout endpoints inside mppi(), with each
                            obstacle propagated to the horizon time by the
                            same constant-velocity prediction the stage costs
                            use, so train/plan semantics match.

Usage inside the controller (see taxi_controller.py --value-net):

    vf = ValueFunction.load("value.npz")
    X  = terminal_features(st, obstacles, beliefs, goal, T, frenet_mode,
                           tangent, d0, s0_fwd, s0_lat)      # (K, 28)
    cost += W_TERM * GAMMA_VALUE**H_MPPI * vf(X)             # (K,)

Training lives in train_value.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Kept in sync with taxi_controller.py. Duplicated (not imported) so this module
# has no import-time dependency on the controller and the layout a checkpoint
# was trained under travels with the code that reads it.
K_OBS        = 3                     # obstacle slots in the observation
N_ROUTE      = 3                     # route hypotheses {straight, branch-L, branch-R}
OBS_SENTINEL = 900.0                 # |dy| above this ⇒ zero-padded obstacle slot
EGO_FEATS    = 4
OBS_FEATS    = 4 + N_ROUTE + 1       # geometry(4) + belief(3) + valid(1)
FEAT_DIM     = EGO_FEATS + K_OBS * OBS_FEATS   # 28

UNIFORM_BELIEF = np.full(N_ROUTE, 1.0 / N_ROUTE)


# ── MLP core (tanh hidden layers, linear scalar head) ─────────────────────────

@dataclass
class MLPParams:
    """Weights + input/target normalisation. Everything needed for inference."""
    weights: List[np.ndarray]          # per-layer weight matrices (in_dim, out_dim)
    biases:  List[np.ndarray]          # per-layer bias vectors (out_dim,)
    x_mean:  np.ndarray                 # feature standardisation (FEAT_DIM,)
    x_std:   np.ndarray
    y_mean:  np.ndarray                 # target standardisation (1,)
    y_std:   np.ndarray

    def save(self, path: str) -> None:
        arrays = {
            "n_layers": np.int64(len(self.weights)),
            "x_mean": self.x_mean, "x_std": self.x_std,
            "y_mean": self.y_mean, "y_std": self.y_std,
        }
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = w.astype(np.float32)
            arrays[f"b{i}"] = b.astype(np.float32)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "MLPParams":
        d = np.load(path)
        n = int(d["n_layers"])
        return cls(
            weights=[d[f"W{i}"].astype(np.float64) for i in range(n)],
            biases=[d[f"b{i}"].astype(np.float64) for i in range(n)],
            x_mean=d["x_mean"].astype(np.float64), x_std=d["x_std"].astype(np.float64),
            y_mean=d["y_mean"].astype(np.float64), y_std=d["y_std"].astype(np.float64),
        )

    @classmethod
    def init(cls, in_dim: int, out_dim: int, hidden_sizes: List[int],
             rng: np.random.Generator) -> "MLPParams":
        """He-ish init for a fresh network; norm stats default to identity."""
        dims = [in_dim] + list(hidden_sizes) + [out_dim]
        weights, biases = [], []
        for din, dout in zip(dims[:-1], dims[1:]):
            weights.append(rng.normal(0.0, np.sqrt(2.0 / din), size=(din, dout)))
            biases.append(np.zeros(dout))
        return cls(weights=weights, biases=biases,
                   x_mean=np.zeros(in_dim), x_std=np.ones(in_dim),
                   y_mean=np.zeros(out_dim), y_std=np.ones(out_dim))


def forward(params: MLPParams, x_norm: np.ndarray, cache: bool = False):
    """
    Forward pass on ALREADY-STANDARDISED inputs, returning STANDARDISED outputs.
    x_norm: (B, FEAT_DIM). If cache=True also returns activations/pre-activations
    for train_value.py's backprop.
    """
    acts = [x_norm]
    pre  = []
    a = x_norm
    n = len(params.weights)
    for i, (w, b) in enumerate(zip(params.weights, params.biases)):
        z = a @ w + b
        pre.append(z)
        a = np.tanh(z) if i < n - 1 else z   # linear output head
        acts.append(a)
    if cache:
        return a, acts, pre
    return a


# ── Situation encoders (shared by training and planning) ──────────────────────

def _is_frenet_obs(obs: np.ndarray) -> bool:
    """Same stub-tangent test as taxi_controller.obs_to_state."""
    return not (abs(float(obs[18])) < 1e-4 and abs(float(obs[19]) - 1.0) < 1e-4)


def features_from_obs(obs: np.ndarray,
                      belief_slots: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Encode ONE logged 20-D observation into the (FEAT_DIM,) situation vector.

    belief_slots : (K_OBS, N_ROUTE) per-slot route posteriors aligned with the
                   observation's obstacle slots (the recorder saves these), or
                   None → uniform prior in every slot (the V(s) ablation).
    """
    obs = np.asarray(obs, dtype=np.float64)
    if belief_slots is None:
        belief_slots = np.tile(UNIFORM_BELIEF, (K_OBS, 1))

    v, d_cross, theta_e, goal_rem = float(obs[3]), float(obs[1]), float(obs[2]), float(obs[16])

    # Ego forward/left axes in the observation's (Z, X) world axes.
    if _is_frenet_obs(obs):
        tan_x, tan_z = float(obs[18]), float(obs[19])
        n = float(np.hypot(tan_x, tan_z)) or 1.0
        cth, sth = tan_z / n, tan_x / n          # fwd_hat = (tan_z, tan_x)/|t|
    else:
        cth, sth = float(np.cos(obs[2])), float(np.sin(obs[2]))

    feats = [v, d_cross, theta_e, goal_rem]
    for i in range(K_OBS):
        base = 4 + i * 4
        dz, dx, vz, vx = (float(obs[base + j]) for j in range(4))
        if abs(dx) > OBS_SENTINEL:               # padded slot
            feats += [0.0] * 4 + list(UNIFORM_BELIEF) + [0.0]
            continue
        fwd_d   =  dz * cth + dx * sth
        lat_d   = -dz * sth + dx * cth
        closing = -(vz * cth + vx * sth)          # > 0 ⇒ moving toward the ego
        cross_v = -vz * sth + vx * cth
        feats += [fwd_d, lat_d, closing, cross_v] + list(belief_slots[i]) + [1.0]
    return np.asarray(feats, dtype=np.float64)


def terminal_features(st: np.ndarray,
                      obstacles: Sequence[Tuple[np.ndarray, np.ndarray]],
                      beliefs: Optional[Sequence[Optional[np.ndarray]]],
                      goal: float,
                      T: float,
                      frenet_mode: bool,
                      tangent: Optional[np.ndarray] = None,
                      d0: float = 0.0,
                      s0_fwd: float = 0.0,
                      s0_lat: float = 0.0) -> np.ndarray:
    """
    Encode ALL K rollout endpoints into a (K, FEAT_DIM) batch, matching
    features_from_obs() semantics.

    st        : (K, 6) terminal rollout states [z, x, theta, v, delta, accel]
                (frenet mode: z/x are displacements from a zeroed origin —
                exactly what mppi()'s _rollout_step produces).
    obstacles : the mppi() obstacle list [(rel_xy, obs_v), ...], nearest-first,
                ego-relative at t=0 in world (Z, X) axes.
    beliefs   : per-obstacle posteriors aligned with `obstacles` (None entries
                or beliefs=None → uniform prior).
    T         : prediction time of the endpoint [s] (= H_MPPI · DT). Each
                obstacle is advanced by obs_v·T — the same constant-velocity
                ray the stage costs used.
    d0/s0_*   : mppi()'s frame bookkeeping (initial cross-track error and the
                global-mode start offsets) so cross-track/goal math matches the
                stage costs exactly.
    """
    st = np.asarray(st, dtype=np.float64)
    K  = st.shape[0]
    v  = st[:, 3]

    if frenet_mode:
        tan_x, tan_z = float(tangent[0]), float(tangent[1])
        n = float(np.hypot(tan_x, tan_z)) or 1.0
        th_tan  = np.arctan2(tan_x, tan_z)
        d_t     = d0 + tan_z * st[:, 1] - tan_x * st[:, 0]
        theta_e = th_tan - st[:, 2]
        goal_rem = goal - (tan_z * st[:, 0] + tan_x * st[:, 1])
        cth = np.full(K, tan_z / n)               # path-frame axes, constant over K
        sth = np.full(K, tan_x / n)
    else:
        d_t     = st[:, 1]
        theta_e = st[:, 2]
        goal_rem = goal - (st[:, 0] - s0_fwd)
        cth, sth = np.cos(st[:, 2]), np.sin(st[:, 2])

    cols = [v, d_t, theta_e, goal_rem]
    for i in range(K_OBS):
        if i < len(obstacles):
            rel, ov = obstacles[i]
            b = None if beliefs is None else beliefs[i]
            if b is None:
                b = UNIFORM_BELIEF
            # Obstacle world offset from the rollout ORIGIN at time T, then
            # re-expressed relative to each rollout endpoint.
            oz = s0_fwd + float(rel[0]) + float(ov[0]) * T - st[:, 0]
            ox = s0_lat + float(rel[1]) + float(ov[1]) * T - st[:, 1]
            fwd_d   =  oz * cth + ox * sth
            lat_d   = -oz * sth + ox * cth
            closing = -(float(ov[0]) * cth + float(ov[1]) * sth)
            cross_v = -float(ov[0]) * sth + float(ov[1]) * cth
            cols += [fwd_d, lat_d, closing, cross_v,
                     np.full(K, float(b[0])), np.full(K, float(b[1])),
                     np.full(K, float(b[2])), np.ones(K)]
        else:
            cols += [np.zeros(K)] * 4 + \
                    [np.full(K, p) for p in UNIFORM_BELIEF] + [np.zeros(K)]
    return np.stack(cols, axis=1)


# ── Inference wrapper ──────────────────────────────────────────────────────────

class ValueFunction:
    """
    Inference wrapper for the terminal value inside taxi_controller.py.

        vf = ValueFunction.load("value.npz")
        cost_to_go = vf(X)          # X: (K, FEAT_DIM) → (K,) — lower is better
    """

    def __init__(self, params: MLPParams):
        if params.x_mean.shape[0] != FEAT_DIM:
            raise ValueError(
                f"Checkpoint expects {params.x_mean.shape[0]}-D features, but this "
                f"code builds {FEAT_DIM}-D — feature layout mismatch, retrain.")
        self.p = params

    @classmethod
    def load(cls, path: str) -> "ValueFunction":
        return cls(MLPParams.load(path))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        x_norm = (X - self.p.x_mean) / self.p.x_std
        y_norm = forward(self.p, x_norm)
        return (y_norm * self.p.y_std + self.p.y_mean).ravel()
