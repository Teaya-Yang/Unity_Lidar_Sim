"""
rollout.py
==========
One open-loop multi-step rollout interface, implemented once, used by every model
in the comparison.

This file exists to make the evaluation FAIR. The temptation is to let each model
roll itself forward its own way, but then a position error could come from a
different integration convention rather than from a better dynamics prediction.
Here every predictor — analytic, least-squares-adapted, oracle, or learned —
produces the same thing per step:

    (v_new, delta_new, accel_new, dtheta)

and the pose is then advanced by the single shared `integrate_pose`, which
matches the simulator's convention exactly (NEW speed, OLD heading).

"Open loop" means each model consumes its OWN prediction as the next input; the
ground truth is never re-injected. That is what makes errors compound, and it is
exactly how a planner uses the model over an H-step horizon — so the metric
reflects planning use, not a flattering one-step score.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .config import DT, IDX_TH, IDX_V, IDX_DELTA, IDX_ACCEL, CORE_SLICE
from .dynamics import analytic_step, integrate_pose, wrap_angle


def rollout_analytic(state0: np.ndarray, acts: np.ndarray,
                     eta: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """Roll the analytic model H steps.

    state0 (B,6), acts (B,H,2) -> (B,H,6) predicted states at t+1..t+H.

    `eta` may hold scalars (fixed nominal) or (B,) arrays (per-sample estimated
    or oracle parameters); analytic_step handles both by broadcasting.
    """
    B, H, _ = acts.shape
    st = state0.copy()
    out = np.empty((B, H, 6), dtype=np.float64)
    for h in range(H):
        st = analytic_step(st, acts[:, h], eta)
        out[:, h] = st
    return out


def rollout_learned(model, b: Dict[str, np.ndarray], device: str = "cpu") -> np.ndarray:
    """Roll the learned model H steps and integrate the pose the same way.

    The network predicts the clean CORE state (v, delta, accel) and the per-step
    heading change; x/y/theta are integrated here so that the learned model and
    the analytic baselines share an identical pose integrator. The encoder is fed
    the OBSERVATION (obs0) — equal to the clean core in plain mode, the noisy
    sensor suite in enriched mode — while the pose is integrated from the clean
    recorded state, so a position error never reflects an integrator difference.
    """
    import torch

    acts   = b["acts"]
    state0 = b["state0"].astype(np.float64)
    B, H, _ = acts.shape
    with torch.no_grad():
        core, dth = model.rollout(
            torch.as_tensor(b["ctx"], dtype=torch.float32, device=device),
            torch.as_tensor(b["obs0"], dtype=torch.float32, device=device),
            torch.as_tensor(acts, dtype=torch.float32, device=device),
        )
    core = core.cpu().numpy().astype(np.float64)     # (B,H,3)
    dth  = dth.cpu().numpy().astype(np.float64)      # (B,H)

    out = np.empty((B, H, 6), dtype=np.float64)
    x, y, th = state0[:, 0].copy(), state0[:, 1].copy(), state0[:, IDX_TH].copy()
    for h in range(H):
        v_new = np.maximum(core[:, h, 0], 0.0)       # speed cannot go negative
        x, y, th = integrate_pose(x, y, th, v_new, dth[:, h])
        out[:, h, 0] = x
        out[:, h, 1] = y
        out[:, h, IDX_TH]    = th
        out[:, h, IDX_V]     = v_new
        out[:, h, IDX_DELTA] = core[:, h, 1]
        out[:, h, IDX_ACCEL] = core[:, h, 2]
    return out


# ── Error metrics ────────────────────────────────────────────────────────────
def per_horizon_errors(pred: np.ndarray, truth: np.ndarray) -> Dict[str, np.ndarray]:
    """RMSE at each horizon step, reported per physical quantity.

    Quantities are kept SEPARATE rather than summed into one number: metres,
    radians and m/s do not share a scale, and collapsing them would make the
    headline metric depend on an arbitrary unit choice.

    pred, truth : (B, H, 6)  ->  dict of (H,) arrays
    """
    d_pos = np.linalg.norm(pred[:, :, :2] - truth[:, :, :2], axis=-1)     # (B,H)
    d_th  = np.abs(wrap_angle(pred[:, :, IDX_TH] - truth[:, :, IDX_TH]))
    d_v   = np.abs(pred[:, :, IDX_V] - truth[:, :, IDX_V])
    d_dl  = np.abs(pred[:, :, IDX_DELTA] - truth[:, :, IDX_DELTA])

    def _rmse(e):  return np.sqrt((e ** 2).mean(axis=0))
    return dict(pos=_rmse(d_pos), theta=_rmse(d_th), v=_rmse(d_v), delta=_rmse(d_dl))


# ── Teacher forcing + compounding ratio (the SkyJEPA Fig. 6 comparison) ──────
def teacher_forced(model, b: Dict[str, np.ndarray], device: str = "cpu") -> np.ndarray:
    """One-step-from-truth predictions over the horizon.

    At each step h the model predicts state t+h+1 from the TRUE state at t+h
    (re-encoded), rather than from its own previous prediction. Comparing this
    against the free rollout isolates the error caused purely by recursion.

    Works for any model exposing `encode_context` + `step_from_core` (both the
    JEPA and the predictive baseline do). Pose is integrated from the true
    previous pose each step, via the shared integrator.
    """
    import torch

    ctx  = torch.as_tensor(b["ctx"], dtype=torch.float32, device=device)
    acts = torch.as_tensor(b["acts"], dtype=torch.float32, device=device)
    B, H, _ = acts.shape
    c = model.encode_context(ctx)

    obs0     = b["obs0"].astype(np.float64)                       # (B,obs_dim)
    obs_tgt  = b["obs_tgt"].astype(np.float64)                    # (B,H,obs_dim)
    pose0    = b["state0"][:, :3].astype(np.float64)              # (B,3)
    pose_tgt = b["state_tgt"][:, :, :3].astype(np.float64)        # (B,H,3)

    out = np.empty((B, H, 6), dtype=np.float64)
    for h in range(H):
        prev_obs  = obs0 if h == 0 else obs_tgt[:, h - 1]         # TRUE observation
        prev_pose = pose0 if h == 0 else pose_tgt[:, h - 1]
        with torch.no_grad():
            core_next, dth = model.step_from_obs(
                torch.as_tensor(prev_obs, dtype=torch.float32, device=device),
                acts[:, h],
                c)
        core_next = core_next.cpu().numpy().astype(np.float64)
        dth = dth.cpu().numpy().astype(np.float64)
        v_new = np.maximum(core_next[:, 0], 0.0)
        x, y, th = integrate_pose(prev_pose[:, 0], prev_pose[:, 1], prev_pose[:, 2],
                                  v_new, dth)
        out[:, h, 0], out[:, h, 1], out[:, h, IDX_TH] = x, y, th
        out[:, h, IDX_V], out[:, h, IDX_DELTA], out[:, h, IDX_ACCEL] = \
            v_new, core_next[:, 1], core_next[:, 2]
    return out


def compounding_ratio(model, b: Dict[str, np.ndarray], device: str = "cpu"
                      ) -> Dict[str, np.ndarray]:
    """Per-horizon compounding ratio CR_k = e_rollout / e_teacherforced, and the
    error-growth rate ER_k = e_rollout[k] - e_rollout[k-1], on POSITION error.

    CR near 1 means recursion adds little error; growing well above 1 means the
    model's own imperfect outputs are destabilising the rollout. This is the
    SkyJEPA paper's headline evidence (their Fig. 6) that latent rollout
    compounds less than autoregressive state-space prediction.
    """
    truth = b["state_tgt"].astype(np.float64)
    e_roll = per_horizon_errors(rollout_learned(model, b, device), truth)["pos"]
    e_tf   = per_horizon_errors(teacher_forced(model, b, device), truth)["pos"]
    cr = e_roll / np.maximum(e_tf, 1e-9)
    er = np.diff(e_roll, prepend=0.0)
    return dict(cr=cr, er=er, e_roll=e_roll, e_tf=e_tf)


def summarize(errs: Dict[str, np.ndarray], horizons=(1, 5, 10, 15)) -> Dict[str, float]:
    """Pick out a few horizons for a compact table. 1-step shows raw fidelity;
    the long horizon shows compounding, which is what a planner actually feels."""
    out = {}
    H = len(errs["pos"])
    for h in horizons:
        if h <= H:
            for k in ("pos", "theta", "v"):
                out[f"{k}@{h}"] = float(errs[k][h - 1])
    return out
