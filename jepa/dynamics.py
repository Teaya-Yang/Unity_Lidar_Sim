"""
dynamics.py
===========
The ANALYTIC side of the comparison, plus the shared pose integrator every model
(learned or analytic) is rolled through.

Three things live here:

  1. `analytic_step` — a faithful, batched NumPy copy of the simulator's
     ApplyBicycleDynamics / the planners' `_rollout_step`. Parameterised by an
     eta dict so the SAME code serves all three analytic baselines below.

  2. `integrate_pose` — the position/heading update. Shared by the learned model
     so that a position error is never an artefact of two different integrators.
     Matches `_rollout_step` exactly: NEW speed, OLD heading.

  3. `estimate_eta_ls` — closed-form least squares recovery of the three
     identifiable parameters (ACCEL_TAU, DRAG_COEFF, L) from a context window.
     This powers the STRONG analytic baseline. Without it the analytic model is
     handicapped (fixed nominal, no adaptation) and beating it proves little.

Why three analytic baselines matter
-----------------------------------
In simulation the analytic equations ARE the plant (Unity runs the same five
lines), so on conditions A and B the analytic model's only deficiency is not
knowing eta. That makes `analytic_oracle` — the model handed the TRUE eta — an
upper bound no learned model can beat on A/B. Reporting it keeps the evaluation
honest: a learned model "winning" on A/B only ever means it beat a handicapped
baseline, whereas on condition C even the oracle has an irreducible error floor,
because no parameter setting of dθ = (v/L)·tan(δ) can produce a v²·δ slip term.
That gap is the only place a learned model can win *structurally*.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import DT, IDX_TH, IDX_V, IDX_DELTA, IDX_ACCEL


# ── Nominal parameters ───────────────────────────────────────────────────────
# Imported from the live controller when available so the baseline is literally
# the model the planners use; falls back to the same constants otherwise (this
# module must stay importable without CasADi / ML-Agents).
def _load_nominal() -> Dict[str, float]:
    keys = ("L", "DRAG_COEFF", "ACCEL_TAU", "MAX_STEER_RATE", "STEER_ROLLOFF_SPD",
            "STEER_ROLLOFF_MIN", "A_MIN", "A_MAX", "DELTA_LIM")
    try:
        import taxi_controller_mppi as tc
        return {k: float(getattr(tc, k)) for k in keys}
    except Exception:
        return dict(L=6.0, DRAG_COEFF=0.04, ACCEL_TAU=0.5, MAX_STEER_RATE=0.6,
                    STEER_ROLLOFF_SPD=15.0, STEER_ROLLOFF_MIN=0.25,
                    A_MIN=-4.0, A_MAX=1.5, DELTA_LIM=0.5)


NOMINAL: Dict[str, float] = _load_nominal()


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]. Heading differences must never be taken raw: a rollout
    that crosses the branch cut would otherwise show a ~2*pi error spike."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ── Shared pose integration ──────────────────────────────────────────────────
def integrate_pose(x: np.ndarray, y: np.ndarray, th: np.ndarray,
                   v_new: np.ndarray, dtheta: np.ndarray) -> tuple:
    """One pose step, matching `_rollout_step` exactly: the position update uses
    the NEW speed with the OLD heading, and the heading is advanced afterwards.

    Every model in the evaluation goes through this function, so any position
    error reflects the model's speed/heading prediction, never a difference in
    integration convention.
    """
    x_new  = x + v_new * np.cos(th) * DT
    y_new  = y + v_new * np.sin(th) * DT
    th_new = th + dtheta
    return x_new, y_new, th_new


# ── The analytic bicycle model ───────────────────────────────────────────────
def analytic_step(state: np.ndarray, action: np.ndarray,
                  eta: Optional[Dict[str, float]] = None) -> np.ndarray:
    """One step of the augmented kinematic bicycle, batched.

    state  : (B, 6) [x, y, theta, v, delta_actual, accel_actual]
    action : (B, 2) [a_cmd, delta_cmd]
    eta    : parameter dict; NOMINAL if None. Scalar values, or (B,) arrays for
             per-sample parameters (used by the least-squares baseline).

    Returns (B, 6).
    """
    e = NOMINAL if eta is None else eta
    v     = state[:, IDX_V]
    delta = state[:, IDX_DELTA]
    accel = state[:, IDX_ACCEL]
    a_cmd, delta_cmd = action[:, 0], action[:, 1]

    # 1+2 — rate-limited, speed-dependent steering authority
    speed_frac  = np.clip(v / np.maximum(e["STEER_ROLLOFF_SPD"], 1e-3), 0.0, 1.0)
    eff_limit   = e["DELTA_LIM"] * (1.0 - speed_frac * (1.0 - e["STEER_ROLLOFF_MIN"]))
    delta_tgt   = np.clip(delta_cmd, -eff_limit, eff_limit)
    max_step    = e["MAX_STEER_RATE"] * DT
    delta_new   = delta + np.clip(delta_tgt - delta, -max_step, max_step)

    # 3 — first-order acceleration lag
    a_clamped = np.clip(a_cmd, e["A_MIN"], e["A_MAX"])
    tau       = np.maximum(e["ACCEL_TAU"], 1e-3)
    accel_new = accel + (a_clamped - accel) * (DT / tau)

    # 4 — drag + speed integration (speed is floored at zero, as in Unity)
    v_new = np.maximum(0.0, v + (accel_new - e["DRAG_COEFF"] * v) * DT)

    # Bicycle geometry
    dtheta = v_new / np.maximum(e["L"], 1e-3) * np.tan(delta_new) * DT
    x_new, y_new, th_new = integrate_pose(
        state[:, 0], state[:, 1], state[:, IDX_TH], v_new, dtheta)

    return np.stack([x_new, y_new, th_new, v_new, delta_new, accel_new], axis=1)


# ── Online parameter estimation (the STRONG analytic baseline) ───────────────
def estimate_eta_ls(ctx_states: np.ndarray, ctx_actions: np.ndarray,
                    ctx_next: np.ndarray) -> Dict[str, np.ndarray]:
    """Recover (ACCEL_TAU, DRAG_COEFF, L) per sample by closed-form least squares
    on a context window. The remaining six parameters stay nominal: the limits
    (A_MIN/A_MAX/DELTA_LIM) and the steering envelope are only identifiable when
    the trajectory actually saturates them, which most windows do not, so
    estimating them from short windows adds variance rather than accuracy.

    ctx_states  : (B, K, 6) states at t
    ctx_actions : (B, K, 2) actions at t
    ctx_next    : (B, K, 6) states at t+1

    Each estimate falls back to NOMINAL where the regressor is degenerate (a
    near-zero denominator means the window contains no excitation for that
    parameter — e.g. straight-line driving says nothing about wheelbase).
    """
    B = ctx_states.shape[0]
    v      = ctx_states[:, :, IDX_V]
    accel  = ctx_states[:, :, IDX_ACCEL]
    v_n     = ctx_next[:, :, IDX_V]
    accel_n = ctx_next[:, :, IDX_ACCEL]
    delta_n = ctx_next[:, :, IDX_DELTA]
    a_cmd   = ctx_actions[:, :, 0]

    dtheta = wrap_angle(ctx_next[:, :, IDX_TH] - ctx_states[:, :, IDX_TH])

    def _ls(num: np.ndarray, den: np.ndarray, fallback: float,
            lo: float, hi: float) -> np.ndarray:
        """Per-sample ratio with a degeneracy guard and a sanity clamp."""
        out = np.full(B, fallback, dtype=float)
        ok  = den > 1e-8
        out[ok] = num[ok] / den[ok]
        return np.clip(out, lo, hi)

    # ACCEL_TAU:  accel_{t+1} - accel_t = (a_clamped - accel_t) * (DT / tau)
    a_cl  = np.clip(a_cmd, NOMINAL["A_MIN"], NOMINAL["A_MAX"])
    resid = accel_n - accel
    drive = a_cl - accel
    g     = _ls((resid * drive).sum(1), (drive * drive).sum(1),
                DT / NOMINAL["ACCEL_TAU"], 1e-3, 1.0)     # g = DT / tau
    accel_tau = DT / g

    # DRAG_COEFF:  (v_{t+1} - v_t)/DT - accel_{t+1} = -c * v_t
    r = (v_n - v) / DT - accel_n
    c = _ls(-(r * v).sum(1), (v * v).sum(1), NOMINAL["DRAG_COEFF"], 0.0, 0.5)

    # L:  dtheta = (v_{t+1} / L) * tan(delta_{t+1}) * DT   ->  regress 1/L
    u       = v_n * np.tan(delta_n) * DT
    inv_L   = _ls((dtheta * u).sum(1), (u * u).sum(1), 1.0 / NOMINAL["L"],
                  1.0 / 30.0, 1.0 / 1.0)
    L = 1.0 / inv_L

    eta = {k: np.full(B, val, dtype=float) for k, val in NOMINAL.items()}
    eta["ACCEL_TAU"]  = accel_tau
    eta["DRAG_COEFF"] = c
    eta["L"]          = L
    return eta


def eta_from_json(eta_dicts, keys=None) -> Dict[str, np.ndarray]:
    """Stack per-rollout eta dicts (from the packed `eta_json`) into arrays,
    for the ORACLE baseline. Missing keys fall back to nominal — condition C
    rollouts carry extra unmodeled keys which the analytic model has no slot
    for, and that is precisely the point of condition C."""
    keys = keys or list(NOMINAL.keys())
    return {k: np.array([float(d.get(k, NOMINAL[k])) for d in eta_dicts])
            for k in keys}
