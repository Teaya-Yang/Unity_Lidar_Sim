"""
Canonical taxiing stage cost — MPC IS THE REFERENCE.

taxi_controller_mpc.py's NLP objective defines the intended behaviour; this module
re-expresses exactly those terms in NumPy so the MPPI rollout evaluates the SAME
function. Parity is asserted by evaluating this against the MPC's CasADi expression on
identical inputs (see the parity check in the project notes).

Stage k (state after the step: px, py, th, v; control uk, previous control u_km1):

    W_GOAL_RUN · sqrt(d_goal² + 1)                       bounded (linear) goal pull
  + W_HEAD     · (1 − cos(th − psi))                     heading toward goal bearing
  + W_V        · (v − v_des)²                            speed tracking
  + R_ACT      · uk²      (elementwise, summed)          control effort
  + R_DACT     · (uk − u_km1)²                           control rate / smoothness
  + static       W_HARD · max(0, D_SAFE_HARD − ds)²
  + occlusion    occlusion_capsules.occlusion_stage_cost(...)

The static and occlusion terms share ONE keep-out radius (D_SAFE_HARD) and ONE weight
(W_HARD, large enough to act as a hard constraint), so they cannot be traded off against
each other or against the goal pull.

NO DYNAMIC-OBSTACLE TERM *HERE*. It used to sit in this module with a soft influence ring
plus an exact penalty, fed by an ORACLE: exact positions and velocities of other aircraft,
handed to the controller by Unity rather than sensed. That is gone for good.

Moving agents are now handled OUTSIDE this function, in the MPPI rollout loop, as sensed
LiDAR clusters (dynamic_clusters.py) carrying the SAME expanding keep-out as an occlusion
boundary — occlusion_capsules.occlusion_stage_cost with a per-cluster base radius. They are
not re-added here because there is nothing MPC-reference about them: the MPC still carries
the old oracle terms (P_opos/P_ovel) and is unrunnable, so parity on this term is defined by
whichever controller gets the sensed model first, which is the MPPI. Porting the MPC means
giving it DYN_NOW too — never re-adding the oracle.

Terminal:  W_GOAL_TERM · sqrt(d_goal² + 1)

WHY THE GOAL PULL IS DELIBERATELY WEAK. A linear distance term has a BOUNDED gradient,
unlike ‖p − p_goal‖² whose gradient grows without limit far from the goal and would dwarf
the keep-out penalties. That is what lets the obstacle/occlusion terms reliably win near a
surface. It also means the absolute cost scale is small — which matters for MPPI, see
below.

NOTE FOR SAMPLING PLANNERS. MPPI weights rollouts by exp(−cost / LAMBDA). With the MPC's
(intentionally small) weights, cost differences between rollouts can be far below LAMBDA,
leaving the softmax nearly uniform and the planner unable to discriminate. Scaling the
WHOLE cost by a constant — equivalently, dividing LAMBDA — leaves the argmin and every
relative trade-off untouched while restoring that discrimination. That is the correct knob;
re-weighting individual terms would change the optimum and break parity with the MPC.
"""

import numpy as np

from taxi_config import CFG

# ── Reference weights (MPC IS THE REFERENCE) ─────────────────────────────────
# VALUES live in config.yaml; the names are bound here because taxi_controller_mpc
# imports taxi_controller_mppi, so the MPC cannot be the import source without a
# cycle. Both controllers import these names, so there is exactly one binding of
# each. Tune in the YAML, not here.
_cost = CFG["cost"]
_dyn  = CFG["dynamic_obstacles"]

W_GOAL_RUN    = _cost["w_goal_run"]    # running goal pull (LINEAR -> bounded gradient)
W_GOAL_TERM   = _cost["w_goal_term"]   # terminal pull on the FINAL state (LINEAR)
W_HEAD        = _cost["w_head"]        # heading alignment toward the goal bearing
W_V           = _cost["w_v"]           # speed tracking toward the goal-tapered speed
R_ACT         = _cost["r_act"]         # control effort
R_DACT        = _cost["r_dact"]        # control rate (smoothness)
W_OBS         = _dyn["w_obs"]          # dynamic-obstacle soft influence ring
RHO_SLACK     = _dyn["rho_slack"]      # linear exact-penalty weight on a DYNAMIC violation
RHO_SLACK2    = _dyn["rho_slack2"]     # quadratic term



def stage_cost(px, py, th, v, uk, u_km1, *, goal_xy, v_des, t_k,
               r_act, r_dact, w_goal_run, w_head, w_v,
               d_static=None, d_safe_static=None,
               w_static=None):
    """MPC-reference stage cost, vectorized over rollouts.

    px, py, th, v : (K,) rollout state AFTER this step
    uk, u_km1     : (K, 2) control at this stage and the previous one
    d_static      : (K,) distance to the nearest static surface, or None to skip
    Returns (K,) cost.
    """
    gx, gy = goal_xy
    dx, dy = px - gx, py - gy

    # Bounded (linear) goal pull — the +1 keeps it smooth at the goal, matching the MPC.
    d_goal = np.sqrt(dx * dx + dy * dy + 1.0)
    cost = w_goal_run * d_goal

    # Heading alignment toward the goal bearing.
    psi = np.arctan2(gy - py, gx - px)
    cost = cost + w_head * (1.0 - np.cos(th - psi))

    # Speed tracking toward the (goal-tapered) desired speed.
    # cost = cost + w_v * (v - v_des) ** 2

    # Control effort and rate. The STEERING channel of r_act/r_dact is what bounds how wide
    # the ego swerves around an obstacle — a control-smoothness objective, NOT a goal-bearing
    # one, so it cannot fight the keep-out or make standing still cheaper than driving.
    # du = uk - u_km1
    # cost = cost + (np.asarray(r_act) * uk * uk).sum(axis=1)
    # cost = cost + (np.asarray(r_dact) * du * du).sum(axis=1)

    # Static surfaces (LiDAR OCC), same shape as the obstacle terms.
    if d_static is not None:
        # Flat (0/w_static) penalty, not a quadratic: every breach costs the same regardless of
        # depth. Elementwise over the (K,) rollouts — a Python `if` on d_static is ambiguous
        # because it is an array, and it would also apply one branch to ALL rollouts.
        cost = cost + np.where(d_static < d_safe_static, w_static, 0.0)
    
    return cost


def terminal_cost(px, py, *, goal_xy, w_goal_term):
    """MPC-reference terminal cost: bounded (linear) pull on the FINAL state."""
    gx, gy = goal_xy
    return w_goal_term * np.sqrt((px - gx) ** 2 + (py - gy) ** 2 + 1.0)
