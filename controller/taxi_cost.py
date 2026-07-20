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
  + Σ_obstacles  W_OBS · max(0, D_INFL − d)²  +  ρ·viol + ρ₂·viol²,  viol = (D_SAFE² − d²)₊
  + static       W_STATIC_RING · max(0, D_INFL_STATIC − ds)²  +  ρ·viols + ρ₂·viols²
  + occlusion    occlusion_capsules.occlusion_stage_cost(...)

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

# ── Reference weights (MPC IS THE REFERENCE) ─────────────────────────────────
# Defined here rather than in taxi_controller_mpc.py because taxi_controller_mpc
# imports taxi_controller, so the MPC cannot be the import source without a cycle.
# Both controllers import these, so there is exactly one definition of each.
W_GOAL_RUN    = 0.05    # running goal pull (LINEAR distance -> bounded gradient)
W_GOAL_TERM   = 2.0     # terminal pull on the FINAL state (LINEAR)
W_HEAD        = 6.0     # heading alignment toward the goal bearing
W_V           = 2.0     # speed tracking toward the goal-tapered desired speed
R_ACT         = np.array([0.05, 0.20])   # control effort
R_DACT        = np.array([0.10, 0.70])   # control rate (smoothness)
W_OBS         = 8.0     # dynamic-obstacle soft influence ring
W_STATIC_RING = 15.0    # static-surface soft influence ring
RHO_SLACK     = 10.0    # linear exact-penalty weight on a keep-out violation
RHO_SLACK2    = 5.0     # quadratic term



def stage_cost(px, py, th, v, uk, u_km1, *, goal_xy, v_des, t_k,
               obstacles=None, d_infl, d_safe, w_obs, rho, rho2,
               r_act, r_dact, w_goal_run, w_head, w_v,
               d_static=None, d_infl_static=None, d_safe_static=None,
               w_static_ring=None):
    """MPC-reference stage cost, vectorized over rollouts.

    px, py, th, v : (K,) rollout state AFTER this step
    uk, u_km1     : (K, 2) control at this stage and the previous one
    obstacles     : list of (world_x, world_y) arrays already advanced to t_k, or None
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
    cost = cost + w_v * (v - v_des) ** 2

    # Control effort and rate.
    du = uk - u_km1
    cost = cost + (np.asarray(r_act) * uk * uk).sum(axis=1)
    cost = cost + (np.asarray(r_dact) * du * du).sum(axis=1)

    # Dynamic obstacles: soft influence ring + exact keep-out penalty.
    if obstacles:
        for ox, oy in obstacles:
            d2 = (px - ox) ** 2 + (py - oy) ** 2
            d = np.sqrt(d2 + 1e-6)
            cost = cost + w_obs * np.maximum(0.0, d_infl - d) ** 2
            viol = np.maximum(0.0, d_safe * d_safe - d2)
            cost = cost + rho * viol + rho2 * viol * viol

    # Static surfaces (LiDAR OCC), same shape as the obstacle terms.
    if d_static is not None:
        cost = cost + w_static_ring * np.maximum(0.0, d_infl_static - d_static) ** 2
        viols = np.maximum(0.0, d_safe_static ** 2 - d_static ** 2)
        cost = cost + rho * viols + rho2 * viols * viols

    return cost


def terminal_cost(px, py, *, goal_xy, w_goal_term):
    """MPC-reference terminal cost: bounded (linear) pull on the FINAL state."""
    gx, gy = goal_xy
    return w_goal_term * np.sqrt((px - gx) ** 2 + (py - gy) ** 2 + 1.0)
