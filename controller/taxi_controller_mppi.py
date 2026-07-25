"""
taxi_controller.py
==================
External Python controller for the Unity taxiing environment.
Connects via the ML-Agents Python API, runs MPPI + HOCBF-QP, and sends
[a, delta] actions back to Unity each decision step.

Observation contract (must match TaxiAgent.cs CollectObservations, OBS_SIZE=20):

  World/Euclidean frame:
    obs[0]   x_ego    — Unity Z position [m]
    obs[1]   y_ego    — Unity X position [m]
    obs[2]   theta    — heading [rad]
    obs[3]   v        — speed [m/s]
    obs[4..15]        3 × obstacle (dx_global, dy_global, vx, vy) — 12 floats
    obs[16]  goal     — remaining Euclidean distance to goal [m]
    obs[17]  cbf_h    — barrier value h = dist^2 - D^2 of nearest obstacle
    obs[18]  goal_z   — goal world Z position [m] (same axis as obs[0])
    obs[19]  goal_x   — goal world X position [m] (same axis as obs[1])

Action contract:
  act[0]  a_cmd     — acceleration [m/s^2], clipped to [A_MIN, A_MAX]
  act[1]  delta_cmd — steering [rad],       clipped to [-DELTA_LIM, DELTA_LIM]

MPPI runs entirely in the world frame: s0 = [x, y, theta, v, delta, accel].
CBF uses Euclidean obstacle distances in the same frame.
"""

import numpy as np
import argparse
import time
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.exception import UnityCommunicatorStoppedException

from occlusion_capsules import (occlusion_stage_cost, point_segments_min_distance,
                                point_segment_distance_np)
import taxi_cost as tcost
# MPC-reference weights — single definition in taxi_cost, imported by both controllers.
from taxi_cost import (W_GOAL_RUN, W_GOAL_TERM, W_HEAD, W_V, W_OBS,
                       RHO_SLACK, RHO_SLACK2, R_ACT, R_DACT)

# ── Tuning comes from config.yaml ────────────────────────────────────────────
# Every constant below that is a NUMBER YOU MIGHT TUNE is bound from the config;
# the ALL_CAPS names are kept so call sites read exactly as before. Rationale for
# each value lives next to it in the YAML. Derived values (OBS_SIZE), contract
# values (the observation layout) and runtime state (LIDAR_COSTMAP, the CLI flags)
# stay in code. --config swaps the file — see _apply_config_arg() below.
from taxi_config import CFG
_veh, _lim, _goal = CFG["vehicle"], CFG["limits"], CFG["goal"]
_dyn, _keep, _occ = CFG["dynamic_obstacles"], CFG["keepout"], CFG["occlusion"]
_trk, _scan       = CFG["occlusion_tracker"], CFG["scan"]
_mppi             = CFG["mppi"]

# ── Parameters — keep in sync with TaxiAgent.cs inspector fields ─────────────
DT        = _veh["dt"]          # Fixed Timestep in Unity (Project Settings → Time)
L         = _veh["wheelbase"]   # [m]
V_DES     = _veh["v_des"]       # desired taxi speed [m/s]
GOAL_SLOWDOWN_DIST = _goal["slowdown_dist"]
GOAL_MIN_SPEED     = _goal["min_speed"]
W_HALF    = _dyn["w_half"]      # taxiway half-width [m]
D_SAFE    = _dyn["d_safe"]      # keep-out radius [m]
D_INFL    = _dyn["d_infl"]      # MPPI obstacle influence radius [m]
UNC_GROWTH     = _dyn["unc_growth"]
UNC_GROWTH_MAX = _dyn["unc_growth_max"]
D_INFL_PASS    = _dyn["d_infl_pass"]
HEADON_GIVEWAY = _dyn["headon_giveway"]
A_MIN     = _lim["a_min"]
A_MAX     = _lim["a_max"]
DELTA_LIM = _lim["delta_lim"]

# ── Realistic kinematic extensions — must match TaxiAgent.cs inspector values ─
DRAG_COEFF        = _veh["drag_coeff"]
ACCEL_TAU         = _veh["accel_tau"]
MAX_STEER_RATE    = _veh["max_steer_rate"]
STEER_ROLLOFF_SPD = _veh["steer_rolloff_spd"]
STEER_ROLLOFF_MIN = _veh["steer_rolloff_min"]

H_MPPI    = _mppi["horizon"]    # planning horizon (steps)
K_MPPI    = _mppi["samples"]    # rollout samples
LAMBDA    = _mppi["lambda"]     # MPPI temperature
SIG_A     = _mppi["sig_a"]      # noise std for acceleration samples
SIG_D     = _mppi["sig_d"]      # noise std for steering samples

# MPPI stage costs
W_LAT, W_HEAD, W_V, W_CTRL = (_mppi["w_lat"], _mppi["w_head"],
                              _mppi["w_v"], _mppi["w_ctrl"])
# Action cost   ℓact = ||u||²_R + ||Δu||²_RΔ,  u = [a, delta], Δu_k = u_k - u_{k-1}.
#   R  (R_ACT)  — diagonal effort weight: penalises large commands (energy/effort).
#   RΔ (R_DACT) — diagonal rate weight: penalises command CHANGE (smoothness, no jerk).
# Stored as the diagonals of the R / RΔ matrices, per channel [a, delta].
# R_ACT / R_DACT now come from taxi_cost (MPC reference) — see the import below.
# W_OBS lowered and W_PROG raised (was 12.0 / 0.4) so forward progress competes with the obstacle
# penalty. On a head-on pass the ego can't keep D_SAFE either way (the oncoming agent comes to it),
# so an over-weighted W_OBS made "stop and wait" the cheapest option; a stronger progress pull makes
# driving through win. Trade-off: the ego is slightly less conservative around crossers/convergers.
W_OFF, W_PROG               = _mppi["w_off"], _mppi["w_prog"]   # (W_OBS from taxi_cost)
# Goal-approach reward (paper's ℓgoal). Per stage k:  ℓgoal = -C_GOAL * max(0, d0 - d_k), with
# d0 = Euclidean distance to the goal at the rollout start and d_k = ||p_k - p_goal|| the true
# Euclidean distance to the goal at stage k. A heavier TERMINAL copy at the last stage H-1
# (ℓgoal,H-1 = -C_GOAL_TERM * max(0, d0 - d_{H-1})) weights the END position of the rollout, so
# a rollout is allowed to detour around an obstacle as long as it ENDS closer to the goal — this
# is what prevents greedy stop-short behaviour. Paper values: C_GOAL 5.0 (goal in line-of-sight)
# or 0.125 (goal occluded → encourage exploration); C_GOAL_TERM 10.0.
C_GOAL                      = _mppi["c_goal"]
C_GOAL_TERM                 = _mppi["c_goal_term"]
C_PROGRESS                  = _mppi["c_progress"]

# ── Occlusion-aware safety: forward-reachable-set keep-out + RSS sightline speed cap ──
# SHARED with taxi_controller_mpc.py (the MPC imports these), so both controllers use the SAME
# occlusion model and constants. A worst-case hidden agent sits on a discrete blind-corner
# boundary point (the range-jump corner from ObstacleCircles.occlusion_segments — same set the MPC
# constrains, filtered to within OCC_QUERY_R of the ego), and
# could be anywhere within V_TARGET·t_k of it after t_k = (k+1)·DT. Two coupled terms, gated on
# --occlusion-aware:
#   * expanding keep-out circle:  r_keep = D_SAFE_HARD + V_TARGET·t_k, penalised with the
#     hard-constraint weight W_HARD — the ego swings wide around the blind corner;
#   * RSS sightline speed cap:     v_safe = sqrt(2·A_BRAKE_SIGHT·d_vis), floored at V_SIGHT_FLOOR —
#     the ego slows so it can always stop before the nearest occlusion.
# Keep V_TARGET realistic — too high and the bubble swallows the horizon and the ego freezes.
V_TARGET       = _occ["v_target"]
OCC_USE_CAPSULES = _occ["use_capsules"]   # False ⇒ collapse each boundary to its corner
OCC_QUERY_R    = _occ["query_r"]
OCC_HORIZON    = _occ["horizon"]
K_OCC          = _occ["k_occ"]
W_SIGHT        = _occ["w_sight"]
A_BRAKE_SIGHT  = _occ["a_brake_sight"]
V_SIGHT_FLOOR  = _occ["v_sight_floor"]

N_SCEN = _mppi["n_scen"]
W_INFO = _mppi["w_info"]
INFO_RANGE = _mppi["info_range"]

# Lister obstacles cost
LIDAR_COSTMAP  = None    # ObstacleCircles instance when --lidar-costmap is set (name kept for
                         # brevity across the many call sites; it is the circle-cover model now)
STATIC_AVOID   = False   # set True by --lidar-costmap: adds the static keep-out/soft-ring cost
                         # to MPPI, using clearance to the nearest obstacle-circle surface.
# ── ONE hard keep-out, shared by static surfaces AND occlusion boundaries ─────
# Both hazards are things the ego must simply not enter, so they get the same radius and
# the same weight; the only difference is that the occlusion radius EXPANDS along the
# horizon by V_TARGET·t_k (the hidden agent's forward reachable set) while the static one
# is fixed. W_HARD stands in for an infinite (hard-constraint) weight: it is large enough
# that any breach dominates the goal pull, so the optimiser will not trade clearance for
# progress. There is no soft influence ring on either — see taxi_cost.stage_cost and
# occlusion_capsules.occlusion_stage_cost, which now share the same one-sided quadratic.
D_SAFE_HARD    = _keep["d_safe_hard"]
W_HARD         = _keep["w_hard"]

# Range-jump occlusion boundaries (shared with taxi_controller_mpc.py). The scan geometry
# MUST match the Unity PointCloudPublisher Inspector fields; a mismatch is detected and the
# scan ignored. Corners are tracked across scans so the same physical corner keeps ONE
# stable centre instead of a fresh jittered one every detection.
SCAN_FOV_H, SCAN_FOV_V = _scan["fov_h"], _scan["fov_v"]
SCAN_RES_H, SCAN_RES_V = _scan["res_h"], _scan["res_v"]
SCAN_MAX_RANGE         = _scan["max_range"]
OCC_FWD_HALF_ANGLE     = _occ["fwd_half_angle"]
OCC_TRACK_ASSOC = _trk["assoc_radius"]
OCC_TRACK_ALPHA = _trk["alpha"]
OCC_TRACK_TTL   = _trk["ttl"]
OCC_TRACK_HITS  = _trk["min_hits"]
OCC_TRACKER     = None   # OcclusionCornerTracker, created in run()
OCC_SEGS_NOW    = None   # (M,2,2) tracked boundaries for THIS control step, set by run()
                         # once per step and consumed by mppi() — mirrors how the MPC
                         # computes occ_segs in its run loop and passes them to solve().

OCCLUSION_AWARE = False   # set True by --occlusion-aware (needs --lidar-costmap): enables the
                          # FRS keep-out + sightline speed cap above (mirrors the MPC controller).
VISIBILITY_COST = False   # vestigial: --visibility-cost is a documented no-op

DETECTION_RANGE = float("inf")   # default: oracle (off). Set via --detect-range.

# Number of obstacles packed in the observation vector (must match TaxiAgent K_OBS)
K_OBS    = _dyn["k_obs"]
OBS_SIZE = 4 + K_OBS * 4 + 2 + 2   # 20: ego(4) + K_OBS*4 + goal(1) + cbf_h(1) + tangent(2)

rng = np.random.default_rng(42)



# ── Observation unpacking ────────────────────────────────────────────────────

def obs_to_state(obs: np.ndarray, prev_delta: float = 0.0, prev_accel: float = 0.0):
    """
    Unpack the Unity 20-D observation vector into world/Euclidean coordinates.

    Returns
    -------
    s          : np.ndarray shape (6,) — [x_fwd, y_lat, theta, v, delta_actual, accel_actual]
    obstacles  : list of (rel_xy, obs_v), rel_xy ego-relative (dx, dy) in world coords.
    goal_xy    : np.ndarray shape (2,) — goal world position (z, x), same axes as ego (obs[18:20]).

    A zero-padded obstacle slot has dy == 999; those are skipped.
    """
    ego0    = float(obs[0])   # x_ego (world)
    ego1    = float(obs[1])   # y_ego (world)
    ego2    = float(obs[2])   # theta (world)
    v       = float(obs[3])
    goal_xy = np.array([float(obs[18]), float(obs[19])])   # goal world (z, x)

    s = np.array([ego0, ego1, ego2, v, prev_delta, prev_accel])

    obstacles = []
    for i in range(K_OBS):
        base   = 4 + i * 4
        dx     = float(obs[base + 0])
        dy     = float(obs[base + 1])
        vx_obs = float(obs[base + 2])
        vy_obs = float(obs[base + 3])

        if abs(dy) > 900:   # zero-padded sentinel
            continue

        dist = np.hypot(dx, dy)
        if dist > DETECTION_RANGE:  # outside sensor range — not observable
            continue

        # rel_xy: obstacle position relative to ego, regardless of frame
        rel_xy = np.array([dx, dy])
        obs_v  = np.array([vx_obs, vy_obs])
        obstacles.append((rel_xy, obs_v))

    return s, obstacles, goal_xy


def inject_sensor_noise(obs: np.ndarray, noise_std: float, rng_local) -> np.ndarray:
    """
    Add Gaussian noise to the obstacle observation slots [4..15].
    Ego state [0..3], goal/cbf [16..17], and goal position [18..19] are not corrupted.
    """
    if noise_std <= 0.0:
        return obs
    noisy = obs.copy()
    noisy[4:16] += rng_local.normal(0.0, noise_std, 12).astype(obs.dtype)
    return noisy


# ── MPPI ─────────────────────────────────────────────────────────────────────

def _rollout_step(st, a_cmd, delta_cmd):
    """
    One-step realistic bicycle dynamics for MPPI rollouts.
    st columns: [x, y, theta, v, delta_actual, accel_actual]
    Mirrors TaxiAgent.ApplyBicycleDynamics exactly.
    """
    v            = st[:, 3]
    delta_actual = st[:, 4]
    accel_actual = st[:, 5]

    # 1+2: rate-limited, speed-dependent steering
    speed_frac    = np.clip(v / max(STEER_ROLLOFF_SPD, 1e-3), 0., 1.)
    eff_limit     = DELTA_LIM * (1. - speed_frac * (1. - STEER_ROLLOFF_MIN))
    delta_target  = np.clip(delta_cmd, -eff_limit, eff_limit)
    max_delta_step = MAX_STEER_RATE * DT
    delta_new     = delta_actual + np.clip(delta_target - delta_actual,
                                           -max_delta_step, max_delta_step)

    # 3: first-order acceleration lag
    a_clamped  = np.clip(a_cmd, A_MIN, A_MAX)
    if ACCEL_TAU > 1e-3:
        accel_new = accel_actual + (a_clamped - accel_actual) * (DT / ACCEL_TAU)
    else:
        accel_new = a_clamped

    # 4: drag + speed integration
    drag  = DRAG_COEFF * v
    v_new = np.maximum(0., v + (accel_new - drag) * DT)

    # Bicycle geometry
    dtheta = v_new / L * np.tan(delta_new) * DT
    x_new  = st[:, 0] + v_new * np.cos(st[:, 2]) * DT
    y_new  = st[:, 1] + v_new * np.sin(st[:, 2]) * DT
    th_new = st[:, 2] + dtheta

    st_new = np.stack([x_new, y_new, th_new, v_new, delta_new, accel_new], axis=1)
    return st_new


def mppi(s0, mean, obstacles, goal_xy, u_prev=None):
    """
    Sample K_MPPI rollouts with realistic 6D state dynamics.

    s0 = [x, y, theta_global, v, delta, accel]. Rollout in world frame.
    goal_xy = goal world position (x_fwd, y_lat), same axes as the rollout state.
    u_prev  = last APPLIED control [a, delta], used for the Δu_0 = u_0 - u_prev term
              (defaults to zero if None).
    Lane cost penalises lateral position y; obstacles in world coords.

    Returns (u_nom, new_mean).
    """
    # Nominal MPPI weights / limits. The frontal-threat "go-around" gate (in_corridor / aimed_at_us)
    # has been removed to keep the planner simple: plain lane-following + obstacle rings, and the
    # LiDAR visibility term is what drives any deliberate off-lane motion.
    sig_d_eff  = SIG_D
    w_off_eff  = W_OFF
    w_lat_eff  = W_LAT
    a_min_eff  = A_MIN
    lane_bias  = 0.0
    d_infl_arr = np.full(len(obstacles), D_INFL)   # per-obstacle influence ring
    d_safe_arr = np.full(len(obstacles), D_SAFE)   # per-obstacle hard keep-out

    noise = rng.normal(0, [SIG_A, sig_d_eff], (K_MPPI, H_MPPI, 2))
    na    = mean + noise
    na[:, :, 0] = np.clip(na[:, :, 0], a_min_eff, A_MAX)
    na[:, :, 1] = np.clip(na[:, :, 1], -DELTA_LIM, DELTA_LIM)

    # Previous applied control for the Δu_0 term (u_0 - u_prev). Zero if not supplied.
    u_prev = np.zeros(2) if u_prev is None else np.asarray(u_prev, dtype=float)

    cost = np.zeros(K_MPPI)
    st   = np.tile(s0, (K_MPPI, 1)).astype(float)

    # Taper the speed TARGET down as the goal gets close (see GOAL_SLOWDOWN_DIST doc comment) so
    # the aircraft's turn radius shrinks enough to actually hit the small arrival capture radius
    # instead of sweeping past it at cruise speed.
    s0_fwd = s0[0]
    s0_lat = s0[1]

    # Goal world position (sent by Unity, obs[18:20]) and the initial Euclidean distance to it.
    goal_fwd, goal_lat = float(goal_xy[0]), float(goal_xy[1])
    d0 = np.hypot(goal_fwd - s0_fwd, goal_lat - s0_lat)

    slow_frac = np.clip(d0 / GOAL_SLOWDOWN_DIST, 0.0, 1.0)
    v_des_eff = GOAL_MIN_SPEED + (V_DES - GOAL_MIN_SPEED) * slow_frac

    prev_fwd = np.full(K_MPPI, s0_fwd)   # ego position at the previous step (p_k), for ℓprogress
    prev_lat = np.full(K_MPPI, s0_lat)

    s0_theta = float(s0[2])              # initial heading — the straight-line lock reference

    # Occlusion set for this solve: the DISCRETE blind-corner boundary points (like the MPC's
    # P_occ), taken as the NEAR endpoints (corners) of the range-jump boundary segments. Filter
    # to within OCC_QUERY_R of the ego (== the MPC) and hold them fixed over the horizon; d_occ
    # per rollout pose = distance to the nearest.
    occ_segs = None
    if OCCLUSION_AWARE and LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
        # Range-jump boundary SEGMENTS, tracked across scans (same source as the MPC).
        # The keep-out is the segment dilated by r_keep — a CAPSULE (a disc at each
        # endpoint plus the rectangle between them), so the hidden agent is assumed to
        # lurk anywhere along the sightline, not only at the corner. Selection and
        # ordering still key on the CORNER (segs[:, 0]): that is the point the phantom
        # is anchored to, and it is what the MPC's _pack_params filters on too.
        _seg = OCC_SEGS_NOW
        _seg = np.asarray(_seg, float).reshape(-1, 2, 2) if (
            _seg is not None and len(_seg)) else None
        if _seg is not None and len(_seg):
            if not OCC_USE_CAPSULES:
                # Collapse each boundary onto its corner ⇒ a degenerate segment, whose
                # capsule is exactly the old expanding CIRCLE. Same switch, same place
                # in the pipeline as the MPC's.
                _seg = _seg.copy()
                _seg[:, 1, :] = _seg[:, 0, :]
            _op = _seg[:, 0, :]
            # Same selection the MPC's _pack_params does: drop boundaries sitting within
            # the fully-expanded keep-out of the GOAL (the goal is known-safe, so a phantom
            # must not be assumed to hide on it), then take the K_OCC nearest in range.
            if goal_xy is not None:
                _rgc = D_SAFE_HARD + V_TARGET * H_MPPI * DT
                _gd = np.hypot(_op[:, 0] - goal_xy[0], _op[:, 1] - goal_xy[1])
                _seg = _seg[_gd > _rgc]
            if len(_seg):
                # Range-gate on true point-to-CAPSULE-axis distance, not on the corner:
                # a long boundary whose corner sits beyond OCC_QUERY_R can still have its
                # far end sweeping past the ego, and dropping it would silently un-see it.
                _ego = np.array([[s0_fwd, s0_lat]])
                _d = np.array([point_segment_distance_np(_ego, s[0], s[1])[0]
                               for s in _seg])
                _in = _d < OCC_QUERY_R
                if _in.any():
                    _near = _seg[_in]
                    occ_segs = _near[np.argsort(_d[_in])[:K_OCC]]

    def _dist_to_occ(px, py):
        """(K,) distance from each rollout pose to the nearest occlusion CAPSULE axis [m].

        min over boundaries of min(disc at corner, disc at far end, rectangle between)
        — see occlusion_capsules.point_segments_min_distance for why that is one
        point-to-segment evaluation rather than three.
        """
        return point_segments_min_distance(px, py, occ_segs)

    for k in range(H_MPPI):
        st = _rollout_step(st, na[:, k, 0], na[:, k, 1])
        fwd, lat, th, vv = st[:, 0], st[:, 1], st[:, 2], st[:, 3]

        u_k   = na[:, k, :]                                  # (K, 2)
        u_km1 = u_prev[None, :] if k == 0 else na[:, k - 1, :]

        # MPC-REFERENCE stage cost (taxi_cost.stage_cost): bounded linear goal pull,
        # heading alignment, speed tracking, control effort + rate, dynamic-obstacle and
        # static keep-outs. This REPLACES MPPI's former closure/progress rewards and the
        # exp-weighted velocity term: those applied ~400x more goal pressure than the MPC,
        # so the identical occlusion penalty was far weaker in relative terms and the ego
        # cut corners the MPC would not. Obstacles are advanced by a constant-velocity ray
        # to t_k, exactly as the MPC's P_opos + P_ovel*t_k does.
        _obs_world = [(s0_fwd + rel[0] + ov[0] * ((k + 1) * DT),
                       s0_lat + rel[1] + ov[1] * ((k + 1) * DT))
                      for rel, ov in obstacles]
        _d_static = None
        if STATIC_AVOID and LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
            # ABSOLUTE world (a0,a1): the circles live in world coords and distance()
            # expects absolute queries (see the [CHK] debug and stage_cost's obstacles/
            # goal, which are all absolute too). Passing ego-relative deltas here made the
            # clearance huge, so the static keep-out never fired (ego drove through walls).
            _d_static = LIDAR_COSTMAP.distance(fwd, lat)
        cost += tcost.stage_cost(
            fwd, lat, th, vv, u_k, u_km1,
            goal_xy=(goal_fwd, goal_lat), v_des=v_des_eff, t_k=(k + 1) * DT,
            obstacles=_obs_world, d_infl=D_INFL, d_safe=D_SAFE, w_obs=W_OBS,
            rho=RHO_SLACK, rho2=RHO_SLACK2, r_act=R_ACT, r_dact=R_DACT,
            w_goal_run=W_GOAL_RUN, w_head=W_HEAD, w_v=W_V,
            d_static=_d_static, d_safe_static=D_SAFE_HARD, w_static=W_HARD)

        # ── Occlusion-aware safety: FRS keep-out + RSS sightline cap ──────────
        # Canonical formula shared with the MPC (occlusion_capsules.occlusion_stage_cost).
        if occ_segs is not None:
            cost += occlusion_stage_cost(
                _dist_to_occ(fwd, lat), vv, (k + 1) * DT,
                V_TARGET, D_SAFE_HARD, W_HARD)

        # (The forward-reachable-set keep-out now lives in the unified occlusion-aware block
        # above, gated on OCCLUSION_AWARE with the shared constants — see the MPC for parity.)

        # Obstacles — world frame in both modes. rel_xy is ego-relative (world Z, X).
        # t_elapsed = (k + 1) * DT
        # if scen is not None:
        #     # Scenario-based prediction: each obstacle has N_SCEN sampled futures.
        #     for oi, ((rel_xy, _ov), (ovx, ovy), ent_b) in enumerate(zip(obstacles, scen, ent)):
        #         di, ds = d_infl_arr[oi], d_safe_arr[oi]
        #         oz = s0_fwd + rel_xy[0] + ovx * t_elapsed        # (N_SCEN,)
        #         ox = s0_lat + rel_xy[1] + ovy * t_elapsed        # (N_SCEN,)
        #         d_obs = np.hypot(fwd[:, None] - oz[None, :],
        #                          lat[:, None] - ox[None, :])      # (K, N_SCEN)
        #         cost_obs_scen += np.where(d_obs < di, W_OBS * (di - d_obs)**2, 0.)
        #         cost_obs_scen += np.where(d_obs < ds, BIG, 0.)
        # else:
        #     # Deterministic (single constant-velocity ray) obstacle cost: soft influence ring +
        #     # hard keep-out, per obstacle.
        #     # hard keep-out, per obstacle.
        #     for oi, (rel_xy, obs_v) in enumerate(obstacles):
        #         di, ds = d_infl_arr[oi], d_safe_arr[oi]
        #         obs_z = s0_fwd + rel_xy[0] + obs_v[0] * t_elapsed
        #         obs_x = s0_lat + rel_xy[1] + obs_v[1] * t_elapsed
        #         d_obs = np.hypot(fwd - obs_z, lat - obs_x)
        #         cost += np.where(d_obs < di, W_OBS * (di - d_obs)**2, 0.)
        #         cost += np.where(d_obs < ds, BIG, 0.)

    # Terminal goal cost ℓgoal,H-1 = -C_GOAL_TERM * max(0, d0 - d_{H-1}): a heavier copy of the
    # goal reward evaluated at the FINAL rollout state, so a rollout that ENDS closer to the goal
    # wins even if it detoured en route (lets the planner go around obstacles instead of stopping
    # short). st is the final rollout state here.
    # MPC-reference terminal cost: bounded (linear) pull on the FINAL state.
    cost += tcost.terminal_cost(st[:, 0], st[:, 1],
                                goal_xy=(goal_fwd, goal_lat), w_goal_term=W_GOAL_TERM)

    # Leave -cost.min() to avoid underflow, softmax to compute the weights
    w   = np.exp(-(cost - cost.min()) / LAMBDA)
    w  /= w.sum()
    opt = (w[:, None, None] * na).sum(axis=0)

    u_nom    = opt[0].copy()
    new_mean = np.vstack([opt[1:], opt[-1]])
    return u_nom, new_mean

# ── System identification ─────────────────────────────────────────────────────

def identify_bicycle_model(env, behavior_name, n_steps=200):
    print("[SysID] Probing bicycle dynamics...")
    decision_steps, _ = env.get_steps(behavior_name)
    speeds = []

    for _ in range(n_steps):
        n = len(decision_steps)
        if n == 0:
            env.step()
            decision_steps, _ = env.get_steps(behavior_name)
            continue

        action = ActionTuple(
            continuous=np.tile([A_MAX * 0.5, 0.0], (n, 1)).astype(np.float32)
        )
        env.set_actions(behavior_name, action)
        env.step()
        decision_steps, _ = env.get_steps(behavior_name)

        if len(decision_steps) > 0:
            speeds.append(float(decision_steps.obs[0][0][3]))

    if len(speeds) > 2:
        empirical = (speeds[-1] - speeds[0]) / (len(speeds) * DT)
        print(f"[SysID] Commanded a={A_MAX*0.5:.2f} m/s^2, "
              f"measured accel≈{empirical:.3f} m/s^2")
        if abs(empirical - A_MAX * 0.5) > 0.2 * A_MAX * 0.5:
            print("[SysID] WARNING: >20% mismatch — check Fixed Timestep or "
                  "ApplyBicycleDynamics in TaxiAgent.cs")
    else:
        print("[SysID] Not enough data — is the Unity environment running?")


# ── Main control loop ─────────────────────────────────────────────────────────

def _save_trajectory(out_dir, verdict, goal_xy, traj, obs_track=None, occ_pts=None,
                     occ_segments=None, capsule_horizon=None,
                     max_capsules=12, single_radius=False, show_expansion=True,
                     occ_ego=None, show_occlusion=True):
    """Persist the run's ego trajectory as CSV + a top-down PNG plot.

    verdict: short outcome tag ("reached" / "collision" / "timeout") used in the
    output filename and plot title.
    traj columns: t, x, y, theta, v, a_cmd, delta_cmd  (x=Unity Z, y=Unity X).
    obs_track (optional): (N, 3) array of per-step dynamic-obstacle world
    positions (t, x, y), same frame as traj — overlaid on the plot.
    occ_pts (optional): (M, 2) array of static occluder (OCC) cell centres
    (x, y) from the LiDAR costmap, same frame — the buildings/walls.
    occ_segments (optional): (S, 2, 2) occlusion BOUNDARY segments from the
    range-jump detector, [[near, far], ...] in the same frame. Each is drawn as
    the nested forward-reachable-set CAPSULES the MPC constrained against, at
    radius D_SAFE_HARD + V_TARGET·t for t across the horizon.
    capsule_horizon (optional): horizon length [s] the keep-out expanded over
    (N·DT). Defaults to 3 s if not given.
    single_radius: True ⇒ Algorithm 1 mode, the ENFORCED radius is the fixed
    D_SAFE_HARD + V_TARGET·capsule_horizon (no per-stage growth).
    show_occlusion: False ⇒ draw none of the occlusion artifacts (keep-out circles,
    expansion rings, corner centres, ego-at-detection markers and their connectors),
    leaving just the trajectory, static occluders, obstacles and goal. The second
    (horizon-time) colorbar is suppressed too.
    occ_ego (optional): (S, 3) array of (ego_x, ego_y, t_episode) recorded when each
    occlusion segment was FIRST constrained — used to tie each keep-out back to the
    point on the trajectory where it applied.
    show_expansion: also draw the intermediate reachable sets at t < horizon,
    showing how the worst-case hidden agent's reachable disc grows. In single-circle
    mode these are illustrative only — the MPC enforces the outermost radius at every
    stage — so they are drawn thin and never as the enforced boundary.
    max_capsules: cap on how many segments get capsules drawn — an episode
    accumulates far more boundaries than are legible at once, so the ones
    nearest the ego path are kept.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"traj_{verdict}")

    header = "t,x,y,theta,v,a_cmd,delta_cmd"
    np.savetxt(f"{stem}.csv", traj, delimiter=",", header=header, comments="")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[traj] saved {stem}.csv (matplotlib missing — no plot)")
        return

    x, y, v = traj[:, 1], traj[:, 2], traj[:, 4]
    fig, ax = plt.subplots(figsize=(14, 14))
    if occ_pts is not None and len(occ_pts):
        ax.scatter(occ_pts[:, 0], occ_pts[:, 1], c="0.35", s=8, marker="s",
                   alpha=0.5, label="static occluders", zorder=1)
    # Expanding forward-reachable-set capsules around each occlusion boundary segment.
    # Drawn first (lowest zorder) so the ego path stays readable on top of them.
    if show_occlusion and occ_segments is not None and len(occ_segments):
        from occlusion_capsules import capsule_polygon, point_segment_distance_np
        segs = np.asarray(occ_segments, float).reshape(-1, 2, 2)
        occ_ego = None if occ_ego is None else np.asarray(occ_ego, float).reshape(-1, 3)
        # Rings are coloured by HORIZON time t_k, on a scale distinct from the speed
        # colormap so the two readings never get confused.
        horizon_cmap = plt.get_cmap("autumn_r")

        # An episode accumulates far more boundaries than are legible; keep those
        # nearest the ego path, which are the ones that actually shaped the trajectory.
        if len(segs) > max_capsules:
            path = np.column_stack((x, y))
            dmin = np.array([point_segment_distance_np(path, s[0], s[1]).min()
                             for s in segs])
            keep = np.argsort(dmin)[:max_capsules]
            segs = segs[keep]
            if occ_ego is not None and len(occ_ego) >= len(dmin):
                occ_ego = occ_ego[keep]

        T = float(capsule_horizon) if capsule_horizon else 3.0
        for si, seg in enumerate(segs):
            first = (si == 0)

            # Intermediate reachable sets along the horizon, COLOURED BY HORIZON TIME t_k.
            # When the radius grows per stage (the paper's Fig. 6) each ring is the
            # constraint enforced at its own timestep — the ego is checked against the set
            # as it stands at that future instant. In single_radius mode they are
            # illustrative, because the outermost radius is applied at every stage.
            n_rings = 6
            for ti, t in enumerate(np.linspace(0.0, T, n_rings)):
                if not show_expansion and t < T:
                    continue
                poly = capsule_polygon(seg[0], seg[1], D_SAFE_HARD + V_TARGET * t)
                ax.plot(poly[:, 0], poly[:, 1], "-",
                        color=horizon_cmap(t / T if T else 0.0),
                        lw=1.4 if ti == n_rings - 1 else 0.8,
                        alpha=0.95 if ti == n_rings - 1 else 0.6,
                        zorder=0)

            # Circle centre = the detected occlusion corner.
            ax.plot(seg[0, 0], seg[0, 1], ".", color="k", ms=5, zorder=2,
                    label="occlusion corner (centre)" if first else None)

            # Tie the keep-out to WHERE the ego was when this boundary was constrained:
            # a hollow marker on the path plus a hairline to the corner. Without this the
            # episode-wide union gives no clue which part of the run each circle came from.
            if occ_ego is not None and si < len(occ_ego):
                ex, ey = float(occ_ego[si][0]), float(occ_ego[si][1])
                ax.plot([ex, seg[0, 0]], [ey, seg[0, 1]], "-", color="0.35",
                        lw=0.7, alpha=0.55, zorder=2)
                ax.plot(ex, ey, "o", mfc="none", mec="k", ms=7, mew=1.2, zorder=3,
                        label="ego when boundary first constrained" if first else None)

    sc = ax.scatter(x, y, c=v, cmap="viridis", s=10, zorder=3)
    ax.plot(x, y, "-", color="0.6", lw=0.8, zorder=2)
    ax.plot(x[0], y[0], "o", color="tab:green", ms=10, label="start", zorder=4)
    ax.plot(x[-1], y[-1], "s", color="tab:red", ms=10, label="end", zorder=4)
    if obs_track is not None and len(obs_track):
        ax.scatter(obs_track[:, 1], obs_track[:, 2], c="tab:orange", s=18, alpha=0.6,
                   marker="x", linewidths=1.2, label="dynamic obstacles", zorder=6)
    if goal_xy is not None:
        ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=18,
                markeredgecolor="k", label="goal", zorder=5)
    fig.colorbar(sc, ax=ax, label="speed [m/s]")
    # Second colorbar for the keep-out rings: which future instant each one constrains.
    if show_occlusion and occ_segments is not None and len(occ_segments):
        import matplotlib as _mpl
        T_cb = float(capsule_horizon) if capsule_horizon else 3.0
        sm = _mpl.cm.ScalarMappable(cmap=horizon_cmap,
                                    norm=_mpl.colors.Normalize(vmin=0.0, vmax=T_cb))
        fig.colorbar(sm, ax=ax, label="keep-out horizon time $t_k$ [s]")
    # Frame the view on the EGO trajectory (not the full scatter of occluders, which
    # can sit hundreds of metres away and would shrink the path to a dot). Square,
    # equal-scale box centred on the path extent with a fixed margin; far occluders
    # simply clip out.
    TRAJ_MARGIN = 20.0                     # [m] padding around the ego path
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    half = 0.5 * max(x.max() - x.min(), y.max() - y.min()) + TRAJ_MARGIN
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x  (Unity Z) [m]"); ax.set_ylabel("y  (Unity X) [m]")
    ax.set_title(f"trajectory — {verdict}")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=120)
    plt.close(fig)
    print(f"[traj] saved {stem}.csv and {stem}.png")


def run(unity_exec_path=None, port=5004, run_sysid=True,
        noise_std=0.0, detect_range=float("inf"),
        uncertainty=False, n_scenarios=N_SCEN, w_info=W_INFO,
        d_infl=D_INFL, d_safe=D_SAFE, info_range=INFO_RANGE,
        lidar_costmap=False, lidar_topic="/point_cloud", visibility_cost=False,
        occlusion_aware=False, save_traj=None, show_occlusion_plot=True):
    global DETECTION_RANGE, UNCERTAINTY, N_SCEN, W_INFO
    global D_INFL, D_SAFE, INFO_RANGE, LIDAR_COSTMAP, VISIBILITY_COST, STATIC_AVOID
    global OCCLUSION_AWARE, OCC_TRACKER, OCC_SEGS_NOW
    DETECTION_RANGE = detect_range
    UNCERTAINTY     = uncertainty
    N_SCEN          = n_scenarios
    W_INFO          = w_info
    D_INFL          = d_infl
    D_SAFE          = d_safe
    INFO_RANGE      = info_range
    if d_infl != 16.0 or d_safe != 14.0:
        print(f"[Controller] Planning radii : D_INFL={d_infl:.1f} m (plan-start), "
              f"D_SAFE={d_safe:.1f} m (keep-out)")
    print(f"[Controller] Connecting to Unity on port {port} ...")
    if detect_range < float("inf"):
        print(f"[Controller] Detection range : {detect_range:.1f} m  (obstacles beyond masked)")
    if visibility_cost:
        # The active-perception visibility term was a product of the removed persistent
        # occupancy grid; the circle model has no accumulated UNKNOWN volume to score.
        print("[Controller] --visibility-cost is no longer supported (needs the removed "
              "persistent grid) — IGNORED.")
    VISIBILITY_COST = False
    if lidar_costmap:
        STATIC_AVOID    = lidar_costmap
        OCCLUSION_AWARE = occlusion_aware and lidar_costmap   # occlusion terms need the map
        from obstacle_circles import ObstacleCircles
        from occlusion_capsules import OcclusionCornerTracker, point_segment_distance_np
        # max_age raised from the 0.5s default: Unity's PointCloudPublisher is configured well
        # below 10Hz (measured ~1Hz cloud_age via [DEBUG lidar]), so 0.5s made `ready` permanently
        # False and the static-avoidance/collision cost never activated. 1.5s covers ~1Hz publish
        # with margin; lower it again if the Unity publish rate is raised instead.
        cm = ObstacleCircles(max_age=1.5)
        if cm.start(topic=lidar_topic):
            LIDAR_COSTMAP = cm
            if OCCLUSION_AWARE:
                # Enable the range-jump segment path and give corners temporal identity.
                cm.configure_scan(SCAN_FOV_H, SCAN_FOV_V, SCAN_RES_H, SCAN_RES_V,
                                  SCAN_MAX_RANGE)
                OCC_TRACKER = OcclusionCornerTracker(
                    assoc_radius=OCC_TRACK_ASSOC, alpha=OCC_TRACK_ALPHA,
                    ttl=OCC_TRACK_TTL, min_hits=OCC_TRACK_HITS)
            feats = [f"static(D_SAFE={D_SAFE_HARD:.1f}m, circle-cover)"]
            if OCCLUSION_AWARE:  feats.append(f"occlusion(D_SAFE={D_SAFE_HARD:.1f}m, "
                                             f"v_target={V_TARGET:.1f}m/s, sightline)")
            print(f"[Controller] LiDAR map     : ON  (topic={lidar_topic}) — {' + '.join(feats)} "
                  f"in the MPPI cost; per-scan obstacle circles")
        else:
            STATIC_AVOID    = False
            OCCLUSION_AWARE = False
            print("[Controller] LiDAR map     : requested but unavailable (no rclpy) — "
                  "running WITHOUT LiDAR-based costs")
    elif occlusion_aware:
        print("[Controller] Occlusion-aware : requested but needs --lidar-costmap — DISABLED")
    env = UnityEnvironment(
        file_name=unity_exec_path,
        base_port=port,
        seed=42,
        no_graphics=unity_exec_path is not None,  # headless when running a build
    )
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec          = env.behavior_specs[behavior_name]
    print(f"[Controller] Behavior  : {behavior_name}")
    print(f"[Controller] Obs shape : {spec.observation_specs[0].shape}")
    print(f"[Controller] Act size  : {spec.action_spec.continuous_size}")

    obs_size = spec.observation_specs[0].shape[0]
    act_size = spec.action_spec.continuous_size
    assert obs_size == OBS_SIZE, (
        f"Expected {OBS_SIZE} observations, got {obs_size}. "
        f"Set BehaviorParameters Vector Observations = {OBS_SIZE} in Unity Inspector "
        f"and ensure TaxiAgent.cs K_OBS = {K_OBS}."
    )
    assert act_size == 2, (
        f"Expected 2 continuous actions, got {act_size}."
    )

    if run_sysid:
        identify_bicycle_model(env, behavior_name)
        env.reset()

    # Unity (Editor or build) can exit mid-run — most often the Editor drops Play mode because
    # a script recompiled, or the window was closed. That surfaces as
    # UnityCommunicatorStoppedException from env.step(); catch it so the run ends cleanly (and
    # still writes its trajectory) instead of dying with a traceback.
    unity_stopped = False
    mean      = np.zeros((H_MPPI, 2))
    min_h     = np.inf
    min_dist  = np.inf
    collided  = False
    reached   = False
    traj      = []   # per-step ego pose: (t, x, y, theta, v, a_cmd, delta_cmd)
    obs_track = []   # per-step obstacle world positions: list of (t, x, y) rows
    # Union across the run (deduped by rounded world cell) of the static occluders and
    # occlusion boundaries the perception produced. Perception is frame-based (no memory),
    # so a single end-of-run snapshot only shows what is visible at the goal — accumulate to
    # show the full extent the ego actually reacted to.
    occ_seen  = {}   # static occluder points  → plotted as "static occluders"
    occ_track = {}   # tracked occlusion boundaries, keyed on stable track id

    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    ep_steps     = 0
    delta_actual = 0.0   # tracked Python-side to feed into MPPI state
    accel_actual = 0.0
    u_prev       = np.zeros(2)   # last applied [a, delta], for the MPPI Δu smoothness cost
    episode_done = False

    while not episode_done:
        # ── Terminal step (episode just ended) ──────────────────────────────
        if len(terminal_steps) > 0:
            collided = min_h < 0.
            reached  = not terminal_steps.interrupted[0] and not collided
            episode_done = True
            break

        if len(decision_steps) == 0:
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            continue

        obs   = decision_steps.obs[0][0]          # shape (OBS_SIZE,)
        ep_steps += 1

        obs_n = inject_sensor_noise(obs, noise_std, rng)
        s, obstacles, goal_xy = obs_to_state(obs_n, delta_actual, accel_actual)

        # Rebuild the LiDAR map from the latest cloud+pose, scrubbing returns near
        # the known dynamic agents (the oracle handles those). ego_fwd (world a0,a1)
        # orients the visibility ROI wedge — same axes as the mppi rollout frame.
        if LIDAR_COSTMAP is not None:
            ego_fwd = np.array([np.cos(s[2]), np.sin(s[2])])
            LIDAR_COSTMAP.update(ego_fwd)
            # Detect + track occlusion boundaries ONCE per control step, exactly as the
            # MPC's run loop does, then hand the result to both the planner and the
            # plot recorder. Previously mppi() re-queried and the recorder called the
            # tracker again with None, so the tracker advanced twice per step and the
            # plot lagged the planner by one step.
            if OCCLUSION_AWARE and OCC_TRACKER is not None and LIDAR_COSTMAP.ready:
                _s = LIDAR_COSTMAP.occlusion_segments(
                    ego_fwd=ego_fwd, fwd_half_angle_deg=OCC_FWD_HALF_ANGLE)
                _s = OCC_TRACKER.update(_s, time.monotonic())
                if _s is not None and len(_s):
                    # Circle keep-out ⇒ collapse each boundary onto its corner (as the
                    # MPC does), so cost and plot use identical geometry.
                    _s = _s.copy()
                    _s[:, 1, :] = _s[:, 0, :]
                OCC_SEGS_NOW = _s
            else:
                OCC_SEGS_NOW = None
            # Accumulate the per-scan obstacle-circle centres into the episode-wide
            # union (deduped by rounded world position) for the trajectory plot.
            if save_traj is not None and LIDAR_COSTMAP.ready:
                _circ = LIDAR_COSTMAP.circles()
                if _circ is not None:
                    for wx, wy, _r in _circ:
                        occ_seen[(round(float(wx), 1), round(float(wy), 1))] = None
                # Only track occlusion boundaries when occlusion-awareness is actually
                # active — otherwise the planner ignores them and plotting them is
                # misleading (the purple diamonds imply a constraint that isn't there).
                if OCCLUSION_AWARE and OCC_TRACKER is not None:
                    # Record the TRACKED range-jump boundaries within OCC_QUERY_R,
                    # keyed on stable track id so a corner re-seen across scans is ONE
                    # entry, not a smear of jittered near-duplicates. Value keeps the
                    # ego pose + episode time of the first sighting.
                    _segs = OCC_SEGS_NOW
                    if _segs is not None and len(_segs):
                        _tids = OCC_TRACKER.ids()
                        _ego = np.array([s[0], s[1]])
                        for _i, _sg in enumerate(_segs):
                            if _i >= len(_tids):
                                break
                            if point_segment_distance_np(_ego[None, :], _sg[0],
                                                         _sg[1])[0] <= OCC_QUERY_R:
                                occ_track.setdefault(
                                    _tids[_i], (_sg[0][0], _sg[0][1], _sg[1][0],
                                                _sg[1][1], s[0], s[1], ep_steps * DT))
        u_nom, mean = mppi(s, mean, obstacles, goal_xy, u_prev)

        u_cmd      = u_nom
        cbf_engaged = False

        a_cmd     = float(np.clip(u_cmd[0], A_MIN, A_MAX))
        delta_cmd = float(np.clip(u_cmd[1], -DELTA_LIM, DELTA_LIM))
        u_prev    = np.array([a_cmd, delta_cmd])   # feed the Δu smoothness cost next step

        if LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready and ep_steps % 20 == 0:
            _circ = LIDAR_COSTMAP.circles()
            if _circ is not None and len(_circ):
                w0, w1, wr = _circ[:, 0], _circ[:, 1], _circ[:, 2]
                surf = np.hypot(w0 - s[0], w1 - s[1]) - wr          # clearance to surface
                k = int(np.argmin(surf))
                d_field = LIDAR_COSTMAP.distance(np.array([s[0]]), np.array([s[1]]))[0]
                print(f"[CHK] ego=({s[0]:.1f},{s[1]:.1f})  nearest circle=({w0[k]:.1f},{w1[k]:.1f}"
                      f",r={wr[k]:.1f}) d_field={d_field:.1f}")


        # Advance Python-side kinematic state to match what Unity will compute
        v = s[3]
        speed_frac   = min(v / max(STEER_ROLLOFF_SPD, 1e-3), 1.0)
        eff_limit    = DELTA_LIM * (1.0 - speed_frac * (1.0 - STEER_ROLLOFF_MIN))
        delta_target = float(np.clip(delta_cmd, -eff_limit, eff_limit))
        delta_actual = delta_actual + float(np.clip(
            delta_target - delta_actual, -MAX_STEER_RATE * DT, MAX_STEER_RATE * DT))
        if ACCEL_TAU > 1e-3:
            accel_actual += (float(np.clip(a_cmd, A_MIN, A_MAX)) - accel_actual) * (DT / ACCEL_TAU)
        else:
            accel_actual = float(np.clip(a_cmd, A_MIN, A_MAX))

        # Log nearest obstacle distance — measured directly from geometry (min |rel_xy|) so it's
        # correct regardless of the Unity-side dSafe baked into obs[17]. (Back-computing it from
        # obs[17] + D_SAFE² read 0.00 whenever the retuned Python D_SAFE < Unity's dSafe.)
        h_val = float(obs[17])
        dist  = float(min((np.hypot(rel[0], rel[1]) for rel, _ in obstacles), default=np.inf))
        min_h    = min(min_h, h_val)
        min_dist = min(min_dist, dist)


        # Record ego pose this step (world frame: x=Unity Z, y=Unity X).
        traj.append((ep_steps * DT, s[0], s[1], s[2], s[3], a_cmd, delta_cmd))
        # Record obstacle world positions this step (rel_xy is ego-relative world coords).
        for rel, _ov in obstacles:
            obs_track.append((ep_steps * DT, s[0] + rel[0], s[1] + rel[1]))

        action = ActionTuple(
            continuous=np.array([[a_cmd, delta_cmd]], dtype=np.float32)
        )
        env.set_actions(behavior_name, action)
        try:
            env.step()
        except UnityCommunicatorStoppedException:
            unity_stopped = True
            break
        decision_steps, terminal_steps = env.get_steps(behavior_name)

    if unity_stopped:
        print("\n[Controller] Unity stopped early (Editor left Play mode, or the app closed).\n"
              "             If this was unexpected: check the Unity Console for an exception, "
              "and avoid editing C# scripts while the Editor is in Play mode (that triggers a "
              "recompile and drops the connection).")
    try:
        env.close()
    except Exception:
        pass

    verdict = "collision" if collided else ("reached" if reached else "timeout")

    if save_traj is not None and traj:
        # Overlay the run-wide UNION of static occluders and occlusion boundaries the
        # perception produced (not a single decaying end-of-run frame), so the plot shows
        # every wall/blind corner the ego reacted to. obs_track dynamic slots are often
        # empty when no agent ever enters range.
        occ_pts = np.array(list(occ_seen.keys())) if occ_seen else None
        if occ_track:
            _vals = np.array(list(occ_track.values()), dtype=float)
            occ_segments, occ_ego = _vals[:, :4].reshape(-1, 2, 2), _vals[:, 4:]
        else:
            occ_segments = occ_ego = None
        _save_trajectory(save_traj, verdict, goal_xy, np.asarray(traj),
                         np.asarray(obs_track) if obs_track else None,
                         occ_pts, occ_segments=occ_segments, occ_ego=occ_ego,
                         # OCC_HORIZON, not the PLANNING horizon: the keep-out is
                         # capped at OCC_HORIZON in the cost, so drawing H_MPPI*DT
                         # would overstate the constraint by V_TARGET*(6.0-3.0)=9 m.
                         capsule_horizon=OCC_HORIZON,
                         show_occlusion=show_occlusion_plot)

    print(f"\n[Controller] steps={ep_steps:4d}  min_dist={min_dist:5.2f} m "
          f"(target >= {D_SAFE:.1f} m)  min_h={min_h:8.2f} (target >= 0)  → {verdict.upper()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",         default=None,
                   help="Path to a tuning YAML (default: config.yaml next to this script). "
                        "Applied at IMPORT time by taxi_config, not here — this entry exists "
                        "so the flag appears in --help. $TAXI_CONFIG does the same.")
    p.add_argument("--exec",           default=None)
    p.add_argument("--port",           default=5004,  type=int)
    p.add_argument("--sysid",          default=False,  type=lambda x: x.lower() == "true")
    p.add_argument("--noise-std",      default=0.0,   type=float,
                   help="Std-dev of Gaussian noise injected into obstacle obs [m]. 0=off.")
    p.add_argument("--detect-range",   default=float("inf"), type=float,
                   help="Euclidean detection range [m]. Obstacles beyond this distance "
                        "are masked from MPPI and CBF, simulating finite sensor range. "
                        "Default=inf (oracle). Try 25 for a tight reaction window.")
    p.add_argument("--n-scenarios",    default=N_SCEN, type=int,
                   help="Number of sampled obstacle futures per MPPI call. Only used with --uncertainty.")
    p.add_argument("--w-info",         default=W_INFO, type=float,
                   help="Weight on the uncertainty-caution term (slow down when approaching an "
                        "obstacle with an ambiguous predicted future). 0=off. Only used with --uncertainty.")
    p.add_argument("--d-infl",         default=D_INFL, type=float,
                   help="MPPI obstacle influence radius [m] — the distance at which the planner "
                        "STARTS bending around an obstacle. Raise it (e.g. 40-60) for long-range "
                        "planning against long-range threats. Must stay >= --d-safe.")
    p.add_argument("--d-safe",         default=D_SAFE, type=float,
                   help="Hard keep-out radius [m] for the CBF barrier and the BIG close-in penalty. "
                        "Governs close-in avoidance; leave at default unless you want a larger "
                        "physical standoff. Must stay <= --d-infl.")
    p.add_argument("--info-range",     default=INFO_RANGE, type=float,
                   help="Radius [m] of the uncertainty (belief-entropy) caution term. Raise it to "
                        "match --d-infl so the ego slows for ambiguous distant threats. Only used "
                        "with --uncertainty.")
    p.add_argument("--lidar-costmap",  action="store_true",
                   help="Down-sample the published PointCloud2 each control step and cover the "
                        "static obstacles (walls/parked aircraft/buildings) with circles, added "
                        "to the MPPI cost as clearance ≥ D_SAFE_HARD to each circle surface. "
                        "Needs ROS 2 sourced (rclpy) and the ros_tcp_endpoint running.")
    p.add_argument("--lidar-topic",    default="/point_cloud",
                   help="PointCloud2 topic for the LiDAR map. Default: /point_cloud.")
    p.add_argument("--visibility-cost", action="store_true",
                   help="DEPRECATED / no-op: the active-perception visibility term relied on the "
                        "removed persistent occupancy grid and is ignored.")
    p.add_argument("--no-occlusion-plot", action="store_true",
                   help="omit the occlusion keep-out circles, corner centres and "
                        "ego-at-detection markers from the saved trajectory plots")
    p.add_argument("--occlusion-aware", action="store_true",
                   help="Add occlusion-aware forward-reachable-set keep-outs + an RSS sightline "
                        "speed cap (same model/constants as the MPC controller): a worst-case "
                        "hidden agent on each blind-corner frontier seeds an EXPANDING keep-out "
                        "(radius grows as v_target·t) and the ego is speed-capped so it can stop "
                        "before the nearest occlusion. Requires --lidar-costmap.")
    p.add_argument("--save-traj", default=None, metavar="DIR",
                   help="Save the run's ego trajectory as CSV + a top-down PNG plot into DIR.")
    args = p.parse_args()

    if args.d_infl < args.d_safe:
        p.error(f"--d-infl ({args.d_infl}) must be >= --d-safe ({args.d_safe}): "
                "the soft influence ring cannot be inside the hard keep-out radius.")

    run(unity_exec_path=args.exec if args.exec != "None" else None,
        port=args.port,
        run_sysid=args.sysid,
        noise_std=args.noise_std,
        detect_range=args.detect_range,
        n_scenarios=args.n_scenarios,
        w_info=args.w_info,
        d_infl=args.d_infl,
        d_safe=args.d_safe,
        info_range=args.info_range,
        lidar_costmap=args.lidar_costmap,
        lidar_topic=args.lidar_topic,
        visibility_cost=args.visibility_cost,
        occlusion_aware=args.occlusion_aware,
        show_occlusion_plot=not args.no_occlusion_plot,
        save_traj=args.save_traj)
