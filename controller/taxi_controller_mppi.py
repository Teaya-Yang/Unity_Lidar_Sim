"""
taxi_controller.py
==================
External Python controller for the Unity taxiing environment.
Connects via the ML-Agents Python API, runs MPPI + HOCBF-QP, and sends
[a, delta] actions back to Unity each decision step.

Observation contract (must match TaxiAgent.cs CollectObservations, OBS_SIZE=7):

  World/Euclidean frame:
    obs[0]   x_ego    — Unity Z position [m]
    obs[1]   y_ego    — Unity X position [m]
    obs[2]   theta    — heading [rad]
    obs[3]   v        — speed [m/s]
    obs[4]   goal     — remaining Euclidean distance to goal [m]
    obs[5]   goal_z   — goal world Z position [m] (same axis as obs[0])
    obs[6]   goal_x   — goal world X position [m] (same axis as obs[1])

  NO DYNAMIC-OBSTACLE SLOTS. Other aircraft used to arrive here as exact positions
  and velocities lifted straight from Unity's scenario manager — an oracle. They are
  now perceived ONLY through the LiDAR point cloud, like every other obstacle, so the
  planner cannot use knowledge the sensor model does not provide. --dynamic-obstacles
  turns those returns back into moving agents by SENSING them: cluster, track, classify
  by displacement, then apply the same expanding keep-out the occlusion phantoms get
  (see dynamic_clusters.py).

Action contract:
  act[0]  a_cmd     — acceleration [m/s^2], clipped to [A_MIN, A_MAX]
  act[1]  delta_cmd — steering [rad],       clipped to [-DELTA_LIM, DELTA_LIM]

MPPI runs entirely in the world frame: s0 = [x, y, theta, v, delta, accel].
"""

import numpy as np
import argparse
import time
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.exception import UnityCommunicatorStoppedException

from occlusion_capsules import (occlusion_stage_cost, point_segments_min_distance,
                                point_segment_distance_np)
import dynamic_clusters as dyn_clusters
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

OCC_RANGE_DBG   = None   # (min_ego_dist, corner_x, corner_y, ego_x, ego_y) for the nearest
                         # boundary that survived the goal drop — says whether the range gate
                         # is right to reject it.
OCC_GATE_DBG    = None   # (n_in, n_after_goal_drop, r_goal, min_goal_dist) from the LAST
                         # solve — lets the [DEBUG occ] trace name the gate that emptied the set.
OCC_PLAN        = None   # (H,2) the PLANNED future path from the last solve, in world
                         # (a0,a1). Stage k of this array is the pose the occlusion cost
                         # checked against radius r(t_k) — the pairing the plan figure draws.
OCC_SEGS_USED   = None   # (M,2,2) the segments the LAST solve actually constrained against —
                         # post-gating, so it is what the cost saw, not what perception found.
                         # The trajectory plot replays these per timestamp.
OCC_USED_N      = 0      # boundaries that actually entered the LAST solve's cost. Written by
                         # mppi(), read by the [DEBUG occ] trace in run(): if this is 0 while
                         # segments exist, the ego is provably driving the occlusion-UNAWARE
                         # trajectory no matter what --occlusion-aware was passed.

OCCLUSION_AWARE = False   # set True by --occlusion-aware (needs --lidar-costmap): enables the
                          # FRS keep-out + sightline speed cap above (mirrors the MPC controller).

# ── Sensed dynamic obstacles (LiDAR clusters) ────────────────────────────────
# The same expanding keep-out as the occlusion one, but seeded by an agent the LiDAR can
# actually SEE rather than by a hypothetical one behind a corner: obstacle returns are
# clustered (dynamic_clusters.cluster_points), tracked across scans, and the tracks whose
# estimated speed clears DYN_V_MIN get r_keep(t_k) = D_SAFE_HARD + r_cluster +
# V_TARGET·(t_k + age). Same V_TARGET, same W_HARD, same one-sided quadratic
# (occlusion_stage_cost) — one avoidance mechanism, two sources of hazard.
#
# NOT a replacement for the static circle term: a dynamic cluster's points stay in the
# obstacle circles too. The expanding disc strictly contains the static keep-out, so the
# overlap changes nothing, whereas removing a MIS-classified aircraft from the static set
# would delete the only constraint protecting the ego from it.
DYN_CELL       = CFG["dynamic_clusters"]["cell"]
DYN_MIN_POINTS = CFG["dynamic_clusters"]["min_points"]
DYN_MAX_RADIUS = CFG["dynamic_clusters"]["max_radius"]
DYN_ASSOC      = CFG["dynamic_clusters"]["assoc_radius"]
DYN_ALPHA      = CFG["dynamic_clusters"]["alpha"]
DYN_VEL_BETA   = CFG["dynamic_clusters"]["vel_beta"]
DYN_TTL        = CFG["dynamic_clusters"]["ttl"]
DYN_MIN_HITS   = CFG["dynamic_clusters"]["min_hits"]
DYN_V_MIN      = CFG["dynamic_clusters"]["v_min"]
DYN_MIN_DYN_HITS = CFG["dynamic_clusters"]["min_dyn_hits"]
DYN_REQUIRE_MOTION = CFG["dynamic_clusters"]["require_motion"]
DYN_WINDOW     = CFG["dynamic_clusters"]["dyn_window"]
DYN_EXTENT_FRAC = CFG["dynamic_clusters"]["extent_frac"]
DYN_RESEG_RATIO = CFG["dynamic_clusters"]["resegment_ratio"]
DYN_GROW_HORIZON = CFG["dynamic_clusters"]["grow_horizon"]
DYN_QUERY_R    = CFG["dynamic_clusters"]["query_r"]
K_DYN          = CFG["dynamic_clusters"]["k_dyn"]
DYN_INCLUDE_AGE = CFG["dynamic_clusters"]["include_age"]

DYNAMIC_AVOID  = False   # set True by --dynamic-obstacles (needs --lidar-costmap)
DYN_TRACKER    = None    # DynamicClusterTracker, created in run()
DYN_NOW        = None    # (K,4) [c0, c1, r_cluster, age] confirmed movers for THIS control
                         # step, written once per step by run() and consumed by mppi() —
                         # exactly the OCC_SEGS_NOW pattern, and for the same reason: the
                         # tracker must be advanced once per step, not once per consumer.
DYN_USED       = None    # (K,4) the movers the LAST solve actually constrained against,
                         # post-gating. Recorded for the trajectory plot.
DYN_DBG        = None    # (n_clusters, n_tracks, n_dynamic, max_speed) from the last step
DYN_PUB        = None    # DynamicClusterPublisher — feeds DynamicAgentVisualizer.cs so the
                         # Unity Scene view shows the SAME expanding circles the cost used
VISIBILITY_COST = False   # vestigial: --visibility-cost is a documented no-op

DETECTION_RANGE = float("inf")   # vestigial: it masked ORACLE obstacle slots beyond a radius,
                                 # and there are no such slots any more. Real sensor range is
                                 # now whatever the LiDAR reports (scan.max_range).

OBS_SIZE = 4 + 1 + 2   # 7: ego(4) + goal distance(1) + goal position(2)

rng = np.random.default_rng(42)



# ── Observation unpacking ────────────────────────────────────────────────────

def obs_to_state(obs: np.ndarray, prev_delta: float = 0.0, prev_accel: float = 0.0):
    """
    Unpack the Unity 7-D observation vector into world/Euclidean coordinates.

    Returns
    -------
    s          : np.ndarray shape (6,) — [x_fwd, y_lat, theta, v, delta_actual, accel_actual]
    goal_xy    : np.ndarray shape (2,) — goal world position (z, x), same axes as ego (obs[5:7]).

    Dynamic agents are NOT unpacked here — they are not in the vector any more. The only
    obstacle information the planner receives is the LiDAR circle map, so there is nothing
    to inject sensor noise into either (the old inject_sensor_noise corrupted the obstacle
    slots and deliberately left ego/goal clean; with the slots gone it had no effect).
    """
    ego0    = float(obs[0])   # x_ego (world)
    ego1    = float(obs[1])   # y_ego (world)
    ego2    = float(obs[2])   # theta (world)
    v       = float(obs[3])
    goal_xy = np.array([float(obs[5]), float(obs[6])])   # goal world (z, x)

    s = np.array([ego0, ego1, ego2, v, prev_delta, prev_accel])
    return s, goal_xy


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


def mppi(s0, mean, goal_xy, u_prev=None):
    """
    Sample K_MPPI rollouts with realistic 6D state dynamics.

    s0 = [x, y, theta_global, v, delta, accel]. Rollout in world frame.
    goal_xy = goal world position (x_fwd, y_lat), same axes as the rollout state.
    u_prev  = last APPLIED control [a, delta], used for the Δu_0 = u_0 - u_prev term
              (defaults to zero if None).

    Obstacles are NOT an argument: dynamic agents no longer arrive as an oracle feed.
    Everything the rollouts avoid comes from the LiDAR — static circles via
    LIDAR_COSTMAP.distance(), occlusion boundaries via occ_segs below, and SENSED
    movers via DYN_NOW (clustered + tracked in run(), consumed here).

    Returns (u_nom, new_mean).
    """
    global OCC_USED_N, OCC_GATE_DBG, OCC_RANGE_DBG, OCC_SEGS_USED, OCC_PLAN, DYN_USED
    # Nominal MPPI weights / limits. The frontal-threat "go-around" gate (in_corridor / aimed_at_us)
    # has been removed to keep the planner simple: plain lane-following + obstacle rings, and the
    # LiDAR visibility term is what drives any deliberate off-lane motion.
    sig_d_eff  = SIG_D
    w_off_eff  = W_OFF
    w_lat_eff  = W_LAT
    a_min_eff  = A_MIN
    lane_bias  = 0.0

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
    OCC_USED_N = 0
    OCC_GATE_DBG = None
    OCC_RANGE_DBG = None
    OCC_SEGS_USED = None
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
                # Which gate emptied the set — the goal drop or the range gate below.
                # r_goal is D_SAFE_HARD + V_TARGET*H_MPPI*DT, so a long MPPI horizon makes
                # this exclusion zone large enough to swallow every boundary near the goal.
                OCC_GATE_DBG = (len(_op), int((_gd > _rgc).sum()), _rgc, float(_gd.min()))
            if len(_seg):
                # Range-gate on true point-to-CAPSULE-axis distance, not on the corner:
                # a long boundary whose corner sits beyond OCC_QUERY_R can still have its
                # far end sweeping past the ego, and dropping it would silently un-see it.
                _ego = np.array([[s0_fwd, s0_lat]])
                _d = np.array([point_segment_distance_np(_ego, s[0], s[1])[0]
                               for s in _seg])
                _in = _d < OCC_QUERY_R
                # Where the surviving boundaries actually ARE relative to the ego. If the
                # nearest is far beyond query_r while the ego is metres from a real corner,
                # the fault is upstream in detection/framing, not in this gate.
                # Index into the POST-goal-drop set, so use its own corners, not _op.
                _j = int(np.argmin(_d))
                _opf = _seg[:, 0, :]
                OCC_RANGE_DBG = (float(_d.min()), float(_opf[_j, 0]), float(_opf[_j, 1]),
                                 float(s0_fwd), float(s0_lat))
                if _in.any():
                    _near = _seg[_in]
                    occ_segs = _near[np.argsort(_d[_in])[:K_OCC]]
                    OCC_USED_N = len(occ_segs)
                    OCC_SEGS_USED = occ_segs

    # Sensed dynamic obstacles for this solve. Gated exactly like the occlusion set —
    # nearest K within query_r — and then held FIXED over the horizon: the circle grows
    # with t_k rather than the centre translating, because a two-scan centroid difference
    # at ~1 Hz cannot support a trajectory prediction (see dynamic_clusters' docstring).
    dyn_set = None
    DYN_USED = None
    if DYNAMIC_AVOID and DYN_NOW is not None and len(DYN_NOW) and K_DYN > 0:
        dyn_set = dyn_clusters.select_nearest(DYN_NOW, (s0_fwd, s0_lat),
                                              DYN_QUERY_R, K_DYN)
        DYN_USED = dyn_set

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

        # MPC-REFERENCE stage cost (taxi_cost.stage_cost): bounded linear goal pull plus the
        # static keep-out. This REPLACES MPPI's former closure/progress rewards and the
        # exp-weighted velocity term: those applied ~400x more goal pressure than the MPC,
        # so the identical occlusion penalty was far weaker in relative terms and the ego
        # cut corners the MPC would not. The constant-velocity obstacle ray that used to be
        # built here (mirroring the MPC's P_opos + P_ovel*t_k) is gone with the oracle feed.
        _d_static = None
        if STATIC_AVOID and LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
            # ABSOLUTE world (a0,a1): the circles live in world coords and distance()
            # expects absolute queries (as does stage_cost's goal). Passing ego-relative deltas made the
            # clearance huge, so the static keep-out never fired (ego drove through walls).
            _d_static = LIDAR_COSTMAP.distance(fwd, lat)
        cost += tcost.stage_cost(
            fwd, lat, th, vv, u_k, u_km1,
            goal_xy=(goal_fwd, goal_lat), v_des=v_des_eff, t_k=(k + 1) * DT,
            r_act=R_ACT, r_dact=R_DACT,
            w_goal_run=W_GOAL_RUN, w_head=W_HEAD, w_v=W_V,
            d_static=_d_static, d_safe_static=D_SAFE_HARD, w_static=W_HARD)

        # ── Occlusion-aware safety: FRS keep-out + RSS sightline cap ──────────
        # Canonical formula shared with the MPC (occlusion_capsules.occlusion_stage_cost).
        if occ_segs is not None:
            cost += occlusion_stage_cost(
                _dist_to_occ(fwd, lat), vv, (k + 1) * DT,
                V_TARGET, D_SAFE_HARD, W_HARD)

        # ── Sensed dynamic obstacles: the SAME expanding keep-out, visible source ──
        # occlusion_stage_cost with a per-cluster base radius. Looping over clusters (at
        # most K_DYN, typically <5) keeps the shared cost function untouched while letting
        # each mover carry its own extent; the inner call is still vectorised over the
        # K_MPPI rollouts, which is where the cost actually lives.
        if dyn_set is not None:
            # Expansion time, CAPPED (unlike the occlusion term, which grows over the whole
            # horizon): a visible mover is in every solve, so an uncapped 6 s bubble is a
            # permanent ~45 m no-go disc that shoves the ego off course for a single taxiing
            # aircraft. See dynamic_clusters.grow_horizon in the config.
            t_dyn = (k + 1) * DT
            for c0, c1, r_c, age in dyn_set:
                d_dyn = np.hypot(fwd - c0, lat - c1)
                # Base radius at t_k = 0: ego margin + the cluster's own extent + the
                # travel the agent could already have made since the centroid was measured.
                d_base = D_SAFE_HARD + r_c
                cost += occlusion_stage_cost(
                    d_dyn, vv, t_dyn, V_TARGET, d_base, W_HARD)


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

    # Replay the OPTIMAL sequence deterministically to get the PLANNED future path.
    # `opt` is the MPPI-weighted mean over samples — the sequence the ego actually
    # follows — not the single argmin-cost sample, which is one noisy draw and is not
    # what gets executed. Recorded so the plot can show one solve: the future
    # trajectory, and how large the keep-out is at each of its stages.
    _st = np.tile(np.asarray(s0, float)[None, :], (1, 1))
    _plan = np.empty((H_MPPI, 2))
    for _k in range(H_MPPI):
        _st = _rollout_step(_st, opt[_k:_k + 1, 0], opt[_k:_k + 1, 1])
        _plan[_k] = _st[0, :2]
    OCC_PLAN = _plan
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
                     occ_frames=None, capsule_horizon=None,
                     max_frames=6, single_radius=False, show_expansion=False,
                     show_occlusion=True):
    """Persist the run's ego trajectory as CSV + a top-down PNG plot.

    verdict: short outcome tag ("reached" / "collision" / "timeout") used in the
    output filename and plot title.
    traj columns: t, x, y, theta, v, a_cmd, delta_cmd  (x=Unity Z, y=Unity X).
    obs_track (optional): (N, 3) array of per-step dynamic-obstacle world
    positions (t, x, y), same frame as traj — overlaid on the plot.
    occ_pts (optional): (M, 2) array of static occluder (OCC) cell centres
    (x, y) from the LiDAR costmap, same frame — the buildings/walls.
    occ_frames (optional): per-control-step snapshots (t, ego_x, ego_y, segments) of
    the keep-out the planner was ACTUALLY constrained by at that instant, recorded in
    run() straight from mppi()'s post-gating set.

    THE PLOT IS A TIME SERIES, not an episode-wide union. max_frames timestamps are
    sampled evenly across the recorded ones; each contributes
      * a large dot ON the trajectory at the ego pose at that instant, and
      * the expanding keep-out shape as it stood at that instant,
    both in ONE colour, distinct per timestamp. Reading the plot is then a
    colour match: this is where the ego was, and this is what it was avoiding, at the
    same moment. Earlier versions drew every boundary the run ever saw, tied to the ego
    pose where each was FIRST constrained, which mixed timestamps in a single picture.

    The shape is the one MPPI evaluates — capsule_polygon(near, far, r), the level set
    of the point-to-SEGMENT distance that _dist_to_occ measures. A boundary collapsed to
    its corner (circle mode) is a degenerate segment and comes out as a circle through
    the same call.

    WHAT THE GROWTH ACROSS FRAMES MEANS. The radius drawn at a frame is
    r = D_SAFE_HARD + V_TARGET·min(t − t_first_seen, capsule_horizon): the forward
    reachable set of a phantom that could have been at the corner when it was first
    constrained, given the time elapsed SINCE. That is the physical set, and it is what
    makes the expansion visible along the path.

    It is NOT step-for-step what the cost enforces. Each solve re-applies the growth
    over its OWN prediction horizon, from D_SAFE_HARD at t_k=0 out to
    D_SAFE_HARD + V_TARGET·H·DT at the last stage — it does not accumulate across
    episode time, because a corner still in view is re-observed and the phantom
    re-anchored every scan. So read the rings as "how far a hidden agent could have got
    by this timestamp", not as "the radius the planner used at this timestamp".

    capsule_horizon (optional): horizon length [s] the keep-out expanded over
    (N·DT). Defaults to 3 s if not given.
    single_radius: True ⇒ Algorithm 1 mode, the ENFORCED radius is the fixed
    D_SAFE_HARD + V_TARGET·capsule_horizon (no per-stage growth).
    show_occlusion: False ⇒ draw none of the occlusion artifacts, leaving just the
    trajectory, static occluders, obstacles and goal.
    show_expansion: unused — kept so existing callers do not break. The expansion is
    now shown ACROSS TIMESTAMPS (one ring per sampled frame, growing with the time the
    boundary has been known) rather than as nested rings inside a single frame, which
    is what buried the trajectory before.
    max_frames: how many timestamps to draw. More than a handful and the keep-outs
    overlap into mush, since consecutive frames differ by only a few metres.
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
    # Per-timestamp keep-outs. Drawn first (lowest zorder) so the ego path stays
    # readable on top of them.
    frames = []
    if show_occlusion and occ_frames is not None and len(occ_frames):
        from occlusion_capsules import capsule_polygon
        # One DISTINCT colour per timestamp, from a qualitative palette. A continuous
        # ramp (cool/viridis-like) was hard to read here: neighbouring frames came out
        # nearly the same hue, which is the opposite of what this plot needs — the
        # frames are discrete and the whole point is telling them apart. Hues chosen to
        # stay clear of the viridis speed scatter's green-yellow band.
        # Mid-dark values only: a capsule outline is a thin line on white, so pale
        # hues (pure yellow/cyan) vanish at 0.8 lw.
        frame_colors = ["#e6194b", "#f58231", "#b8860b", "#3cb44b",
                        "#4363d8", "#911eb4", "#008080", "#f032e6"]

        # Sample evenly across the recorded steps. Consecutive frames differ by a few
        # metres of ego motion, so drawing all of them just overlaps into mush; evenly
        # spaced ones show how the keep-out evolved over the run.
        idx = (np.linspace(0, len(occ_frames) - 1, min(max_frames, len(occ_frames)))
               .round().astype(int))
        frames = [occ_frames[i] for i in dict.fromkeys(idx.tolist())]

        T = float(capsule_horizon) if capsule_horizon else 3.0

        # WHICH radius each timestamp gets. The keep-out is r(tau) = D_SAFE_HARD +
        # V_TARGET*tau, where tau is how long the worst-case hidden agent has had to
        # move. Drawing every frame at the same tau (the enforced tau = T) makes a
        # persistent corner render as N identical circles — that is why the plot showed
        # one outline. So tau is measured from when each boundary was FIRST constrained:
        # the frame that introduced it gets tau = 0, later frames get tau = t - t_first,
        # and the corner blooms outward across the timestamps exactly as the forward
        # reachable set does.
        #
        # tau is CAPPED at T because the cost caps it there (occlusion.horizon): past
        # that the planner stops growing the radius, so drawing further growth would
        # overstate the constraint. Timestamps beyond t_first + T therefore share the
        # outermost ring rather than each getting a bigger one.
        t_first = {}          # rounded geometry key -> episode time first constrained
        corner_labelled = False

        def _key(seg):
            # 2 m quantisation: the tracker's EMA jitters a corner well under that
            # between scans, while distinct corners in these scenes sit much further
            # apart — so re-sightings merge and real boundaries do not.
            return tuple(np.round(np.asarray(seg).ravel() / 2.0).astype(int))

        for fi, (t_ep, ex, ey, segs, _plan) in enumerate(frames):
            col = frame_colors[fi % len(frame_colors)]
            segs = np.asarray(segs, float).reshape(-1, 2, 2)

            for seg in segs:
                k = _key(seg)
                t0 = t_first.setdefault(k, float(t_ep))
                tau = min(float(t_ep) - t0, T)

                # capsule_polygon is the level set of the point-to-SEGMENT distance
                # _dist_to_occ measures, so this is the constraint MPPI applied, not an
                # illustration of it. A boundary collapsed to its corner is a degenerate
                # segment and comes out as a circle through the same call.
                poly = capsule_polygon(seg[0], seg[1], D_SAFE_HARD + V_TARGET * tau)
                ax.plot(poly[:, 0], poly[:, 1], "-", color=col, lw=1.6, alpha=0.95,
                        zorder=0)

                # The blind corner the phantom is anchored to. Labelled once for the
                # legend: len(t_first)==1 stayed true across frames while a single
                # boundary persisted, so it emitted a duplicate entry per frame.
                ax.plot(seg[0, 0], seg[0, 1], ".", color="k", ms=5, zorder=2,
                        label=None if corner_labelled else "occlusion corner (centre)")
                corner_labelled = True

            # The ego ON the trajectory at this timestamp, in the SAME colour as the
            # rings drawn for it above.
            ax.plot(ex, ey, "o", mfc=col, mec="k", ms=11, mew=1.4, zorder=6,
                    label=f"ego + keep-out @ t = {float(t_ep):.1f} s")

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
    # No colorbar for the keep-outs: the palette is qualitative, and each timestamp
    # names itself in the legend.
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

    # ── Second figure: ONE solve, its planned future, and the keep-out per stage ──
    # The overview above is an episode summary and cannot show the actual constraint,
    # because the radius is rebuilt every solve (see occ_frames in the docstring). This
    # one shows the pairing the cost really evaluates: stage k of the PLANNED path
    # against radius r(t_k) = D_SAFE_HARD + V_TARGET*t_k, both at the same instant.
    plan_frames = [f for f in (occ_frames or []) if f[4] is not None and len(f[4])]
    if show_occlusion and plan_frames:
        from occlusion_capsules import capsule_polygon, point_segments_min_distance

        # Pick the TIGHTEST solve — the one whose plan came closest to violating the
        # keep-out, max_k (r_keep(t_k) - d_k). That is the step where the constraint was
        # actually doing something; an arbitrary or first step usually shows a plan far
        # from every boundary, which demonstrates nothing.
        def _slack(fr):
            segs, plan = fr[3], fr[4]
            tk = (np.arange(len(plan)) + 1) * DT
            d = point_segments_min_distance(plan[:, 0], plan[:, 1], segs)
            return float(np.max((D_SAFE_HARD + V_TARGET * tk) - d))

        t_sel, ex, ey, segs, plan = max(plan_frames, key=_slack)
        segs = np.asarray(segs, float).reshape(-1, 2, 2)
        tk_all = (np.arange(len(plan)) + 1) * DT
        d_all = point_segments_min_distance(plan[:, 0], plan[:, 1], segs)
        viol = (D_SAFE_HARD + V_TARGET * tk_all) - d_all

        f2, a2 = plt.subplots(figsize=(13, 13))
        if occ_pts is not None and len(occ_pts):
            a2.scatter(occ_pts[:, 0], occ_pts[:, 1], c="0.35", s=8, marker="s",
                       alpha=0.5, label="static occluders", zorder=1)
        # Executed path for context, so the plan can be compared against what happened.
        a2.plot(x, y, "-", color="0.75", lw=1.0, zorder=2, label="executed trajectory")

        # Sample stages across the horizon: each gets a colour, a dot on the PLANNED
        # path, and the keep-out at that stage's radius in the same colour.
        n_show = min(max_frames, len(plan))
        stages = np.linspace(0, len(plan) - 1, n_show).round().astype(int)
        a2.plot(plan[:, 0], plan[:, 1], "-", color="0.4", lw=1.2, zorder=3,
                label="planned future trajectory")
        for si, k in enumerate(dict.fromkeys(stages.tolist())):
            col = frame_colors[si % len(frame_colors)]
            t_k = float(tk_all[k])
            r_k = D_SAFE_HARD + V_TARGET * t_k
            for seg in segs:
                poly = capsule_polygon(seg[0], seg[1], r_k)
                a2.plot(poly[:, 0], poly[:, 1], "-", color=col, lw=1.6, alpha=0.95,
                        zorder=0)
            a2.plot(plan[k, 0], plan[k, 1], "o", mfc=col, mec="k", ms=11, mew=1.4,
                    zorder=6,
                    label=f"$t_k$ = {t_k:.1f} s   r = {r_k:.0f} m   "
                          f"d = {d_all[k]:.0f} m" +
                          ("  VIOLATED" if viol[k] > 0 else ""))
        for seg in segs:
            a2.plot(seg[0, 0], seg[0, 1], ".", color="k", ms=6, zorder=4)
        a2.plot(ex, ey, "s", color="tab:red", ms=11, mec="k", zorder=7,
                label=f"ego now (t = {float(t_sel):.1f} s)")

        px_, py_ = plan[:, 0], plan[:, 1]
        allx = np.concatenate([px_, segs[:, :, 0].ravel(), [ex]])
        ally = np.concatenate([py_, segs[:, :, 1].ravel(), [ey]])
        cx2, cy2 = 0.5 * (allx.min() + allx.max()), 0.5 * (ally.min() + ally.max())
        half2 = 0.5 * max(allx.max() - allx.min(), ally.max() - ally.min()) + 15.0
        a2.set_xlim(cx2 - half2, cx2 + half2); a2.set_ylim(cy2 - half2, cy2 + half2)
        a2.set_aspect("equal", adjustable="box")
        a2.set_xlabel("x  (Unity Z) [m]"); a2.set_ylabel("y  (Unity X) [m]")
        a2.set_title(f"planned horizon at t = {float(t_sel):.1f} s — "
                     f"keep-out per stage (max violation {viol.max():+.1f} m)")
        a2.legend(loc="best", fontsize=9); a2.grid(True, alpha=0.3)
        f2.tight_layout()
        f2.savefig(f"{stem}_plan.png", dpi=120)
        plt.close(f2)
        print(f"[traj] saved {stem}_plan.png "
              f"(solve at t={float(t_sel):.1f}s, max violation {viol.max():+.2f} m)")


def run(unity_exec_path=None, port=5004, run_sysid=True,
        noise_std=0.0, detect_range=float("inf"),
        uncertainty=False, n_scenarios=N_SCEN, w_info=W_INFO,
        d_infl=D_INFL, d_safe=D_SAFE, info_range=INFO_RANGE,
        lidar_costmap=False, lidar_topic="/point_cloud", visibility_cost=False,
        occlusion_aware=False, dynamic_obstacles=False, dynamic_viz=False,
        save_traj=None, show_occlusion_plot=True):
    global DETECTION_RANGE, UNCERTAINTY, N_SCEN, W_INFO
    global D_INFL, D_SAFE, INFO_RANGE, LIDAR_COSTMAP, VISIBILITY_COST, STATIC_AVOID
    global OCCLUSION_AWARE, OCC_TRACKER, OCC_SEGS_NOW
    global DYNAMIC_AVOID, DYN_TRACKER, DYN_NOW, DYN_DBG, DYN_PUB
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
    # Both flags acted on the removed oracle obstacle slots, so they are now no-ops.
    # Announced rather than silently ignored, matching how --visibility-cost is handled.
    if detect_range < float("inf"):
        print("[Controller] --detect-range no longer applies (it masked the removed oracle "
              "obstacle slots; LiDAR range is set by scan.max_range) — IGNORED.")
    if noise_std > 0.0:
        print("[Controller] --noise-std no longer applies (it corrupted the removed oracle "
              "obstacle slots) — IGNORED.")
    if visibility_cost:
        # The active-perception visibility term was a product of the removed persistent
        # occupancy grid; the circle model has no accumulated UNKNOWN volume to score.
        print("[Controller] --visibility-cost is no longer supported (needs the removed "
              "persistent grid) — IGNORED.")
    VISIBILITY_COST = False
    if lidar_costmap:
        STATIC_AVOID    = lidar_costmap
        OCCLUSION_AWARE = occlusion_aware and lidar_costmap   # occlusion terms need the map
        DYNAMIC_AVOID   = dynamic_obstacles and lidar_costmap  # clusters come from the same cloud
        from obstacle_circles import ObstacleCircles
        from occlusion_capsules import OcclusionCornerTracker, point_segment_distance_np
        from dynamic_clusters import DynamicClusterTracker, DynamicClusterPublisher
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
            if DYNAMIC_AVOID:
                DYN_TRACKER = DynamicClusterTracker(
                    assoc_radius=DYN_ASSOC, alpha=DYN_ALPHA, vel_beta=DYN_VEL_BETA,
                    ttl=DYN_TTL, min_hits=DYN_MIN_HITS, v_min=DYN_V_MIN,
                    min_dyn_hits=DYN_MIN_DYN_HITS, require_motion=DYN_REQUIRE_MOTION,
                    dyn_window=DYN_WINDOW, extent_frac=DYN_EXTENT_FRAC,
                    resegment_ratio=DYN_RESEG_RATIO)
                if dynamic_viz:
                    DYN_PUB = DynamicClusterPublisher()
                    if not DYN_PUB.start():
                        DYN_PUB = None
            feats = [f"static(D_SAFE={D_SAFE_HARD:.1f}m, circle-cover)"]
            if OCCLUSION_AWARE:  feats.append(f"occlusion(D_SAFE={D_SAFE_HARD:.1f}m, "
                                             f"v_target={V_TARGET:.1f}m/s, sightline)")
            if DYNAMIC_AVOID:    feats.append(
                f"dynamic(clustered, v_min={DYN_V_MIN:.1f}m/s, "
                f"v_target={V_TARGET:.1f}m/s, k={K_DYN})"
                + ("" if DYN_REQUIRE_MOTION else " [require_motion=OFF: every compact "
                                                 "cluster gets the expanding keep-out]"))
            print(f"[Controller] LiDAR map     : ON  (topic={lidar_topic}) — {' + '.join(feats)} "
                  f"in the MPPI cost; per-scan obstacle circles")
        else:
            STATIC_AVOID    = False
            OCCLUSION_AWARE = False
            print("[Controller] LiDAR map     : requested but unavailable (no rclpy) — "
                  "running WITHOUT LiDAR-based costs")
    elif occlusion_aware or dynamic_obstacles:
        if occlusion_aware:
            print("[Controller] Occlusion-aware : requested but needs --lidar-costmap — DISABLED")
        if dynamic_obstacles:
            print("[Controller] Dynamic obstacles: requested but needs --lidar-costmap "
                  "(the clusters come from the same cloud) — DISABLED")
    if dynamic_viz and not DYNAMIC_AVOID:
        print("[Controller] --dynamic-viz needs --dynamic-obstacles (there is nothing to "
              "draw without the detector) — DISABLED")
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
        f"Set BehaviorParameters Vector Observations = {OBS_SIZE} in the Unity Inspector "
        f"(every scene under Assets/Scenes) to match TaxiAgent.CollectObservations."
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
    min_clear = np.inf   # closest approach to any LiDAR circle SURFACE over the run
    collided  = False
    reached   = False
    traj      = []   # per-step ego pose: (t, x, y, theta, v, a_cmd, delta_cmd)
    # Union across the run (deduped by rounded world cell) of the static occluders and
    # occlusion boundaries the perception produced. Perception is frame-based (no memory),
    # so a single end-of-run snapshot only shows what is visible at the goal — accumulate to
    # show the full extent the ego actually reacted to.
    occ_seen  = {}   # static occluder points  → plotted as "static occluders"
    # Per-control-step snapshots (t, ego_x, ego_y, segments) of what the planner was
    # ACTUALLY constrained by at that instant. The plot samples these, so each drawn
    # keep-out belongs to one timestamp rather than to an episode-wide union.
    occ_frames = []
    # Per-step centres of the dynamic clusters the planner was constrained by. Fills the
    # obs_track slot of _save_trajectory, which used to carry the oracle obstacle feed —
    # so the plot's "dynamic obstacles" markers are now SENSED positions, not ground truth.
    dyn_track = []

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
            collided = min_clear < 0.0
            reached  = not terminal_steps.interrupted[0] and not collided
            episode_done = True
            break

        if len(decision_steps) == 0:
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            continue

        obs   = decision_steps.obs[0][0]          # shape (OBS_SIZE,)
        ep_steps += 1

        s, goal_xy = obs_to_state(obs, delta_actual, accel_actual)

        # Rebuild the LiDAR map from the latest cloud+pose. Returns from OTHER AIRCRAFT
        # are deliberately kept: they are the only signal about moving agents now that the
        # oracle obstacle slots are gone, so scrubbing them would blind the ego entirely.
        # ego_fwd (world a0,a1) orients the visibility ROI wedge — same axes as the mppi
        # rollout frame.
        if LIDAR_COSTMAP is not None:
            ego_fwd = np.array([np.cos(s[2]), np.sin(s[2])])
            LIDAR_COSTMAP.update(ego_fwd)
            # Detect + track occlusion boundaries ONCE per control step, exactly as the
            # MPC's run loop does, then hand the result to both the planner and the
            # plot recorder. Previously mppi() re-queried and the recorder called the
            # tracker again with None, so the tracker advanced twice per step and the
            # plot lagged the planner by one step.
            if OCCLUSION_AWARE and OCC_TRACKER is not None and LIDAR_COSTMAP.ready:
                _raw = LIDAR_COSTMAP.occlusion_segments(
                    ego_fwd=ego_fwd, fwd_half_angle_deg=OCC_FWD_HALF_ANGLE)
                _s = OCC_TRACKER.update(_raw, time.monotonic())
                if _s is not None and len(_s) and not OCC_USE_CAPSULES:
                    # Circle keep-out ⇒ collapse each boundary onto its corner (as the
                    # MPC does), so cost and plot use identical geometry.
                    # The OCC_USE_CAPSULES guard was missing here, so this collapsed
                    # EVERY boundary regardless of the setting — occlusion.use_capsules
                    # was dead, the keep-out was always a disc on the corner, and the
                    # capsules the plot and PhantomAgentVisualizer advertise never
                    # existed. mppi() has the same collapse, correctly gated, at the
                    # point where it builds occ_segs.
                    _s = _s.copy()
                    _s[:, 1, :] = _s[:, 0, :]
                OCC_SEGS_NOW = _s
                # Where the occlusion pipeline is losing boundaries. Each stage can zero the
                # set on its own, and all of them fail SILENTLY — the ego then drives the
                # occlusion-unaware trajectory while the flag still reads ON:
                #   raw=0    perception found no range jumps (scan cfg / min_jump / the
                #            fwd_half_angle wedge)
                #   trk=0    raw>0 but the tracker is withholding them until min_hits
                #   used=0   trk>0 but every boundary was gated out of the cost by
                #            query_r, K_OCC, or the goal-proximity drop
                # used is from the PREVIOUS solve (mppi runs after this), which is fine for
                # spotting a stuck-at-zero stage.
                if ep_steps % 10 == 1:
                    print(f"[DEBUG occ] raw={0 if _raw is None else len(_raw)} "
                          f"trk={0 if _s is None else len(_s)} used={OCC_USED_N} "
                          f"query_r={OCC_QUERY_R} k_occ={K_OCC} "
                          + ("" if OCC_GATE_DBG is None else
                             "goal_drop={}->{} (r_goal={:.1f}m, nearest boundary {:.1f}m "
                             "from goal)".format(*OCC_GATE_DBG))
                          + ("" if OCC_RANGE_DBG is None else
                             "  nearest_to_ego={:.1f}m corner=({:.1f},{:.1f}) "
                             "ego=({:.1f},{:.1f})".format(*OCC_RANGE_DBG)))
            else:
                OCC_SEGS_NOW = None

            # Cluster + track the moving obstacles ONCE per control step, same discipline
            # as the occlusion tracker above: advance the filter here, hand the confirmed
            # movers to mppi() through DYN_NOW. clusters() is memoised on the scan stamp,
            # so re-calling it at control rate against a ~1 Hz cloud is free.
            if DYNAMIC_AVOID and DYN_TRACKER is not None and LIDAR_COSTMAP.ready:
                _now = time.monotonic()
                _cl = LIDAR_COSTMAP.clusters(cell=DYN_CELL, min_points=DYN_MIN_POINTS,
                                             max_radius=DYN_MAX_RADIUS)
                DYN_TRACKER.update(_cl, _now)
                DYN_NOW = DYN_TRACKER.dynamic(_now)
                _sp = DYN_TRACKER.speeds()
                DYN_DBG = (0 if _cl is None else len(_cl), DYN_TRACKER.n_tracks,
                           0 if DYN_NOW is None else len(DYN_NOW),
                           max(_sp) if _sp else 0.0)
                # Where the dynamic pipeline is losing agents — every stage fails SILENTLY,
                # and the ego then drives the dynamic-UNAWARE trajectory with the flag ON:
                #   cl=0    clustering found nothing (min_points, or max_radius rejected the
                #           blob as scenery — a large aircraft close up can exceed it)
                #   trk     live tracks; if this climbs with cl steady, association is
                #           failing (assoc_radius too small for the per-scan travel) and
                #           every scan is minting new tracks that never confirm
                #   dyn=0   tracks exist but none cleared v_min/min_dyn_hits — either
                #           nothing is moving, or the centroid is too noisy to tell
                #   used    what actually entered the cost (previous solve; gated by
                #           query_r / k_dyn)
                if ep_steps % 10 == 1:
                    print(f"[DEBUG dyn] cl={DYN_DBG[0]} trk={DYN_DBG[1]} dyn={DYN_DBG[2]} "
                          f"used={0 if DYN_USED is None else len(DYN_USED)} "
                          f"v_max={DYN_DBG[3]:.1f}m/s (v_min={DYN_V_MIN:.1f})")
            else:
                DYN_NOW = None
            # Accumulate the per-scan obstacle-circle centres into the episode-wide
            # union (deduped by rounded world position) for the trajectory plot.
            if save_traj is not None and LIDAR_COSTMAP.ready:
                _circ = LIDAR_COSTMAP.circles()
                if _circ is not None:
                    for wx, wy, _r in _circ:
                        occ_seen[(round(float(wx), 1), round(float(wy), 1))] = None
        u_nom, mean = mppi(s, mean, goal_xy, u_prev)

        # Snapshot what the solve just above was constrained by, tagged with this
        # timestamp and ego pose. Taken AFTER mppi() so it is the post-gating set that
        # entered the cost — the same geometry _dist_to_occ measured against.
        # Feed the Unity Scene-view overlay with the POST-GATING set — what the cost above
        # actually constrained against, not what perception merely found. Published every
        # step including when empty, so the circles disappear the moment the planner stops
        # constraining them rather than lingering as a stale overlay.
        if DYN_PUB is not None:
            _viz = DYN_USED
            if _viz is not None and len(_viz) and not DYN_INCLUDE_AGE:
                # The viewer always folds v_target*age into the base radius. When the cost
                # is NOT doing that, send age = 0 so the drawn circle stays equal to the
                # constrained one — the whole reason the parameters travel in the message.
                _viz = np.asarray(_viz, float).copy()
                _viz[:, 3] = 0.0
            DYN_PUB.publish(_viz, V_TARGET, D_SAFE_HARD, DYN_GROW_HORIZON)

        if save_traj is not None and DYN_USED is not None and len(DYN_USED):
            for _c0, _c1, _r, _age in DYN_USED:
                dyn_track.append((ep_steps * DT, float(_c0), float(_c1)))

        if save_traj is not None and OCC_SEGS_USED is not None and len(OCC_SEGS_USED):
            occ_frames.append((ep_steps * DT, float(s[0]), float(s[1]),
                               np.array(OCC_SEGS_USED, dtype=float),
                               None if OCC_PLAN is None else OCC_PLAN.copy()))

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

        # Closest approach, now measured from PERCEPTION rather than the removed oracle
        # slots: clearance to the nearest LiDAR circle surface. This covers buildings and
        # other aircraft alike, since both are just returns. A negative value means the ego
        # drove inside a covering circle, which is what now counts as a collision.
        if LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
            _c = LIDAR_COSTMAP.circles()
            if _c is not None and len(_c):
                min_clear = min(min_clear, float(
                    np.min(np.hypot(_c[:, 0] - s[0], _c[:, 1] - s[1]) - _c[:, 2])))


        # Record ego pose this step (world frame: x=Unity Z, y=Unity X).
        traj.append((ep_steps * DT, s[0], s[1], s[2], s[3], a_cmd, delta_cmd))

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
    if DYN_PUB is not None:
        DYN_PUB.shutdown()

    verdict = "collision" if collided else ("reached" if reached else "timeout")

    if save_traj is not None and traj:
        # Overlay the run-wide UNION of static occluders and occlusion boundaries the
        # perception produced (not a single decaying end-of-run frame), so the plot shows
        # every wall/blind corner the ego reacted to.
        occ_pts = np.array(list(occ_seen.keys())) if occ_seen else None
        _save_trajectory(save_traj, verdict, goal_xy, np.asarray(traj),
                         # SENSED dynamic-obstacle centres (clustered + tracked from the
                         # LiDAR), not the removed oracle feed.
                         np.asarray(dyn_track) if dyn_track else None,
                         occ_pts, occ_frames=occ_frames,
                         # OCC_HORIZON, not the PLANNING horizon: the keep-out is
                         # capped at OCC_HORIZON in the cost, so drawing H_MPPI*DT
                         # would overstate the constraint by V_TARGET*(6.0-3.0)=9 m.
                         capsule_horizon=OCC_HORIZON,
                         show_occlusion=show_occlusion_plot)

    print(f"\n[Controller] steps={ep_steps:4d}  min_clearance={min_clear:5.2f} m "
          f"(LiDAR circle surfaces, target >= {D_SAFE_HARD:.1f} m)  → {verdict.upper()}")


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
    p.add_argument("--dynamic-obstacles", action="store_true",
                   help="Detect MOVING obstacles from the LiDAR: cluster the obstacle returns, "
                        "track the clusters across scans, and give each one whose estimated "
                        "speed clears dynamic_clusters.v_min an EXPANDING keep-out — radius "
                        "D_SAFE_HARD + cluster extent + occlusion.v_target*t, the same forward-"
                        "reachable-set circle the blind-corner phantoms get. Requires "
                        "--lidar-costmap. Set dynamic_clusters.require_motion=false in the "
                        "config to apply it to every compact cluster, moving or not.")
    p.add_argument("--dynamic-viz", action="store_true",
                   help="Publish the dynamic keep-outs on /dynamic_clusters "
                        "(std_msgs/Float32MultiArray) so DynamicAgentVisualizer.cs can draw "
                        "them in the Unity Scene view. The model parameters ride along in the "
                        "message, so the drawn circle is the one the planner used — no Unity-"
                        "side constants to keep in sync. Requires --dynamic-obstacles.")
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
        dynamic_obstacles=args.dynamic_obstacles,
        dynamic_viz=args.dynamic_viz,
        show_occlusion_plot=not args.no_occlusion_plot,
        save_traj=args.save_traj)
