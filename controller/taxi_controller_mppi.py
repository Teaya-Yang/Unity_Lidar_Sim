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

from taxi_config import CFG
_veh, _lim, _goal = CFG["vehicle"], CFG["limits"], CFG["goal"]
_dyn, _keep, _occ = CFG["dynamic_obstacles"], CFG["keepout"], CFG["occlusion"]
_trk, _scan       = CFG["occlusion_tracker"], CFG["scan"]
_mppi             = CFG["mppi"]

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
# Ego planform — PLOTTING ONLY. No cost term knows the ego has an extent; it is a point
# against d_safe_hard, and drawing the airframe is what makes that assumption visible.
EGO_LENGTH   = _veh["ego_length"]
EGO_SPAN     = _veh["ego_span"]
EGO_NOSE_FWD = _veh["ego_nose_fwd"]

H_MPPI    = _mppi["horizon"]    # planning horizon (steps)
K_MPPI    = _mppi["samples"]    # rollout samples
LAMBDA    = _mppi["lambda"]     # MPPI temperature
SIG_A     = _mppi["sig_a"]      # noise std for acceleration samples
SIG_D     = _mppi["sig_d"]      # noise std for steering samples

# MPPI stage costs
W_LAT, W_HEAD, W_V, W_CTRL = (_mppi["w_lat"], _mppi["w_head"],
                              _mppi["w_v"], _mppi["w_ctrl"])
W_OFF, W_PROG               = _mppi["w_off"], _mppi["w_prog"]
C_GOAL                      = _mppi["c_goal"]
C_GOAL_TERM                 = _mppi["c_goal_term"]
C_PROGRESS                  = _mppi["c_progress"]

V_TARGET       = _occ["v_target"]
OCC_USE_CAPSULES = _occ["use_capsules"]   # False ⇒ collapse each boundary to its corner
OCC_QUERY_R    = _occ["query_r"]
OCC_HORIZON    = _occ["horizon"]
OCC_T_GROW_MAX = _occ["t_grow_max"]   # cap on the FRS expansion time inside the stage cost
OCC_W_SOFT     = _occ.get("w_soft", 0.0)   # soft influence ring outside the hard keep-out
OCC_D_INFL     = _occ.get("d_infl", 0.0)
K_OCC          = _occ["k_occ"]
W_SIGHT        = _occ["w_sight"]
A_BRAKE_SIGHT  = _occ["a_brake_sight"]
V_SIGHT_FLOOR  = _occ["v_sight_floor"]

N_SCEN = _mppi["n_scen"]
W_INFO = _mppi["w_info"]
INFO_RANGE = _mppi["info_range"]

# Lister obstacles cost
LIDAR_COSTMAP  = None 
STATIC_AVOID   = False
D_SAFE_HARD    = _keep["d_safe_hard"]
W_HARD         = _keep["w_hard"]

INFEAS_DEPTH   = _keep["infeasible_depth"]
INFEAS_FRAC    = _keep["infeasible_frac"]
C_INFEAS       = W_HARD * INFEAS_DEPTH
STALL_V        = _keep["stall_v"]

SCAN_FOV_H, SCAN_FOV_V = _scan["fov_h"], _scan["fov_v"]
SCAN_RES_H, SCAN_RES_V = _scan["res_h"], _scan["res_v"]
SCAN_MAX_RANGE         = _scan["max_range"]
SENSOR_FWD, SENSOR_LAT = _scan["sensor_fwd"], _scan["sensor_lat"]
SENSOR_RANGE           = _scan["sensor_range"]
OCC_FWD_HALF_ANGLE     = _occ["fwd_half_angle"]
OCC_TRACK_ASSOC = _trk["assoc_radius"]
OCC_TRACK_ALPHA = _trk["alpha"]
OCC_TRACK_TTL   = _trk["ttl"]
OCC_TRACK_HITS  = _trk["min_hits"]
OCC_TRACKER     = None   # OcclusionCornerTracker, created in run()
OCC_PUB         = None   # OcclusionSegmentPublisher — feeds OcclusionSegmentVisualizer.cs
                         # so the Unity Scene view shows the boundaries THIS detector found
OCC_SEGS_NOW    = None   

OCC_RANGE_DBG   = None   # (min_ego_dist, corner_x, corner_y, ego_x, ego_y) for the nearest
                         # boundary that survived the goal drop — says whether the range gate
                         # is right to reject it.
OCC_GATE_DBG    = None   # (n_in, n_after_goal_drop, r_goal, min_goal_dist) from the LAST
                         # solve — lets the [DEBUG occ] trace name the gate that emptied the set.
OCC_PLAN        = None   # (H,3) [x, y, theta] the PLANNED future path from the last solve, in world
                         # (a0,a1). Stage k of this array is the pose the occlusion cost
                         # checked against radius r(t_k) — the pairing the plan figure draws.
OCC_INFEASIBLE  = False  # the LAST solve found no feasible rollout and braked. OCC_PLAN is
                         # then the braking rollout — what the ego actually executes, and the
                         # thing worth SEEING intersect the keep-outs, so it is still drawn.
OCC_SEGS_USED   = None   # (M,2,2) the segments the LAST solve actually constrained against —
                         # post-gating, so it is what the cost saw, not what perception found.
                         # The trajectory plot replays these per timestamp.
OCC_USED_N      = 0      # boundaries that actually entered the LAST solve's cost. Written by
                         # mppi(), read by the [DEBUG occ] trace in run(): if this is 0 while
                         # segments exist, the ego is provably driving the occlusion-UNAWARE
                         # trajectory no matter what --occlusion-aware was passed.

OCCLUSION_AWARE = False

RVIZ_PUB        = None   # RvizVisualizer — /viz/* MarkerArrays for RViz2
MPPI_ROLLOUTS   = None   # (n,H,3) sampled rollout paths [x, y, theta] from the LAST
                         # solve — the heading so the figures can place the SENSOR on each
                         # sample, not just the ego root. Kept only
                         # when --rviz-viz is on: it is pure visualisation, and holding
                         # every sample's path costs K*H*2 floats per step otherwise.
MPPI_COSTS      = None   # (n,) their total costs, for the colour ramp
MPPI_KEEP_ROLLOUTS = 0   # how many sampled paths mppi() records (0 = off)
MPPI_BEST       = None   # (path, cost) of the CHEAPEST sample of the LAST solve — the
                         # rollout the video draws. Recorded separately from MPPI_ROLLOUTS
                         # so it is available without --plot-rollouts, and exactly: it is
                         # the argmin over all K samples, not over a kept subset.
MPPI_TRACK_BEST = False  # set by --traj-video: costs one (K,H,2) scratch buffer
DYN_CELL       = CFG["dynamic_clusters"]["cell"]
DYN_MIN_POINTS = CFG["dynamic_clusters"]["min_points"]
DYN_MAX_RADIUS = CFG["dynamic_clusters"]["max_radius"]
DYN_ASSOC      = CFG["dynamic_clusters"]["assoc_radius"]
DYN_Q_ACCEL    = CFG["dynamic_clusters"]["q_accel"]
DYN_R_FRAC     = CFG["dynamic_clusters"]["r_frac"]
DYN_R_MIN      = CFG["dynamic_clusters"]["r_min"]
DYN_SIGMA_V0   = CFG["dynamic_clusters"]["sigma_v0"]
DYN_EXT_ALPHA  = CFG["dynamic_clusters"]["extent_alpha"]
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
DYN_LAST_STAMP = None    # ObstacleCircles.stamp of the scan last FUSED into the tracker.
                         #   Gates the Kalman correction to once per scan; see the control loop.
DYN_CL_N       = 0       # cluster count from that scan, carried across predict-only steps
DYN_SOURCE     = None    # measurement source feeding DYN_TRACKER: ObstacleCircles (grid
                         #   clustering, the default) or DetectionSource (PointPillars).
                         #   Both expose .clusters()/.stamp/.ready, so only the
                         #   MEASUREMENT differs — tracking and cost are shared.
DYN_PUB        = None    # DynamicClusterPublisher — feeds DynamicAgentVisualizer.cs so the
                         # Unity Scene view shows the SAME expanding circles the cost used
VISIBILITY_COST = False   # vestigial: --visibility-cost is a documented no-op

DETECTION_RANGE = float("inf")   # vestigial: it masked ORACLE obstacle slots beyond a radius,
                                 # and there are no such slots any more. Real sensor range is
                                 # now whatever the LiDAR reports (scan.max_range).

OBS_SIZE = 4 + 1 + 2   # 7: ego(4) + goal distance(1) + goal position(2)

rng = np.random.default_rng(42)

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


def sensor_xy(x, y, theta):
    """Ego-root world position + heading -> LIDAR world position, same (a0, a1) axes.

    The planner's state is the ego ROOT (see scan.sensor_fwd in config.yaml); the point
    cloud, the occlusion boundaries and the cluster centroids are all placed in the world
    from the SENSOR. This is the conversion between them, and it needs the heading: the
    offset is a body-frame vector, so it rotates with the aircraft.

    Axes: (a0, a1) = (Unity Z, Unity X) and theta is the Unity yaw, so body forward is
    (cos t, sin t) and body right is (-sin t, cos t) — consistent with _rollout_step,
    which integrates x along cos(theta) and y along sin(theta).
    """
    x = np.asarray(x, float); y = np.asarray(y, float); theta = np.asarray(theta, float)
    c, s = np.cos(theta), np.sin(theta)
    return (x + SENSOR_FWD * c - SENSOR_LAT * s,
            y + SENSOR_FWD * s + SENSOR_LAT * c)


def sensor_horizon(cx, cy, n=180):
    """Closed (n+1,2) circle of radius scan.sensor_range around the LIDAR at (cx, cy).

    The edge of what the sensor can report. Outside it the figure is not empty because the
    apron is clear, it is empty because nothing was measured — a distinction the keep-outs
    and the cluster markers cannot make on their own, and the one that decides whether a
    gap in the drawing is information.
    """
    a = np.linspace(0.0, 2.0 * np.pi, n + 1)
    return np.column_stack([cx + SENSOR_RANGE * np.cos(a),
                            cy + SENSOR_RANGE * np.sin(a)])


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
    global OCC_INFEASIBLE, MPPI_ROLLOUTS, MPPI_COSTS, MPPI_BEST
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
    # Same total, split by origin. cost_keep is the KEEP-OUT part only (occlusion capsules
    # + sensed movers); cost_nom is everything else (goal, heading, control effort, static
    # LiDAR surfaces). The feasibility test below is a statement about keep-out intrusion
    # depth, so it must read cost_keep — the total carries a goal-distance baseline of
    # ~(w_goal_run*H + w_goal_term)*d, which alone exceeds C_INFEAS far from the goal.
    cost_keep = np.zeros(K_MPPI)
    cost_nom  = np.zeros(K_MPPI)
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
            if goal_xy is not None:
                _rgc = D_SAFE_HARD + V_TARGET * H_MPPI * DT
                _gd = np.hypot(_op[:, 0] - goal_xy[0], _op[:, 1] - goal_xy[1])
                _seg = _seg[_gd > _rgc]

                OCC_GATE_DBG = (len(_op), int((_gd > _rgc).sum()), _rgc, float(_gd.min()))
            if len(_seg):
                _ego = np.array([[s0_fwd, s0_lat]])
                _d = np.array([point_segment_distance_np(_ego, s[0], s[1])[0]
                               for s in _seg])
                _in = _d < OCC_QUERY_R

                _j = int(np.argmin(_d))
                _opf = _seg[:, 0, :]
                OCC_RANGE_DBG = (float(_d.min()), float(_opf[_j, 0]), float(_opf[_j, 1]),
                                 float(s0_fwd), float(s0_lat))
                if _in.any():
                    _near = _seg[_in]
                    occ_segs = _near[np.argsort(_d[_in])[:K_OCC]]
                    OCC_USED_N = len(occ_segs)
                    OCC_SEGS_USED = occ_segs

    OCC_INFEASIBLE = False
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

    # Sampled rollout paths for the RViz overlay. Filled in the loop below so the drawn
    # lines are the SAMPLES the softmax weighted, not a re-simulation of them.
    MPPI_ROLLOUTS = MPPI_COSTS = MPPI_BEST = None
    paths = (np.empty((K_MPPI, H_MPPI, 3))
             if (MPPI_KEEP_ROLLOUTS > 0 or MPPI_TRACK_BEST) else None)

    for k in range(H_MPPI):
        st = _rollout_step(st, na[:, k, 0], na[:, k, 1])
        fwd, lat, th, vv = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
        if paths is not None:
            paths[:, k, 0], paths[:, k, 1], paths[:, k, 2] = fwd, lat, th

        u_k   = na[:, k, :]                                  # (K, 2)
        u_km1 = u_prev[None, :] if k == 0 else na[:, k - 1, :]

       
        _d_static = None
        if STATIC_AVOID and LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
            _d_static = LIDAR_COSTMAP.distance(fwd, lat)
        _c_nom = tcost.stage_cost(
            fwd, lat, th, vv, u_k, u_km1,
            goal_xy=(goal_fwd, goal_lat), v_des=v_des_eff, t_k=(k + 1) * DT,
            r_act=R_ACT, r_dact=R_DACT,
            w_goal_run=W_GOAL_RUN, w_head=W_HEAD, w_v=W_V,
            d_static=_d_static, d_safe_static=D_SAFE_HARD, w_static=W_HARD)
        cost += _c_nom
        cost_nom += _c_nom

        if occ_segs is not None:
            _c_occ = occlusion_stage_cost(
                _dist_to_occ(fwd, lat), vv, (k + 1) * DT,
                V_TARGET, D_SAFE_HARD, W_HARD, t_grow_max=OCC_T_GROW_MAX,
                w_soft=OCC_W_SOFT, d_infl=OCC_D_INFL,
                cost_current = cost, action = np.column_stack((fwd, lat)))
            cost += _c_occ
            cost_keep += _c_occ

        if dyn_set is not None:
            t_dyn = (k + 1) * DT
            for c0, c1, r_c, age in dyn_set:
                d_dyn = np.hypot(fwd - c0, lat - c1)

                d_base = D_SAFE_HARD + r_c
                _c_dyn = occlusion_stage_cost(
                    d_dyn, vv, t_dyn, V_TARGET, d_base, W_HARD,
                    # Same cap the /viz bubbles are drawn with (DYN_PUB.publish below).
                    # Uncapped, r_keep reached d_safe + r_c + v_target*H*dt = 15 + r_c + 56 m
                    # by the end of the horizon, so every rollout was inside it — and the
                    # enforced keep-out did not match the one RViz drew.
                    t_grow_max=DYN_GROW_HORIZON,
                    w_sight=W_SIGHT, a_brake=A_BRAKE_SIGHT, v_floor=V_SIGHT_FLOOR,  dyn = True, cost_current = cost)
                cost += _c_dyn
                cost_keep += _c_dyn



    print("MIN_cost_trajectory: ", cost.min())
    print("MAX_cost_trajectory: ", cost.max())

    cost += tcost.terminal_cost(st[:, 0], st[:, 1],
                                goal_xy=(goal_fwd, goal_lat), w_goal_term=W_GOAL_TERM)

    if paths is not None and MPPI_KEEP_ROLLOUTS > 0:
        keep = np.argsort(cost)[:MPPI_KEEP_ROLLOUTS]
        MPPI_ROLLOUTS, MPPI_COSTS = paths[keep], cost[keep]

    if paths is not None and MPPI_TRACK_BEST:
        # Cheapest sample on the TOTAL cost (terminal included — the same number the
        # softmax weights). Recorded even on an infeasible solve, which is returned from
        # below: that is the case where seeing the best available option matters most.
        _i_lo = int(np.argmin(cost))
        MPPI_BEST = (paths[_i_lo].copy(), float(cost[_i_lo]))

    # Feasibility is about keep-out intrusion ONLY. Testing the total here made the goal
    # baseline decide it: at ~22 m of cost per metre to the goal, every rollout is over
    # C_INFEAS = 1e3 whenever the goal is more than ~45 m away, so the planner braked for
    # the whole run regardless of what was actually in front of it.
    n_feasible = int((cost_keep < C_INFEAS).sum())
    if n_feasible <= INFEAS_FRAC * K_MPPI:
        print(f"[MPPI] INFEASIBLE: {n_feasible}/{K_MPPI} rollouts under {C_INFEAS:.2e} "
              f"(depth {INFEAS_DEPTH:.2f} m), min keep-out cost {cost_keep.min():.3e} "
              f"(nominal {cost_nom.min():.3e}, total {cost.min():.3e}) — braking")
        u_stop = np.array([A_MIN, 0.0])                 # max decel, hold wheel straight
        # Roll the braking command out over the horizon so the figure can show WHERE the
        # ego still ends up while stopping — the case where it clips the keep-outs is
        # exactly the one worth looking at, so this must not be left as a stale plan.
        _st = np.asarray(s0, float)[None, :]
        # (H,3) [x, y, theta]: the heading is carried so the figure can place the SENSOR
        # at each sampled stage. Taking it from the state is exact; differencing the
        # positions is not, because a stage's position was integrated with the PREVIOUS
        # stage's heading.
        _brake = np.empty((H_MPPI, 3))
        for _k in range(H_MPPI):
            _st = _rollout_step(_st, u_stop[0:1], u_stop[1:2])
            _brake[_k] = _st[0, :3]
        OCC_PLAN = _brake
        OCC_INFEASIBLE = True
        return u_stop, np.zeros((H_MPPI, 2))

    # Leave -cost.min() to avoid underflow, softmax to compute the weights
    w   = np.exp(-(cost - cost.min()) / LAMBDA)
    w  /= w.sum()
    opt = (w[:, None, None] * na).sum(axis=0)

    u_nom    = opt[0].copy()
    new_mean = np.vstack([opt[1:], opt[-1]])

    _st = np.tile(np.asarray(s0, float)[None, :], (1, 1))
    _plan = np.empty((H_MPPI, 3))            # [x, y, theta] — see the braking branch
    for _k in range(H_MPPI):
        _st = _rollout_step(_st, opt[_k:_k + 1, 0], opt[_k:_k + 1, 1])
        _plan[_k] = _st[0, :3]
    OCC_PLAN = _plan
    return u_nom, new_mean

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

# Colour per sampled horizon stage t_k, shared by the still figure, the rollout figure
# and the video so the same t_k is the same colour everywhere.
STAGE_COLORS = ["#00e5ff", "#ff00a0", "#ffd400", "#00ff7f",
                "#7c4dff", "#ff6d00", "#00b0ff", "#c6ff00"]


def _dyn_bubbles(boxes):
    """(N,3) [c0, c1, r] keep-out seeds from the ground-truth mover boxes published on
    /dynamic_obstacles: the box centre and the half-diagonal, so the disc encloses the
    whole footprint whatever its yaw."""
    b = (np.asarray(boxes, float).reshape(-1, 5)
         if boxes is not None and len(boxes) else np.empty((0, 5)))
    if not len(b):
        return np.empty((0, 3))
    r = 0.5 * np.hypot(b[:, 2], b[:, 3])
    return np.column_stack([b[:, 0], b[:, 1], r])


# How far a sensed track may sit from a ground-truth box and still be called the same
# object. Generous on purpose: the whole point of the comparison is to SHOW localisation
# error, so the threshold must not be so tight that a badly-but-genuinely tracked object
# is reported as lost. Scaled by the box's own size, floored for small objects.
TRACK_MATCH_R = 15.0   # [m]


def _match_tracked(fr):
    """(boxes (M,5), tracked (M,) bool, sensed (M,2) matched centre or nan).

    Greedy nearest-neighbour between the GROUND-TRUTH movers of one solve and the SENSED
    tracks it constrained against. `tracked=False` means Unity says an object is there and
    the planner has no keep-out for it — it is invisible to the cost, whatever the reason
    (never confirmed as moving, occluded, out of query_r, or crowded out of the k_dyn
    slots). That is the state this exists to make visible.
    """
    b = (np.asarray(fr.get("dyn_boxes"), float).reshape(-1, 5)
         if fr.get("dyn_boxes") is not None and len(fr["dyn_boxes"]) else np.empty((0, 5)))
    d = fr.get("dyn_set")
    s = (np.asarray(d, float).reshape(-1, 4)[:, :2]
         if d is not None and len(d) else np.empty((0, 2)))
    tracked = np.zeros(len(b), dtype=bool)
    matched = np.full((len(b), 2), np.nan)
    if not len(b) or not len(s):
        return b, tracked, matched

    free = list(range(len(s)))
    for i, (bx, by, sx, sy, _yaw) in enumerate(b):
        if not free:
            break
        r_gate = max(TRACK_MATCH_R, 0.5 * float(np.hypot(sx, sy)))
        dist = [float(np.hypot(s[j, 0] - bx, s[j, 1] - by)) for j in free]
        j_min = int(np.argmin(dist))
        if dist[j_min] <= r_gate:
            tracked[i] = True
            matched[i] = s[free[j_min]]
            free.pop(j_min)
    return b, tracked, matched


def _dyn_seeds(fr):
    """(N,3) [c0, c1, r] the keep-out circles of one solve expand from.

    The SENSED tracks — LiDAR clusters, Kalman-filtered and predicted forward to the solve
    time — because that is where the planner believes the traffic is and therefore what it
    actually constrained against. The ground-truth boxes are drawn too, as filled
    rectangles, so the offset between a box and its circle reads directly as the
    localisation error.

    A solve that tracked NOTHING returns nothing — no circles are drawn, because none were
    enforced. Distinguishing that from "this recording has no sensed field at all" is the
    reason the test is on the KEY and not on the value: falling back to ground truth when
    the tracker simply lost everything would draw keep-outs the planner never had, which
    is the exact confusion these figures exist to remove.
    """
    if "dyn_set" in fr:
        d = fr["dyn_set"]
        if d is None or not len(d):
            return np.empty((0, 3))
        return np.asarray(d, float).reshape(-1, 4)[:, :3]
    return _dyn_bubbles(fr.get("dyn_boxes"))


def _dyn_keepout_polys(fr, t_k):
    """Closed keep-out outlines at horizon time t_k, one per mover the solve constrained
    against — the same circles the cost enforces:

        r = D_SAFE_HARD + r_cluster + V_TARGET * min(t_k, DYN_GROW_HORIZON)

    The min() is the growth cap the dynamic term applies (and the occlusion term does not,
    its cap being longer than the horizon), so the outermost stages coincide once t_k
    passes it. That is the cap being visible, not a drawing bug.
    """
    from occlusion_capsules import capsule_polygon

    t_eff = t_k
    return [capsule_polygon((c0, c1), (c0, c1), D_SAFE_HARD + r_c + V_TARGET * t_eff)
            for c0, c1, r_c in _dyn_seeds(fr)]


def _dyn_markers(fr):
    """Centres (N,2) of the drawn keep-outs — the sensed track positions."""
    return _dyn_seeds(fr)[:, :2]


def _save_state_plot(stem, verdict, traj, goal_xy=None, t_mark=None, solve_ts=None):
    """SECOND figure, `<stem>_state.png`: how the ego's own state evolves over the run —
    one stacked panel per quantity, all sharing episode time. Separate from the map so
    each keeps its own natural aspect.

    Commanded vs actual is the point of the accel/steer panels: the planner emits a_cmd
    and delta_cmd, but the vehicle model applies a first-order lag and a rate + speed
    roll-off limit, so the two diverge exactly where the ego cannot follow the plan.

    `t_mark` [s] marks the solve the map figure draws; `solve_ts` ticks every solve.
    """
    import matplotlib.pyplot as plt

    t     = traj[:, 0]
    theta = traj[:, 3]
    v     = traj[:, 4]
    a_cmd, delta_cmd = traj[:, 5], traj[:, 6]
    # delta_act/a_act only exist on runs recorded after those columns were added.
    delta_act = traj[:, 7] if traj.shape[1] > 7 else None
    a_act     = traj[:, 8] if traj.shape[1] > 8 else None

    n_ax = 5 if goal_xy is not None else 4
    fig, axes = plt.subplots(n_ax, 1, figsize=(12, 2.0 * n_ax), sharex=True)
    ax_v, ax_a, ax_d, ax_h = axes[:4]

    ax_v.plot(t, v, "-", color="tab:blue", lw=1.4, label="v")
    ax_v.axhline(V_TARGET, ls="--", color="0.5", lw=1.0, label=f"V_TARGET {V_TARGET:.1f}")
    ax_v.set_ylabel("speed\n[m/s]")

    ax_a.plot(t, a_cmd, "-", color="tab:red", lw=1.2, label="a commanded")
    if a_act is not None:
        ax_a.plot(t, a_act, "-", color="k", lw=1.0, alpha=0.7, label="a actual (lagged)")
    ax_a.axhline(A_MAX, ls=":", color="0.5", lw=1.0)
    ax_a.axhline(A_MIN, ls=":", color="0.5", lw=1.0, label="a limits")
    ax_a.set_ylabel("accel\n[m/s²]")

    ax_d.plot(t, np.degrees(delta_cmd), "-", color="tab:green", lw=1.2, label="δ commanded")
    if delta_act is not None:
        ax_d.plot(t, np.degrees(delta_act), "-", color="k", lw=1.0, alpha=0.7,
                  label="δ actual (rate + roll-off limited)")
    ax_d.axhline(np.degrees(DELTA_LIM), ls=":", color="0.5", lw=1.0)
    ax_d.axhline(-np.degrees(DELTA_LIM), ls=":", color="0.5", lw=1.0, label="δ limits")
    ax_d.set_ylabel("steer\n[deg]")

    ax_h.plot(t, np.degrees(np.unwrap(theta)), "-", color="tab:purple", lw=1.2,
              label="heading θ")
    ax_h.set_ylabel("heading\n[deg]")

    # What the whole cost is chasing: a flat stretch here while v > 0 means the ego is
    # moving without making progress.
    if goal_xy is not None:
        ax_g = axes[4]
        ax_g.plot(t, np.hypot(traj[:, 1] - goal_xy[0], traj[:, 2] - goal_xy[1]),
                  "-", color="tab:brown", lw=1.2, label="distance to goal")
        ax_g.set_ylabel("d_goal\n[m]")

    axes[-1].set_xlabel("episode time t [s]")
    axes[0].set_title(f"{verdict} — ego internal state over the run")
    for a in axes:
        a.grid(True, alpha=0.3)
        a.set_xlim(t[0], t[-1])
        for ts in (solve_ts or []):
            a.axvline(ts, color="0.88", lw=0.5, zorder=0)
        if t_mark is not None:
            a.axvline(t_mark, color="k", ls="--", lw=1.2, zorder=1)
        a.legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=2)
    if t_mark is not None:
        axes[0].annotate(f"solve drawn on the map figure  t = {t_mark:.1f} s",
                         xy=(t_mark, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(3, 3), textcoords="offset points", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{stem}_state.png", dpi=120)
    plt.close(fig)


def _save_rollouts_plot(stem, verdict, fr, goal_xy=None, static_boxes=None,
                        max_frames=6):
    """`{stem}_rollouts.png`: every recorded SAMPLED rollout of ONE solve, coloured by
    total cost, with the chosen plan and that solve's keep-outs on top.

    Same solve and same keep-out geometry as `{stem}.png` — that figure shows the one
    plan the softmax produced, this one shows the candidate set it produced it from.
    Only the rollouts mppi() kept are available (see --plot-rollouts), and they are the
    CHEAPEST N, so the spread drawn is the good tail of the distribution, not all K.

    LIKE THE MAP FIGURE, every ego-side thing here is the LIDAR — the cloud included.
    mppi() records each sample as [x, y, theta], so the same body-frame offset can be
    rotated onto every sample at every stage; the cloud is therefore the set of paths the
    SENSOR would trace, which is the set that can be read against keep-outs seeded by
    sensor data. On a turning sample the converted path is not a shifted copy of the root
    path — it bows outward — which is real and is the reason this is worth doing properly.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from occlusion_capsules import capsule_polygon
    from static_obstacles import box_polygon, aircraft_polygon

    rollouts = fr.get("rollouts")
    if rollouts is None or not len(rollouts):
        return False
    rollouts = np.asarray(rollouts, float)
    if rollouts.shape[-1] >= 3:
        # (n,H,3) [x, y, theta] -> the sensor path of every sample.
        _rx, _ry = sensor_xy(rollouts[:, :, 0], rollouts[:, :, 1], rollouts[:, :, 2])
        rollouts = np.stack([_rx, _ry], axis=-1)
    # else: a recording from before the heading was carried — left at the ego root, which
    # is visibly ~14 m off the markers rather than silently wrong.
    costs = fr.get("rollout_costs")
    costs = None if costs is None else np.asarray(costs, float)

    plan = fr["plan"]                                   # (H,3) ego-root [x, y, theta]
    plan_x, plan_y = sensor_xy(plan[:, 0], plan[:, 1], plan[:, 2])
    tk = (np.arange(len(plan)) + 1) * DT
    segs = (np.asarray(fr["segs"], float).reshape(-1, 2, 2)
            if fr.get("segs") is not None and len(fr["segs"]) else np.empty((0, 2, 2)))
    dyn_c = _dyn_markers(fr)

    fig, ax = plt.subplots(figsize=(11, 11))

    sb = (np.asarray(static_boxes, float).reshape(-1, 5)
          if static_boxes is not None and len(static_boxes) else np.empty((0, 5)))
    for i, (bx, by, bsx, bsy, byaw) in enumerate(sb):
        poly = box_polygon(bx, by, bsx, bsy, byaw)
        ax.fill(poly[:, 0], poly[:, 1], color="0.80", ec="0.30", lw=1.0, zorder=0,
                label="static obstacles (Unity)" if i == 0 else None)

    db = (np.asarray(fr["dyn_boxes"], float).reshape(-1, 5)
          if fr.get("dyn_boxes") is not None and len(fr["dyn_boxes"]) else np.empty((0, 5)))
    for i, (bx, by, bsx, bsy, byaw) in enumerate(db):
        poly = box_polygon(bx, by, bsx, bsy, byaw)
        ax.fill(poly[:, 0], poly[:, 1], color="tab:orange", alpha=0.45, ec="tab:orange",
                lw=1.2, zorder=2,
                label="dynamic objects, ground truth (Unity)" if i == 0 else None)

    # The rollouts themselves, as one LineCollection — thousands of ax.plot calls would
    # dominate the render time and the legend.
    lc = LineCollection(rollouts, linewidths=0.7, alpha=0.55, zorder=3)
    if costs is not None and len(costs) == len(rollouts) and np.ptp(costs) > 0:
        # Clip the ramp at the 95th percentile: one infeasible sample at w_hard = 1e5
        # would otherwise flatten every feasible rollout into a single colour.
        hi = float(np.percentile(costs, 95))
        lo = float(costs.min())
        lc.set_array(np.clip(costs, lo, max(hi, lo + 1e-9)))
        lc.set_cmap("viridis")
        cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("rollout total cost (clipped at p95)")
    else:
        lc.set_color("tab:blue")
    ax.add_collection(lc)
    # One proxy handle so the legend names the cloud without 200 entries in it.
    ax.plot([], [], "-", color="tab:blue", lw=1.0, alpha=0.7,
            label=f"{len(rollouts)} sampled rollouts (cheapest kept)")

    ax.plot(plan_x, plan_y, "-", color="k", lw=2.5, zorder=6,
            label=("braking rollout — no feasible plan" if fr.get("infeasible") else
                   "chosen plan (softmax of the samples)"))

    # Keep-outs at t_k = 0 and at the same sampled stages as the main figure, so the two
    # can be read side by side.
    for seg in segs:
        poly = capsule_polygon(seg[0], seg[1], D_SAFE_HARD)
        ax.plot(poly[:, 0], poly[:, 1], "-", color="w", lw=1.8, alpha=0.95, zorder=4)
    for poly in _dyn_keepout_polys(fr, 0.0):
        ax.plot(poly[:, 0], poly[:, 1], "--", color="w", lw=1.6, alpha=0.95, zorder=4)

    stages = np.linspace(0, len(plan) - 1, min(max_frames, len(plan))).round().astype(int)
    for si, k in enumerate(dict.fromkeys(stages.tolist())):
        col = STAGE_COLORS[si % len(STAGE_COLORS)]
        t_k = float(tk[k])
        r_k = D_SAFE_HARD + V_TARGET * t_k
        for seg in segs:
            poly = capsule_polygon(seg[0], seg[1], r_k)
            ax.plot(poly[:, 0], poly[:, 1], "-", color=col, lw=1.5, alpha=0.9, zorder=4)
        for poly in _dyn_keepout_polys(fr, t_k):
            ax.plot(poly[:, 0], poly[:, 1], "--", color=col, lw=1.4, alpha=0.9, zorder=4)

    for seg in segs:
        ax.plot(seg[:, 0], seg[:, 1], "-", color="k", lw=2.5, zorder=7)
    for i, (c0, c1) in enumerate(dyn_c):
        ax.plot(c0, c1, "x", color="k", ms=7, mew=1.5, zorder=7,
                label="sensed mover centre (keep-out seed)" if i == 0 else None)

    _hull = aircraft_polygon(fr["ex"], fr["ey"], fr["eth"],
                             EGO_LENGTH, EGO_SPAN, EGO_NOSE_FWD)
    ax.fill(_hull[:, 0], _hull[:, 1], facecolor="w", alpha=0.55, edgecolor="k",
            lw=1.0, zorder=7, label="ego airframe")
    _e0x, _e0y = sensor_xy(fr["ex"], fr["ey"], fr["eth"])
    ax.plot(_e0x, _e0y, "o", mfc="w", mec="k", ms=8, mew=1.2, zorder=8,
            label="ego (LiDAR) at $t_k$ = 0")
    if goal_xy is not None:
        ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=18, mec="k", zorder=8,
                label="goal")

    # Zoom on the sample cloud plus the plan: unlike the main figure there is no executed
    # trajectory to bound, and the cloud IS the subject, so it must not be clipped.
    allx = np.concatenate([rollouts[:, :, 0].ravel(), plan_x, [_e0x]])
    ally = np.concatenate([rollouts[:, :, 1].ravel(), plan_y, [_e0y]])
    margin = 0.15 * max(np.ptp(allx), np.ptp(ally), 1.0) + 5.0
    cx, cy = 0.5 * (allx.min() + allx.max()), 0.5 * (ally.min() + ally.max())
    half_x = 0.5 * np.ptp(allx) + margin
    half_y = max(0.5 * np.ptp(ally) + margin, 0.25 * half_x)
    ax.set_xlim(cx - half_x, cx + half_x); ax.set_ylim(cy - half_y, cy + half_y)
    ax.set_aspect("equal", adjustable="box")
    fig.set_size_inches(12.0, float(np.clip(12.0 * half_y / half_x, 4.0, 12.0)))
    ax.set_xlabel("x  (Unity Z) [m]"); ax.set_ylabel("y  (Unity X) [m]")
    ax.set_title(f"{verdict} — MPPI sample cloud at t = {fr['t']:.1f} s "
                 f"({len(rollouts)} rollouts over {len(plan)} steps = "
                 f"{len(plan) * DT:.1f} s horizon)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=8, borderaxespad=0.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}_rollouts.png", dpi=120)
    plt.close(fig)
    return True


def _draw_solve_map(ax, fr, verdict, traj, goal_xy=None, static_boxes=None,
                    max_frames=6, t_now=None):
    """Draw ONE solve on `ax`: the executed trajectory, that solve's predicted rollout,
    and the expanding keep-out (occlusion capsules + swept mover tubes) at a handful of
    sampled horizon timestamps t_k.

    THE EGO IS DRAWN AT THE LIDAR, not at the ego root the planner integrates, and the
    airframe outline is drawn around it. Everything the ego is compared against here comes
    from the sensor (the occlusion boundaries and the mover centroids are world-placed
    from /laser_scan_pose), so plotting the root put every apparent clearance
    scan.sensor_fwd ~ 14 m out along the fuselage. The keep-outs are still the ones the
    COST enforced, and the cost enforces them against the ROOT — so the sensor marker can
    sit inside a capsule on a solve the planner called feasible. That is the real state of
    affairs (the nose was inside it all along) and showing it is the point; the fix is to
    carry the offset into the cost, which is a controller change, not a plotting one.

    Shared by the single-frame figure and the video, so both show exactly the same
    geometry. `t_now` [s], when given (video), clips the executed trajectory to what had
    already been driven at this solve instead of drawing the whole run.

    Returns (segs, dyn_c, stages) for the caller's summary line. The caller owns the axis
    limits, aspect and legend — the video needs those fixed across frames.
    """
    from occlusion_capsules import capsule_polygon
    from static_obstacles import box_polygon, aircraft_polygon

    def _ego_outline(ex, ey, eth):
        """Planform at an EGO-ROOT pose — the frame the dimensions are measured in."""
        return aircraft_polygon(ex, ey, eth, EGO_LENGTH, EGO_SPAN, EGO_NOSE_FWD)

    # traj columns are t,x,y,theta,... so column 3 is the heading that rotates the offset.
    x, y = sensor_xy(traj[:, 1], traj[:, 2], traj[:, 3])
    if t_now is not None:
        n = max(int(np.searchsorted(traj[:, 0], t_now, side="right")), 1)
        x, y = x[:n], y[:n]

    plan = fr["plan"]                                   # (H,3) ego-root [x, y, theta]
    # The same rollout expressed at the SENSOR. Not a rigid shift of the root path: the
    # offset rotates with the aircraft, so on a turn the sensor traces a WIDER arc.
    plan_x, plan_y = sensor_xy(plan[:, 0], plan[:, 1], plan[:, 2])
    tk = (np.arange(len(plan)) + 1) * DT
    segs = (np.asarray(fr["segs"], float).reshape(-1, 2, 2)
            if fr.get("segs") is not None and len(fr["segs"]) else np.empty((0, 2, 2)))
    # Movers as Unity publishes them on /dynamic_obstacles, at THIS solve. Each seeds an
    # expanding CIRCLE of radius D_SAFE_HARD + r_obj + V_TARGET*t_k — the same growth law
    # as the occlusion capsules, so they are drawn at the same sampled t_k, same colour.
    dyn_c = _dyn_markers(fr)

    # Static obstacles as Unity reports them (StaticObstaclePublisher.cs).
    sb = (np.asarray(static_boxes, float).reshape(-1, 5)
          if static_boxes is not None and len(static_boxes) else np.empty((0, 5)))
    for i, (bx, by, bsx, bsy, byaw) in enumerate(sb):
        poly = box_polygon(bx, by, bsx, bsy, byaw)
        ax.fill(poly[:, 0], poly[:, 1], color="0.80", ec="0.30", lw=1.0, zorder=0,
                label="static obstacles (Unity)" if i == 0 else None)

    # Ground-truth movers where they were AT the drawn solve (t_k = 0), not at the end.
    # An object with no sensed track behind it is outlined in red: Unity says it is there
    # and the planner is carrying no keep-out for it.
    db, tracked, matched = _match_tracked(fr)
    _seen_lbl = {True: False, False: False}
    for i, (bx, by, bsx, bsy, byaw) in enumerate(db):
        poly = box_polygon(bx, by, bsx, bsy, byaw)
        ok = bool(tracked[i])
        lbl = None
        if not _seen_lbl[ok]:
            lbl = ("dynamic objects, ground truth (Unity)")
            _seen_lbl[ok] = True
        ax.fill(poly[:, 0], poly[:, 1], color="tab:orange", alpha=0.45,
                ec=("tab:orange" if ok else "red"), lw=(1.2 if ok else 2.2), zorder=2,
                label=lbl)
        # Truth → estimate: the length of this tie IS the localisation error.
        if ok:
            ax.plot([bx, matched[i, 0]], [by, matched[i, 1]], "-", color="0.35", lw=0.9,
                    zorder=2)

    ax.plot(x, y, "-", color="0.65", lw=1.2, zorder=1,
            label="executed trajectory, LiDAR"
                  + (" (so far)" if t_now is not None else ""))
    ax.plot(x[0], y[0], "o", color="tab:green", ms=9, zorder=3, label="start")
    if t_now is None:
        ax.plot(x[-1], y[-1], "s", color="tab:red", ms=9, zorder=3, label="end")
    if goal_xy is not None:
        ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=18, mec="k", zorder=3,
                label="goal")

    ax.plot(plan_x, plan_y, "-", color="k", lw=2.0, zorder=4,
            label=("braking rollout, LiDAR — no feasible plan "
                   f"(solve at t = {fr['t']:.1f} s)" if fr.get("infeasible") else
                   f"predicted rollout, LiDAR (solve at t = {fr['t']:.1f} s)"))

    # Sampled predicted timestamps along that rollout.
    # t_k = 0: the ego pose at this solve, with the un-expanded keep-out.
    for seg in segs:
        poly = capsule_polygon(seg[0], seg[1], D_SAFE_HARD)
        ax.plot(poly[:, 0], poly[:, 1], "-", color="w", lw=1.8, alpha=0.95, zorder=1)
    for poly in _dyn_keepout_polys(fr, 0.0):
        ax.plot(poly[:, 0], poly[:, 1], "--", color="w", lw=1.6, alpha=0.95, zorder=1)
    _e0x, _e0y = sensor_xy(fr["ex"], fr["ey"], fr["eth"])
    # Sensor horizon, centred on the LIDAR (not the ego root) because that is what it is
    # measured from. Drawn under everything and left out of the zoom bounds — it is 250 m
    # across and would shrink the ego to a few pixels if the window had to contain it.
    _hz = sensor_horizon(_e0x, _e0y)
    ax.plot(_hz[:, 0], _hz[:, 1], "-", color="0.55", lw=1.0, alpha=0.7, zorder=0,
            label=f"LiDAR range ({SENSOR_RANGE:.0f} m)")
    _hull = _ego_outline(fr["ex"], fr["ey"], fr["eth"])
    ax.fill(_hull[:, 0], _hull[:, 1], facecolor="w", alpha=0.55, edgecolor="k",
            lw=1.0, zorder=4, label="ego airframe")
    ax.plot(_e0x, _e0y, "o", mfc="w", mec="k", ms=6, mew=0.9, zorder=5,
            label=f"$t_k$ = 0.0 s   r = {D_SAFE_HARD:.0f} m")

    stages = np.linspace(0, len(plan) - 1, min(max_frames, len(plan))).round().astype(int)
    for si, k in enumerate(dict.fromkeys(stages.tolist())):
        col = STAGE_COLORS[si % len(STAGE_COLORS)]
        t_k = float(tk[k])
        # The occlusion keep-out AT THIS STAGE: a worst-case hidden agent leaving the
        # boundary at t=0 at V_TARGET can be anywhere within D_SAFE_HARD + V_TARGET·t_k,
        # so each sampled timestamp gets its own, larger, capsule.
        r_k = D_SAFE_HARD + V_TARGET * t_k
        for seg in segs:
            poly = capsule_polygon(seg[0], seg[1], r_k)
            ax.plot(poly[:, 0], poly[:, 1], "-", color=col, lw=1.8, alpha=0.95, zorder=1)
        # Same t_k, same colour, dashed: the mover keep-out grows from the object's own
        # radius, so r = D_SAFE_HARD + r_obj + V_TARGET·t_k.
        for poly in _dyn_keepout_polys(fr, t_k):
            ax.plot(poly[:, 0], poly[:, 1], "--", color=col, lw=1.6, alpha=0.95, zorder=1)
        # The airframe as it will be oriented at this stage — outline only, so stages
        # further down the horizon overlap without hiding each other or the keep-outs.
        hull = _ego_outline(plan[k, 0], plan[k, 1], plan[k, 2])
        ax.plot(hull[:, 0], hull[:, 1], "-", color=col, lw=1.0, alpha=0.85, zorder=4)
        ax.plot(plan_x[k], plan_y[k], "o", mfc=col, mec="k", ms=6, mew=0.9, zorder=5,
                label=f"$t_k$ = {t_k:.1f} s   r = {r_k:.0f} m")

    for seg in segs:
        ax.plot(seg[:, 0], seg[:, 1], "-", color="k", lw=2.5, zorder=6)
        ax.plot(seg[0, 0], seg[0, 1], ".", color="k", ms=8, zorder=6)

    for i, (c0, c1) in enumerate(dyn_c):
        ax.plot(c0, c1, "x", color="k", ms=7, mew=1.5, zorder=6,
                label="sensed mover centre (keep-out seed)" if i == 0 else None)

    ax.set_xlabel("x  (Unity Z) [m]"); ax.set_ylabel("y  (Unity X) [m]")
    # State the reference point ON the figure: the keep-outs are enforced at the ego root
    # while the markers are the sensor, and a reader cannot tell that from the geometry.
    ax.text(0.01, 0.01,
            f"ego markers = LiDAR (laser_link, {SENSOR_FWD:+.2f} m fwd / "
            f"{SENSOR_LAT:+.2f} m lat of the ego root);  keep-outs enforced at the ego root",
            transform=ax.transAxes, fontsize=7, color="0.35", va="bottom", zorder=7)
    _what = " + ".join((["occlusion"] if len(segs) else []) +
                       (["dynamic-obstacle"] if len(dyn_c) else []))
    ax.set_title(f"{verdict} — predicted rollout at t = {fr['t']:.1f} s with the "
                 f"expanding {_what} keep-out per sampled $t_k$"
                 if (len(segs) or len(dyn_c)) else
                 f"{verdict} — executed trajectory + predicted rollout at "
                 f"t = {fr['t']:.1f} s")
    ax.grid(True, alpha=0.3)
    return segs, dyn_c, stages


def _save_tracking_plot(stem, verdict, frames, traj, static_boxes=None):
    """`{stem}_tracking.png`: WHERE and WHEN the ego loses each mover.

    Two panels sharing the run:

    TOP, map — each ground-truth mover's true path, drawn green where a sensed track was
    behind it at that solve and red where none was, with a marker at every transition. A
    red stretch is a stretch of the run in which Unity had an object on the map and the
    planner was carrying no keep-out for it. The matched sensed positions are dotted on
    top, so the gap between the two paths is the localisation error along the way.

    BOTTOM, timeline — the same tracked/untracked state against episode time, over the
    ground-truth distance from the ego. Reading the two together answers why a track was
    lost: a loss at ~query_r is a gating drop, a loss at close range with the ego behind a
    wall is an occlusion drop, and a loss while the object is stationary is the
    require_motion test.

    Movers are identified by their index in the /dynamic_obstacles message, which Unity
    publishes in a stable order; the sensed side has no ids, so truth and estimate are
    associated per solve by nearest neighbour (_match_tracked).
    """
    import matplotlib.pyplot as plt
    from static_obstacles import box_polygon

    fr_dyn = [f for f in frames
              if f.get("dyn_boxes") is not None and len(f["dyn_boxes"])]
    if not fr_dyn:
        return False

    n_mov = max(len(np.asarray(f["dyn_boxes"], float).reshape(-1, 5)) for f in fr_dyn)
    t = np.array([f["t"] for f in fr_dyn])
    # Per mover: true xy, matched sensed xy, tracked flag, distance from the ego. NaN
    # wherever that mover was absent from the message at that solve.
    gt = np.full((n_mov, len(fr_dyn), 2), np.nan)
    est = np.full((n_mov, len(fr_dyn), 2), np.nan)
    trk = np.zeros((n_mov, len(fr_dyn)), dtype=bool)
    d_ego = np.full((n_mov, len(fr_dyn)), np.nan)
    for j, f in enumerate(fr_dyn):
        b, tracked, matched = _match_tracked(f)
        for i in range(len(b)):
            gt[i, j] = b[i, :2]
            est[i, j] = matched[i]
            trk[i, j] = tracked[i]
            d_ego[i, j] = np.hypot(b[i, 0] - f["ex"], b[i, 1] - f["ey"])

    fig, (ax, ax_t) = plt.subplots(
        2, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [2.4, 1.0]})

    sb = (np.asarray(static_boxes, float).reshape(-1, 5)
          if static_boxes is not None and len(static_boxes) else np.empty((0, 5)))
    for i, b in enumerate(sb):
        poly = box_polygon(*b)
        ax.fill(poly[:, 0], poly[:, 1], color="0.85", ec="0.60", lw=0.8, zorder=0,
                label="static obstacles (Unity)" if i == 0 else None)
    ax.plot(traj[:, 1], traj[:, 2], "-", color="0.55", lw=1.2, zorder=1,
            label="ego trajectory")
    ax.plot(traj[0, 1], traj[0, 2], "o", color="tab:green", ms=8, zorder=3)

    n_lost = 0
    for i in range(n_mov):
        ok = np.isfinite(gt[i, :, 0])
        if not ok.any():
            continue
        # Colour the TRUE path by tracking state, segment by segment: this is the "where".
        for j in range(len(fr_dyn) - 1):
            if not (ok[j] and ok[j + 1]):
                continue
            ax.plot(gt[i, j:j + 2, 0], gt[i, j:j + 2, 1], "-",
                    color=("tab:green" if trk[i, j] else "red"), lw=2.0, zorder=2)
        ax.plot(est[i, :, 0], est[i, :, 1], ":", color="tab:blue", lw=1.0, zorder=2)
        # Transitions. A tracked -> untracked edge is the moment of the loss.
        edges = np.flatnonzero(np.diff(trk[i].astype(int)) != 0)
        for e in edges:
            lost = trk[i, e] and not trk[i, e + 1]
            n_lost += int(lost)
            ax.plot(gt[i, e + 1, 0], gt[i, e + 1, 1], "x" if lost else "+",
                    color=("red" if lost else "tab:green"), ms=13, mew=2.5, zorder=5)
            if lost:
                ax.annotate(f"lost t={t[e + 1]:.1f}s\nd={d_ego[i, e + 1]:.0f}m",
                            xy=(gt[i, e + 1, 0], gt[i, e + 1, 1]),
                            xytext=(8, 8), textcoords="offset points", fontsize=8,
                            color="red")
        ax.annotate(f"#{i}", xy=(gt[i, ok.argmax(), 0], gt[i, ok.argmax(), 1]),
                    xytext=(4, -12), textcoords="offset points", fontsize=9, color="0.2")

    ax.plot([], [], "-", color="tab:green", lw=2.0, label="mover, TRACKED")
    ax.plot([], [], "-", color="red", lw=2.0, label="mover, not tracked")
    ax.plot([], [], ":", color="tab:blue", lw=1.2, label="sensed (estimated) position")
    ax.plot([], [], "x", color="red", ms=11, mew=2.5, label="track lost here")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x  (Unity Z) [m]"); ax.set_ylabel("y  (Unity X) [m]")
    ax.set_title(f"{verdict} — mover tracking: where the ego stops seeing each object "
                 f"({n_lost} loss event{'s' if n_lost != 1 else ''})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)
    ax.grid(True, alpha=0.3)

    for i in range(n_mov):
        ok = np.isfinite(d_ego[i])
        if not ok.any():
            continue
        ax_t.plot(t[ok], d_ego[i][ok], "-", color="0.75", lw=1.0, zorder=1)
        for state, col in ((True, "tab:green"), (False, "red")):
            m = ok & (trk[i] == state)
            ax_t.plot(t[m], d_ego[i][m], ".", color=col, ms=4, zorder=2)
    ax_t.axhline(DYN_QUERY_R, ls="--", color="tab:purple", lw=1.2,
                 label=f"query_r = {DYN_QUERY_R:.0f} m (gating range)")
    ax_t.set_xlabel("episode time t [s]")
    ax_t.set_ylabel("ground-truth distance\nfrom ego [m]")
    ax_t.set_title("green = a keep-out was enforced for it, red = none", fontsize=10)
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{stem}_tracking.png", dpi=120)
    plt.close(fig)
    return True


def _ffmpeg_exe():
    """Path to an ffmpeg binary, or None. PATH first, then the static one imageio-ffmpeg
    ships — a venv install is enough to get mp4 out without root."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _save_trajectory_video(stem, verdict, frames, traj, goal_xy=None, static_boxes=None,
                           max_frames=6, fps=10, stride=1):
    """`{stem}.mp4` (or `.gif` without ffmpeg): the map figure replayed over every
    recorded solve, so the keep-outs can be watched expanding and being dodged as the run
    progresses. Returns (path, n_frames) or None.

    The window is fixed over the whole run — a per-frame window would make the keep-outs
    appear to breathe when it is the zoom moving, not them.

    WHY THIS DOES NOT JUST LOOP OVER `_draw_solve_map`. That is what a frame costs when
    the whole figure is rebuilt and re-rasterised per frame:

        ax.clear + redraw   ~80 ms      savefig (full canvas)  ~130 ms

    i.e. ~70 s for a 30 s run, nearly all of it re-rendering scenery that never moves.
    Instead everything static (obstacles, the full route, grid, legend) is drawn ONCE and
    cached as a pixel background; each frame restores that background, redraws only the
    ~9 artists that actually change, and pipes the raw RGBA buffer straight to ffmpeg —
    bypassing savefig, which would re-render the whole figure and undo the saving.
    ~10 ms/frame, a 20x speed-up.

    The cost of the split is that the artists here are built by hand rather than by
    `_draw_solve_map`, so the two could drift apart cosmetically. They cannot drift
    GEOMETRICALLY: both take the drawn shapes from the same _dyn_keepout_polys /
    capsule_polygon / _dyn_bubbles helpers and the same STAGE_COLORS.
    """
    import subprocess
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from occlusion_capsules import capsule_polygon
    from static_obstacles import box_polygon, aircraft_polygon

    frames = frames[::max(1, int(stride))]
    if not frames:
        return None

    # Fixed window: the ego's whole executed extent plus every plan drawn in the video.
    # Bound the SENSOR paths, since that is what the frames draw — and bound the rollout
    # the frames actually show, which is the cheapest SAMPLE ("best") where one was
    # recorded, not the softmax mean.
    _tx, _ty = sensor_xy(traj[:, 1], traj[:, 2], traj[:, 3])
    def _drawn(f):
        p = f.get("best")
        p = np.asarray(f["plan"] if p is None or not len(p) else p, float)
        return (sensor_xy(p[:, 0], p[:, 1], p[:, 2]) if p.shape[-1] >= 3
                else (p[:, 0], p[:, 1]))
    _px = [_drawn(f) for f in frames]
    # The horizon ring moves with the ego and the window is fixed for the whole video, so
    # it has to bound the ring at every solve — the ego track inflated by the sensor range.
    _hx = np.array([sensor_xy(f["ex"], f["ey"], f["eth"])[0] for f in frames])
    _hy = np.array([sensor_xy(f["ex"], f["ey"], f["eth"])[1] for f in frames])
    allx = np.concatenate([_tx] + [p[0] for p in _px]
                          + [_hx - SENSOR_RANGE, _hx + SENSOR_RANGE])
    ally = np.concatenate([_ty] + [p[1] for p in _px]
                          + [_hy - SENSOR_RANGE, _hy + SENSOR_RANGE])
    cx, cy = 0.5 * (allx.min() + allx.max()), 0.5 * (ally.min() + ally.max())
    half_x = 0.5 * np.ptp(allx) + 100.0
    half_y = max(0.5 * np.ptp(ally) + 100.0, 0.25 * half_x)

    fig, ax = plt.subplots(figsize=(12.0, float(np.clip(12.0 * half_y / half_x,
                                                        4.0, 12.0))))
    # subplots_adjust, not tight_layout: the canvas size must be identical for every
    # frame — ffmpeg is fed a fixed frame geometry.
    fig.subplots_adjust(left=0.07, right=0.72, top=0.88, bottom=0.09)
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy - half_y, cy + half_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x  (Unity Z) [m]"); ax.set_ylabel("y  (Unity X) [m]")
    ax.grid(True, alpha=0.3)

    # ── static layer: rasterised once into the cached background ──────────────
    sb = (np.asarray(static_boxes, float).reshape(-1, 5)
          if static_boxes is not None and len(static_boxes) else np.empty((0, 5)))
    if len(sb):
        # One PolyCollection, not N ax.fill calls: with a few hundred boxes the per-artist
        # overhead alone is ~200 ms, and it would be paid on every background rebuild.
        ax.add_collection(PolyCollection([box_polygon(*b) for b in sb],
                                         facecolors="0.80", edgecolors="0.30", lw=1.0,
                                         zorder=0, label="static obstacles (Unity)"))
    ax.plot(traj[:, 1], traj[:, 2], "-", color="0.85", lw=1.0, zorder=1,
            label="full route")
    ax.plot(traj[0, 1], traj[0, 2], "o", color="tab:green", ms=9, zorder=3, label="start")
    if goal_xy is not None:
        ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=18, mec="k", zorder=3,
                label="goal")

    # ── dynamic artists: the only things redrawn per frame ────────────────────
    ln_driven, = ax.plot([], [], "-", color="0.55", lw=1.6, zorder=1, animated=True,
                         label="driven so far")
    pc_movers = PolyCollection([], facecolors="tab:orange", alpha=0.45,
                               edgecolors="tab:orange", lw=1.2, zorder=2, animated=True,
                               label="dynamic objects, ground truth (Unity)")
    ax.add_collection(pc_movers)
    # Proxy: the untracked state is an edge colour on pc_movers, which cannot carry a
    # second legend entry of its own.
    ax.plot([], [], "s", mfc="tab:orange", mec="red", mew=2.0, ms=9, alpha=0.7,
            label="ground truth, NOT TRACKED (no keep-out)")
    lc_occ = LineCollection([], linewidths=1.8, alpha=0.95, zorder=1, animated=True)
    lc_dyn = LineCollection([], linewidths=1.6, alpha=0.95, linestyles="--", zorder=1,
                            animated=True)
    lc_axes = LineCollection([], linewidths=2.5, colors="k", zorder=6, animated=True)
    # Ego planform at t_k = 0 and at every sampled stage. A LineCollection rather than a
    # patch per frame: the video redraws by blitting a FIXED set of artists, so the hulls
    # have to live in one artist whose segment list is swapped per frame.
    lc_hull = LineCollection([], linewidths=1.0, alpha=0.85, zorder=4, animated=True)
    # Sensor horizon: it follows the LiDAR, so it is redrawn per frame like the rest.
    lc_horizon = LineCollection([], linewidths=1.0, colors="0.55", alpha=0.7, zorder=0,
                                animated=True)
    for lc in (lc_occ, lc_dyn, lc_axes, lc_hull, lc_horizon):
        ax.add_collection(lc)
    ax.plot([], [], "-", color="0.55", lw=1.0, alpha=0.7,
            label=f"LiDAR range ({SENSOR_RANGE:.0f} m)")
    ln_plan, = ax.plot([], [], "-", color="k", lw=2.0, zorder=4, animated=True,
                       label="predicted rollout, LiDAR (cheapest sample)")
    ln_ego, = ax.plot([], [], "o", mfc="w", mec="k", ms=6, mew=0.9, zorder=5,
                      animated=True, label=f"$t_k$ = 0.0 s   r = {D_SAFE_HARD:.0f} m")
    ln_mover, = ax.plot([], [], "x", color="k", ms=7, mew=1.5, zorder=6, animated=True,
                        label="sensed mover centre (keep-out seed)")
    sc_stage = ax.scatter([], [], s=36, marker="o", edgecolors="k", linewidths=0.9,
                          zorder=5, animated=True)

    # Stage legend entries, from the first frame: the sampled t_k and their radii are a
    # function of the horizon, which is fixed for the run, so they do not change per frame.
    _p0 = frames[0]["plan"]
    _tk0 = (np.arange(len(_p0)) + 1) * DT
    _stages0 = np.linspace(0, len(_p0) - 1, min(max_frames, len(_p0))).round().astype(int)
    for si, k in enumerate(dict.fromkeys(_stages0.tolist())):
        t_k = float(_tk0[k])
        ax.plot([], [], "o", mfc=STAGE_COLORS[si % len(STAGE_COLORS)], mec="k", ms=6,
                mew=0.9, label=f"$t_k$ = {t_k:.1f} s   r = "
                               f"{D_SAFE_HARD + V_TARGET * t_k:.0f} m")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)

    title = ax.set_title("", animated=True, fontsize=11)
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    w, h = fig.canvas.get_width_height()

    exe = _ffmpeg_exe()
    path = f"{stem}.mp4" if exe else f"{stem}.gif"
    proc = gif_frames = None
    if exe:
        proc = subprocess.Popen(
            [exe, "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             # yuv420p (for players that cannot read 4:4:4) needs even dimensions.
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)
    else:
        gif_frames = []

    for f in frames:
        # The CHEAPEST sampled rollout is the drawn one — the sampled timestamps and their
        # keep-outs hang off it. Falls back to the executed plan (the softmax mean) only
        # for recordings made without the extremes, since those carry no sample.
        plan = f.get("best")
        plan = np.asarray(f["plan"] if plan is None or not len(plan) else plan, float)
        # Ego-root [x, y, theta] -> the SENSOR path, matching the still figure. The third
        # column is what makes this possible; a recording without it stays at the root.
        if plan.shape[-1] >= 3:
            plan_x, plan_y = sensor_xy(plan[:, 0], plan[:, 1], plan[:, 2])
        else:
            plan_x, plan_y = plan[:, 0], plan[:, 1]
        tk = (np.arange(len(plan)) + 1) * DT
        segs = (np.asarray(f["segs"], float).reshape(-1, 2, 2)
                if f.get("segs") is not None and len(f["segs"]) else np.empty((0, 2, 2)))

        # t_k = 0 in white, then one colour per sampled stage — same law, same colours and
        # same helpers as the still figure.
        occ_polys, occ_cols, dyn_polys, dyn_cols = [], [], [], []
        for seg in segs:
            occ_polys.append(capsule_polygon(seg[0], seg[1], D_SAFE_HARD))
            occ_cols.append("w")
        for poly in _dyn_keepout_polys(f, 0.0):
            dyn_polys.append(poly); dyn_cols.append("w")

        stages = np.linspace(0, len(plan) - 1,
                             min(max_frames, len(plan))).round().astype(int)
        stage_pts, stage_cols = [], []
        hulls, hull_cols = [], []
        if plan.shape[-1] >= 3:
            hulls.append(aircraft_polygon(f["ex"], f["ey"], f["eth"],
                                          EGO_LENGTH, EGO_SPAN, EGO_NOSE_FWD))
            # Black, not the white the t_k = 0 keep-out uses: this collection is outline
            # only (no fill to sit on), so white would be invisible on the white canvas.
            hull_cols.append("k")
        for si, k in enumerate(dict.fromkeys(stages.tolist())):
            col = STAGE_COLORS[si % len(STAGE_COLORS)]
            t_k = float(tk[k])
            for seg in segs:
                occ_polys.append(capsule_polygon(seg[0], seg[1],
                                                 D_SAFE_HARD + V_TARGET * t_k))
                occ_cols.append(col)
            for poly in _dyn_keepout_polys(f, t_k):
                dyn_polys.append(poly); dyn_cols.append(col)
            stage_pts.append((plan_x[k], plan_y[k])); stage_cols.append(col)
            if plan.shape[-1] >= 3:
                hulls.append(aircraft_polygon(plan[k, 0], plan[k, 1], plan[k, 2],
                                              EGO_LENGTH, EGO_SPAN, EGO_NOSE_FWD))
                hull_cols.append(col)

        lc_occ.set_segments(occ_polys); lc_occ.set_color(occ_cols)
        lc_dyn.set_segments(dyn_polys); lc_dyn.set_color(dyn_cols)
        lc_axes.set_segments(list(segs))
        lc_hull.set_segments(hulls); lc_hull.set_color(hull_cols)

        db, tracked, _matched = _match_tracked(f)
        pc_movers.set_verts([box_polygon(*b) for b in db])
        # Red outline the moment the track behind an object disappears — the frame where
        # that happens is where the ego stopped seeing it.
        pc_movers.set_edgecolor(["tab:orange" if ok else "red" for ok in tracked])
        pc_movers.set_linewidth([1.2 if ok else 2.4 for ok in tracked])
        dyn_c = _dyn_markers(f)
        ln_mover.set_data(dyn_c[:, 0], dyn_c[:, 1])

        n = max(int(np.searchsorted(traj[:, 0], f["t"], side="right")), 1)
        _dx, _dy = sensor_xy(traj[:n, 1], traj[:n, 2], traj[:n, 3])
        ln_driven.set_data(_dx, _dy)
        ln_plan.set_data(plan_x, plan_y)
        _ex, _ey = sensor_xy(f["ex"], f["ey"], f["eth"])
        ln_ego.set_data([_ex], [_ey])
        lc_horizon.set_segments([sensor_horizon(_ex, _ey)])
        sc_stage.set_offsets(np.asarray(stage_pts, float).reshape(-1, 2))
        sc_stage.set_facecolor(stage_cols)

        # Two lines: the axes are narrow (the legend sits outside them), so a single-line
        # title with the cost appended runs off the canvas.
        _what = " + ".join((["occlusion"] if len(segs) else []) +
                           (["dynamic-obstacle"] if len(dyn_c) else []))
        _l2 = (f"expanding {_what} keep-out per sampled $t_k$" if _what else "")
        _c_lo = f.get("best_cost")
        if _c_lo is not None:
            _l2 += ("   |   " if _l2 else "") + f"cheapest sample cost {_c_lo:.3g}"
        if f.get("infeasible"):
            # The ego BRAKED at this solve; the cheapest sample is drawn all the same,
            # so the line on screen is not the one that was executed.
            _l2 += ("   |   " if _l2 else "") + "INFEASIBLE — ego braked"
        title.set_text(f"{verdict} — solve at t = {f['t']:.1f} s"
                       + (f"\n{_l2}" if _l2 else ""))

        fig.canvas.restore_region(bg)
        for a in (ln_driven, pc_movers, lc_occ, lc_dyn, lc_axes, lc_hull, lc_horizon,
                  ln_plan, ln_ego, ln_mover, sc_stage, title):
            ax.draw_artist(a)
        fig.canvas.blit(fig.bbox)

        buf = np.asarray(fig.canvas.buffer_rgba())
        if proc is not None:
            proc.stdin.write(buf.tobytes())
        else:
            from PIL import Image
            gif_frames.append(Image.fromarray(buf.copy()).convert("P", palette=1))

    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with {proc.returncode}")
    else:
        gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:],
                           duration=int(1000 / fps), loop=0)
    plt.close(fig)
    return path, len(frames)


def _save_trajectory(out_dir, verdict, goal_xy, traj, obs_track=None, occ_pts=None,
                     occ_frames=None, capsule_horizon=None, max_frames=6,
                     show_occlusion=True, static_boxes=None, solve_t=None,
                     video=False, video_fps=10, video_stride=1, **_ignored):
    """CSV of the executed run + ONE figure: the executed trajectory and the predicted
    rollout of a SINGLE solve with a handful of sampled horizon timestamps on it — each
    timestamp drawn together with the expanding occlusion keep-out that applied at it.

    `solve_t` [s] pins WHICH solve that is — the recorded solve closest in episode time
    to it. Left None, the solve drawn is the one whose plan came closest to (or inside)
    its expanding occlusion set; with no occlusion boundaries anywhere it falls back to
    the solve whose plan bent hardest.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    # One fixed name, verdict-independent: a collision run must OVERWRITE the previous
    # figure, not add a second traj_collision.png next to a stale traj_reached.png. The
    # verdict is still in the title and in the CSV's contents.
    stem = os.path.join(out_dir, "traj")

    header = "t,x,y,theta,v,a_cmd,delta_cmd,delta_act,a_act"
    np.savetxt(f"{stem}.csv", traj, delimiter=",", header=header, comments="")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[traj] saved {stem}.csv (matplotlib missing — no plot)")
        return

    TRAJ_MARGIN = 100.0   # [m] padding around the ego extent — the plot's zoom level

    # Sensor track, matching what _draw_solve_map draws; used only for the window.
    x, y = sensor_xy(traj[:, 1], traj[:, 2], traj[:, 3])

    frames = [f for f in (occ_frames or []) if f.get("plan") is not None and len(f["plan"]) > 1]
    if not frames:
        # The state plot needs no solve, so it is still worth writing.
        _save_state_plot(stem, verdict, traj, goal_xy)
        print(f"[traj] saved {stem}.csv and {stem}_state.png "
              f"(no recorded solve with a plan — no map plot)")
        return

    def _curvature(fr):
        d = np.diff(fr["plan"], axis=0)
        th = np.arctan2(d[:, 1], d[:, 0])
        return float(np.abs(np.unwrap(th)[-1] - np.unwrap(th)[0]))

    def _tightest(fr):
        """max over the horizon of (keep-out radius − distance to the boundary): how
        close this plan came to the expanding keep-out set (occlusion capsules and/or
        movers). Positive ⇒ inside it. Measured against the DRAWN geometry, so the solve
        picked as tightest is the tightest one in the figure that gets written."""
        plan = fr["plan"]
        t = (np.arange(len(plan)) + 1) * DT
        best = -np.inf
        if fr.get("segs") is not None and len(fr["segs"]):
            d = point_segments_min_distance(plan[:, 0], plan[:, 1], fr["segs"])
            best = max(best, float(np.max((D_SAFE_HARD + V_TARGET * t) - d)))
        t_dyn = t if not DYN_GROW_HORIZON else np.minimum(t, DYN_GROW_HORIZON)
        for c0, c1, r_c in _dyn_seeds(fr):
            d = np.hypot(plan[:, 0] - c0, plan[:, 1] - c1)
            best = max(best, float(np.max((D_SAFE_HARD + r_c + V_TARGET * t_dyn) - d)))
        return best

    # Prefer a solve that actually had a keep-out set — a frame without one has nothing
    # expanding to show. Among those, the tightest one.
    occ_ok = [f for f in frames
              if (f.get("segs") is not None and len(f["segs"]))
              or (f.get("dyn_set") is not None and len(f["dyn_set"]))
              or (f.get("dyn_boxes") is not None and len(f["dyn_boxes"]))]
    if solve_t is not None:
        fr = min(frames, key=lambda f: abs(f["t"] - solve_t))
        if abs(fr["t"] - solve_t) > DT:
            print(f"[traj] no solve recorded at t={solve_t:.1f}s — using the nearest, "
                  f"t={fr['t']:.1f}s (recorded {frames[0]['t']:.1f}..{frames[-1]['t']:.1f}s)")
    else:
        fr = max(occ_ok, key=_tightest) if occ_ok else max(frames, key=_curvature)
    plan = fr["plan"]

    fig, ax = plt.subplots(figsize=(11, 11))
    segs, dyn_c, stages = _draw_solve_map(ax, fr, verdict, traj, goal_xy=goal_xy,
                                          static_boxes=static_boxes,
                                          max_frames=max_frames)

    # Zoom on the EGO's own extent (start → end), the plan it is executing, AND the sensor
    # horizon of the drawn solve — the ring is the whole point of drawing it, so a window
    # that clips it shows a legend entry for something invisible. Static obstacles and the
    # keep-out capsules are still left out: a 700 m-away wall would zoom the ego to a dot.
    _px, _py = sensor_xy(plan[:, 0], plan[:, 1], plan[:, 2])
    allx = np.concatenate([x, _px])
    ally = np.concatenate([y, _py])
    cx, cy = 0.5 * (allx.min() + allx.max()), 0.5 * (ally.min() + ally.max())
    half_x = 0.5 * (allx.max() - allx.min()) + TRAJ_MARGIN
    half_y = max(0.5 * (ally.max() - ally.min()) + TRAJ_MARGIN, 0.25 * half_x)
    # Then grow — if needed — to fit the horizon ring, with NO further margin: the ring is
    # already 250 m of slack in every direction, and stacking TRAJ_MARGIN on top of it just
    # shrinks the ego for nothing.
    _hx, _hy = sensor_xy(fr["ex"], fr["ey"], fr["eth"])
    half_x = max(half_x, abs(_hx - cx) + SENSOR_RANGE)
    half_y = max(half_y, abs(_hy - cy) + SENSOR_RANGE)
    ax.set_xlim(cx - half_x, cx + half_x); ax.set_ylim(cy - half_y, cy + half_y)
    # Equal aspect on a SQUARE canvas would force the y window out to the x window's
    # size, undoing the zoom. Match the canvas to the data instead: metres stay square
    # while the window stays tight around the ego.
    ax.set_aspect("equal", adjustable="box")
    fig.set_size_inches(12.0, float(np.clip(12.0 * half_y / half_x, 4.0, 12.0)))
    # Legend OUTSIDE the axes: the zoomed window is short, so "best" placement lands it
    # on top of the rollout every time.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=120)
    plt.close(fig)

    _save_state_plot(stem, verdict, traj, goal_xy, t_mark=fr["t"],
                     solve_ts=[f["t"] for f in frames])

    if _save_tracking_plot(stem, verdict, frames, traj, static_boxes=static_boxes):
        print(f"[traj] saved {stem}_tracking.png (ground-truth vs sensed mover positions, "
              f"and where each track was lost)")

    # Same solve, second figure: the whole sample cloud it chose that plan from.
    if _save_rollouts_plot(stem, verdict, fr, goal_xy=goal_xy,
                           static_boxes=static_boxes, max_frames=max_frames):
        print(f"[traj] saved {stem}_rollouts.png ({len(fr['rollouts'])} sampled rollouts "
              f"at t={fr['t']:.1f}s)")
    else:
        print("[traj] no _rollouts.png — no sampled rollouts recorded "
              "(pass --plot-rollouts N)")

    print(f"[traj] saved {stem}.csv, {stem}.png and {stem}_state.png (rollout from the solve at "
          f"t={fr['t']:.1f}s, {len(set(stages.tolist()))} sampled stages, "
          f"{len(segs)} occlusion boundaries, {len(dyn_c)} dynamic objects)")

    # Same figure, every solve: the run as a video. The PNG above stays as the single
    # frame of the solve worth freezing.
    if video:
        try:
            out = _save_trajectory_video(stem, verdict, frames, traj, goal_xy=goal_xy,
                                         static_boxes=static_boxes,
                                         max_frames=max_frames, fps=video_fps,
                                         stride=video_stride)
        except Exception as e:
            print(f"[traj] video failed: {e}")
        else:
            if out is not None:
                path, n = out
                print(f"[traj] saved {path} ({n} solves @ {video_fps} fps"
                      f"{'' if video_stride == 1 else f', every {video_stride}th solve'})")


def run(unity_exec_path=None, port=5004, run_sysid=True,
        noise_std=0.0, detect_range=float("inf"),
        uncertainty=False, n_scenarios=N_SCEN, w_info=W_INFO,
        d_infl=D_INFL, d_safe=D_SAFE, info_range=INFO_RANGE,
        lidar_costmap=False, lidar_topic="/point_cloud", visibility_cost=False,
        occlusion_aware=False, dynamic_obstacles=False, dynamic_viz=False,
        occlusion_viz=False, save_traj=None, show_occlusion_plot=True, plot_solve_t=None,
        rviz_viz=False, rviz_rollouts=40, plot_rollouts=0,
        traj_video=False, video_fps=10, video_stride=1,
        detector=False, detector_topic="/detections"):
    global DETECTION_RANGE, UNCERTAINTY, N_SCEN, W_INFO
    global D_INFL, D_SAFE, INFO_RANGE, LIDAR_COSTMAP, VISIBILITY_COST, STATIC_AVOID
    global OCCLUSION_AWARE, OCC_TRACKER, OCC_SEGS_NOW, OCC_PUB
    global DYNAMIC_AVOID, DYN_TRACKER, DYN_NOW, DYN_DBG, DYN_PUB, DYN_LAST_STAMP, DYN_CL_N
    global DYN_SOURCE
    global RVIZ_PUB, MPPI_KEEP_ROLLOUTS, MPPI_TRACK_BEST
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
                if occlusion_viz:
                    from occlusion_capsules import OcclusionSegmentPublisher
                    OCC_PUB = OcclusionSegmentPublisher()
                    if not OCC_PUB.start():
                        OCC_PUB = None
            if DYNAMIC_AVOID:
                DYN_TRACKER = DynamicClusterTracker(
                    assoc_radius=DYN_ASSOC, q_accel=DYN_Q_ACCEL, r_frac=DYN_R_FRAC,
                    r_min=DYN_R_MIN, sigma_v0=DYN_SIGMA_V0, extent_alpha=DYN_EXT_ALPHA,
                    ttl=DYN_TTL, min_hits=DYN_MIN_HITS, v_min=DYN_V_MIN,
                    min_dyn_hits=DYN_MIN_DYN_HITS, require_motion=DYN_REQUIRE_MOTION,
                    dyn_window=DYN_WINDOW, extent_frac=DYN_EXTENT_FRAC,
                    resegment_ratio=DYN_RESEG_RATIO)
                if dynamic_viz:
                    DYN_PUB = DynamicClusterPublisher()
                    if not DYN_PUB.start():
                        DYN_PUB = None
                # Measurement source. The detector replaces ONLY the clusters() call;
                # if its node is not up, fall back to grid clustering rather than
                # running blind — a silently empty detector topic would otherwise look
                # exactly like an empty apron.
                DYN_SOURCE = cm
                if detector:
                    from detection_source import DetectionSource
                    ds = DetectionSource()
                    if ds.start(topic=detector_topic):
                        DYN_SOURCE = ds
                    else:
                        print("[Controller] Detector      : requested but rclpy "
                              "unavailable — falling back to grid clustering")
            feats = [f"static(D_SAFE={D_SAFE_HARD:.1f}m, circle-cover)"]
            if OCCLUSION_AWARE:  feats.append(f"occlusion(D_SAFE={D_SAFE_HARD:.1f}m, "
                                             f"v_target={V_TARGET:.1f}m/s, sightline)")
            if DYNAMIC_AVOID:    feats.append(
                f"dynamic({'PointPillars' if detector else 'clustered'}, "
                f"v_min={DYN_V_MIN:.1f}m/s, "
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
    if occlusion_viz and not OCCLUSION_AWARE:
        print("[Controller] --occlusion-viz needs --occlusion-aware (there is nothing to "
              "draw without the detector) — DISABLED")
    if dynamic_viz and not DYNAMIC_AVOID:
        print("[Controller] --dynamic-viz needs --dynamic-obstacles (there is nothing to "
              "draw without the detector) — DISABLED")
    if rviz_viz:
        from rviz_viz import RvizVisualizer
        _rv = RvizVisualizer()
        if _rv.start():
            RVIZ_PUB = _rv
            MPPI_KEEP_ROLLOUTS = max(int(rviz_rollouts), 0)
            print(f"[Controller] RViz feed    : ON  (/viz/*, {MPPI_KEEP_ROLLOUTS} rollouts "
                  f"drawn per solve)")

    # The _rollouts.png figure needs mppi() to keep its samples. Shares the RViz switch —
    # take the larger request, so the two flags do not fight over one global.
    if save_traj is not None and plot_rollouts > 0:
        MPPI_KEEP_ROLLOUTS = max(MPPI_KEEP_ROLLOUTS, int(plot_rollouts))
        print(f"[Controller] Rollout plot : ON  ({MPPI_KEEP_ROLLOUTS} cheapest rollouts "
              f"recorded per solve)")
    if traj_video and save_traj is None:
        print("[Controller] --traj-video needs --save-traj (there is nowhere to write "
              "the video) — ignored.")
        traj_video = False
    elif traj_video:
        # The video draws the CHEAPEST sample per solve, which needs mppi() to keep every
        # sampled path for the length of one solve (one reused (K,H,2) buffer, not a
        # per-step allocation).
        MPPI_TRACK_BEST = True
        print(f"[Controller] Trajectory video: ON  ({video_fps} fps, every "
              f"{video_stride} solve(s), drawing the cheapest sample per solve)")

    if plot_rollouts > 0 and save_traj is None:
        print("[Controller] --plot-rollouts needs --save-traj (there is nowhere to write "
              "the figure) — DISABLED")

    # Static-obstacle footprints from Unity. Plot-only, so it is started only when a
    # trajectory is being saved and its absence never affects control.
    static_obs = None
    dyn_obs = None
    if save_traj is not None:
        from static_obstacles import StaticObstacles
        _so = StaticObstacles()
        static_obs = _so if _so.start() else None
        # Ground-truth movers, snapshotted per step so the plot can show them AT the
        # solve it draws rather than at the end of the run.
        from dynamic_obstacles import DynamicObstacles
        _do = DynamicObstacles()
        dyn_obs = _do if _do.start() else None

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


    unity_stopped = False
    mean      = np.zeros((H_MPPI, 2))
    min_clear = np.inf   # closest approach to any LiDAR circle SURFACE over the run
    collided  = False
    reached   = False
    traj      = []   # per-step ego pose: (t, x, y, theta, v, a_cmd, delta_cmd)
    occ_frames = []   # one dict per solve: {t, ex, ey, plan}; see the loop
    dyn_track = []

    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    ep_steps     = 0
    delta_actual = 0.0   # tracked Python-side to feed into MPPI state
    accel_actual = 0.0
    u_prev       = np.zeros(2)   # last applied [a, delta], for the MPPI Δu smoothness cost
    episode_done = False

    while not episode_done:
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

        # rollout frame.
        if LIDAR_COSTMAP is not None:
            ego_fwd = np.array([np.cos(s[2]), np.sin(s[2])])
            LIDAR_COSTMAP.update(ego_fwd)

            if OCCLUSION_AWARE and OCC_TRACKER is not None and LIDAR_COSTMAP.ready:
                _raw = LIDAR_COSTMAP.occlusion_segments(
                    ego_fwd=ego_fwd, fwd_half_angle_deg=OCC_FWD_HALF_ANGLE)
                _s = OCC_TRACKER.update(_raw, time.monotonic())
                if _s is not None and len(_s) and not OCC_USE_CAPSULES:
                    _s = _s.copy()
                    _s[:, 1, :] = _s[:, 0, :]
                OCC_SEGS_NOW = _s
                if OCC_PUB is not None:
                    # The RAW per-scan detections, not the tracked set: this overlay is
                    # here to show what the jump test found on this scan.
                    OCC_PUB.publish(_raw, V_TARGET, D_SAFE_HARD, OCC_T_GROW_MAX)
            else:
                OCC_SEGS_NOW = None

            _src = DYN_SOURCE if DYN_SOURCE is not None else LIDAR_COSTMAP
            if DYNAMIC_AVOID and DYN_TRACKER is not None and _src.ready:
                _now = time.monotonic()
                _stamp = _src.stamp
                if _stamp != DYN_LAST_STAMP:
                    DYN_LAST_STAMP = _stamp
                    _cl = _src.clusters(cell=DYN_CELL, min_points=DYN_MIN_POINTS,
                                        max_radius=DYN_MAX_RADIUS)
                    DYN_TRACKER.update(_cl, _now)
                    DYN_CL_N = 0 if _cl is None else len(_cl)
                else:
                    DYN_TRACKER.predict_to(_now)
                DYN_NOW = DYN_TRACKER.dynamic(_now)
                _sp = DYN_TRACKER.speeds()
      
                DYN_DBG = (DYN_CL_N, DYN_TRACKER.n_tracks,
                           0 if DYN_NOW is None else len(DYN_NOW),
                           max(_sp) if _sp else 0.0)

            else:
                DYN_NOW = None
        u_nom, mean = mppi(s, mean, goal_xy, u_prev)

        if DYN_PUB is not None:
            _viz = DYN_USED
            if _viz is not None and len(_viz) and not DYN_INCLUDE_AGE:

                _viz = np.asarray(_viz, float).copy()
                _viz[:, 3] = 0.0
            DYN_PUB.publish(_viz, V_TARGET, D_SAFE_HARD, DYN_GROW_HORIZON)

        if RVIZ_PUB is not None:
            RVIZ_PUB.publish(ego=s, goal_xy=goal_xy, plan=OCC_PLAN,
                             infeasible=OCC_INFEASIBLE,
                             rollouts=MPPI_ROLLOUTS, rollout_costs=MPPI_COSTS,
                             occ_segs=OCC_SEGS_USED, occ_segs_all=OCC_SEGS_NOW,
                             dyn_set=DYN_USED, dt=DT, v_target=V_TARGET,
                             d_safe=D_SAFE_HARD, t_grow_max=OCC_T_GROW_MAX)

        if save_traj is not None and DYN_USED is not None and len(DYN_USED):
            for _c0, _c1, _r, _age in DYN_USED:
                dyn_track.append((ep_steps * DT, float(_c0), float(_c1)))

        # One record per solve — EVERY step, feasible or not, so --plot-solve-t can name
        # any moment of the run. An infeasible step records its braking rollout and is
        # flagged; the figure draws it like any other plan.
        if save_traj is not None and OCC_PLAN is not None:
            _segs = OCC_SEGS_USED
            occ_frames.append({"t": ep_steps * DT,
                               # Ego ROOT pose. The figures convert it to the sensor
                               # position with sensor_xy(), which needs the heading.
                               "ex": float(s[0]), "ey": float(s[1]),
                               "eth": float(s[2]),
                               "plan": OCC_PLAN.copy(),
                               # The occlusion boundaries this solve actually constrained
                               # against (post-gating) — what the cost saw, so the drawn
                               # keep-outs are the enforced ones.
                               "segs": (np.array(_segs, dtype=float)
                                        if _segs is not None and len(_segs) else None),
                               # Ground-truth movers as of THIS solve. Drawn as filled
                               # boxes only — the gap between one of these and its keep-out
                               # centre IS the localisation error.
                               "dyn_boxes": (None if dyn_obs is None else dyn_obs.boxes()),
                               # The SENSED movers this solve constrained against:
                               # (K,4) [c0, c1, r_cluster, age]. The figures expand their
                               # bubbles around THESE, because this is where the planner
                               # believes the traffic is.
                               "dyn_set": (None if DYN_USED is None or not len(DYN_USED)
                                           else np.asarray(DYN_USED, float).copy()),
                               # The sampled rollouts the softmax weighted at this solve,
                               # for the _rollouts.png figure. float32 because this is the
                               # one per-solve record whose size scales with K (N*H*2).
                               "rollouts": (None if MPPI_ROLLOUTS is None
                                            else MPPI_ROLLOUTS.astype(np.float32)),
                               "rollout_costs": (None if MPPI_COSTS is None
                                                 else MPPI_COSTS.astype(np.float32)),
                               # Cheapest SAMPLE of this solve — the rollout the video
                               # draws, and what its sampled t_k hang off.
                               "best": (None if MPPI_BEST is None
                                        else MPPI_BEST[0].astype(np.float32)),
                               "best_cost": (None if MPPI_BEST is None
                                             else MPPI_BEST[1]),
                               "infeasible": bool(OCC_INFEASIBLE)})

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

        # Advance Pythos with the canvas sized to the data aspen-side kinematic state to match what Unity will compute
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

        if LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
            _c = LIDAR_COSTMAP.circles()
            if _c is not None and len(_c):
                min_clear = min(min_clear, float(
                    np.min(np.hypot(_c[:, 0] - s[0], _c[:, 1] - s[1]) - _c[:, 2])))


        # Record ego pose this step (world frame: x=Unity Z, y=Unity X).
        traj.append((ep_steps * DT, s[0], s[1], s[2], s[3], a_cmd, delta_cmd,
                     delta_actual, accel_actual))

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
    if OCC_PUB is not None:
        OCC_PUB.shutdown()
    if RVIZ_PUB is not None:
        RVIZ_PUB.shutdown()

    verdict = "collision" if collided else ("reached" if reached else "timeout")

    if save_traj is not None and traj:
        _save_trajectory(save_traj, verdict, goal_xy, np.asarray(traj),
                         # SENSED dynamic-obstacle centres (clustered + tracked from the
                         # LiDAR), not the removed oracle feed.
                         np.asarray(dyn_track) if dyn_track else None,
                         None, occ_frames=occ_frames,
                         static_boxes=(None if static_obs is None
                                       else static_obs.boxes()),
                         show_occlusion=show_occlusion_plot, solve_t=plot_solve_t,
                         video=traj_video, video_fps=video_fps,
                         video_stride=video_stride)

    if static_obs is not None:
        static_obs.shutdown()
    if dyn_obs is not None:
        dyn_obs.shutdown()

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
    p.add_argument("--detector", action="store_true",
                   help="Take dynamic obstacles from lidar_detector_node.py "
                        "(PointPillars) instead of the grid clustering in "
                        "dynamic_clusters.py. Only the MEASUREMENT changes: the same "
                        "Kalman tracker, static/dynamic test and keep-out cost run in "
                        "both modes. Requires --dynamic-obstacles and the detector node "
                        "publishing on --detector-topic; falls back to clustering if it "
                        "is not up.")
    p.add_argument("--detector-topic", default="/detections",
                   help="Float32MultiArray topic from lidar_detector_node.py.")
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
    p.add_argument("--occlusion-viz", action="store_true",
                   help="Publish the boundary segments this scan detected on "
                        "/occlusion_segments (std_msgs/Float32MultiArray) so "
                        "OcclusionSegmentVisualizer.cs can draw them in the Unity Scene "
                        "view. Shows the PYTHON detector's output, unlike "
                        "PhantomAgentVisualizer which draws Unity's own edge pass. "
                        "Requires --occlusion-aware.")
    p.add_argument("--rviz-viz", action="store_true",
                   help="Publish the detected occlusion boundaries (with their expanding "
                        "keep-outs), the sampled MPPI rollouts and the executed plan as "
                        "visualization_msgs/MarkerArray on /viz/* for RViz2, plus the "
                        "map->lidar_link TF that puts them in the same frame as the "
                        "Unity point cloud. Needs ROS 2 sourced (rclpy, tf2_ros).")
    p.add_argument("--rviz-rollouts", type=int, default=40, metavar="N",
                   help="How many of the cheapest sampled rollouts --rviz-viz draws per "
                        "solve. 0 draws only the executed plan.")
    p.add_argument("--save-traj", default=None, metavar="DIR",
                   help="Save the run's ego trajectory as CSV + a top-down PNG plot into DIR.")
    p.add_argument("--plot-rollouts", type=int, default=0, metavar="N",
                   help="Also write traj_rollouts.png: the N cheapest sampled MPPI "
                        "rollouts of the plotted solve, coloured by cost. Needs "
                        "--save-traj. 0 = off. Costs N*horizon*2 float32 per solve of "
                        "memory, so keep it in the low hundreds.")
    p.add_argument("--traj-video", action="store_true",
                   help="Also write traj.mp4 (traj.gif without ffmpeg): the same map "
                        "figure replayed over EVERY recorded solve, so the expanding "
                        "occlusion capsules and mover tubes can be watched growing and "
                        "being dodged. traj.png stays as the single frozen frame. "
                        "Needs --save-traj.")
    p.add_argument("--video-fps", type=int, default=10, metavar="FPS",
                   help="Frame rate of --traj-video. Default 10.")
    p.add_argument("--video-stride", type=int, default=1, metavar="N",
                   help="Draw only every Nth solve in --traj-video — the cheap way to "
                        "shorten a long run's video. Default 1 (every solve).")
    p.add_argument("--plot-solve-t", type=float, default=None, metavar="SEC",
                   help="Episode time [s] of the solve whose rollout the plot draws: the "
                        "t_k = 0 point and the sampled future timestamps hang off it. "
                        "Default: the solve that came closest to its occlusion keep-out.")
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
        occlusion_viz=args.occlusion_viz,
        show_occlusion_plot=not args.no_occlusion_plot,
        save_traj=args.save_traj,
        plot_solve_t=args.plot_solve_t,
        plot_rollouts=args.plot_rollouts,
        traj_video=args.traj_video,
        video_fps=args.video_fps,
        video_stride=args.video_stride,
        rviz_viz=args.rviz_viz,
        rviz_rollouts=args.rviz_rollouts,
        detector=args.detector,
        detector_topic=args.detector_topic)
