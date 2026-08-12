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
OCC_PLAN        = None   # (H,2) the PLANNED future path from the last solve, in world
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
MPPI_ROLLOUTS   = None   # (n,H,2) sampled rollout paths from the LAST solve, kept only
                         # when --rviz-viz is on: it is pure visualisation, and holding
                         # every sample's path costs K*H*2 floats per step otherwise.
MPPI_COSTS      = None   # (n,) their total costs, for the colour ramp
MPPI_KEEP_ROLLOUTS = 0   # how many sampled paths mppi() records (0 = off)
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
    global OCC_INFEASIBLE, MPPI_ROLLOUTS, MPPI_COSTS
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
    MPPI_ROLLOUTS = MPPI_COSTS = None
    paths = (np.empty((K_MPPI, H_MPPI, 2)) if MPPI_KEEP_ROLLOUTS > 0 else None)

    for k in range(H_MPPI):
        st = _rollout_step(st, na[:, k, 0], na[:, k, 1])
        fwd, lat, th, vv = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
        if paths is not None:
            paths[:, k, 0], paths[:, k, 1] = fwd, lat

        u_k   = na[:, k, :]                                  # (K, 2)
        u_km1 = u_prev[None, :] if k == 0 else na[:, k - 1, :]

       
        _d_static = None
        if STATIC_AVOID and LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
          
            _d_static = LIDAR_COSTMAP.distance(fwd, lat)
        cost += tcost.stage_cost(
            fwd, lat, th, vv, u_k, u_km1,
            goal_xy=(goal_fwd, goal_lat), v_des=v_des_eff, t_k=(k + 1) * DT,
            r_act=R_ACT, r_dact=R_DACT,
            w_goal_run=W_GOAL_RUN, w_head=W_HEAD, w_v=W_V,
            d_static=_d_static, d_safe_static=D_SAFE_HARD, w_static=W_HARD)

        if occ_segs is not None:
            cost += occlusion_stage_cost(
                _dist_to_occ(fwd, lat), vv, (k + 1) * DT,
                V_TARGET, D_SAFE_HARD, W_HARD, t_grow_max=OCC_T_GROW_MAX,
                w_soft=OCC_W_SOFT, d_infl=OCC_D_INFL,
                cost_current = cost, action = np.column_stack((fwd, lat)))

        if dyn_set is not None:
            t_dyn = (k + 1) * DT
            for c0, c1, r_c, age in dyn_set:
                d_dyn = np.hypot(fwd - c0, lat - c1)

                d_base = D_SAFE_HARD + r_c

                cost += occlusion_stage_cost(
                    d_dyn, vv, t_dyn, V_TARGET, d_base, W_HARD,
                    w_sight=W_SIGHT, a_brake=A_BRAKE_SIGHT, v_floor=V_SIGHT_FLOOR,  dyn = True, cost_current = cost)



    # print("MIN_cost_trajectory: ", cost.min())
    # print("MAX_cost_trajectory: ", cost.max())

    cost += tcost.terminal_cost(st[:, 0], st[:, 1],
                                goal_xy=(goal_fwd, goal_lat), w_goal_term=W_GOAL_TERM)

    if paths is not None:
        keep = np.argsort(cost)[:MPPI_KEEP_ROLLOUTS]
        MPPI_ROLLOUTS, MPPI_COSTS = paths[keep], cost[keep]

    n_feasible = int((cost <= C_INFEAS).sum())
    if n_feasible <= INFEAS_FRAC * K_MPPI:
        print(f"[MPPI] INFEASIBLE: {n_feasible}/{K_MPPI} rollouts under {C_INFEAS:.2e} "
              f"(depth {INFEAS_DEPTH:.2f} m), min cost {cost.min():.3e} — braking")
        u_stop = np.array([A_MIN, 0.0])                 # max decel, hold wheel straight
        # Roll the braking command out over the horizon so the figure can show WHERE the
        # ego still ends up while stopping — the case where it clips the keep-outs is
        # exactly the one worth looking at, so this must not be left as a stale plan.
        _st = np.asarray(s0, float)[None, :]
        _brake = np.empty((H_MPPI, 2))
        for _k in range(H_MPPI):
            _st = _rollout_step(_st, u_stop[0:1], u_stop[1:2])
            _brake[_k] = _st[0, :2]
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
    _plan = np.empty((H_MPPI, 2))
    for _k in range(H_MPPI):
        _st = _rollout_step(_st, opt[_k:_k + 1, 0], opt[_k:_k + 1, 1])
        _plan[_k] = _st[0, :2]
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


def _save_trajectory(out_dir, verdict, goal_xy, traj, obs_track=None, occ_pts=None,
                     occ_frames=None, capsule_horizon=None, max_frames=6,
                     show_occlusion=True, static_boxes=None, solve_t=None, **_ignored):
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

    from occlusion_capsules import capsule_polygon
    from static_obstacles import box_polygon

    TRAJ_MARGIN = 100.0   # [m] padding around the ego extent — the plot's zoom level

    x, y = traj[:, 1], traj[:, 2]

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
        sensed movers). Positive ⇒ inside it."""
        plan = fr["plan"]
        t = (np.arange(len(plan)) + 1) * DT
        best = -np.inf
        if fr.get("segs") is not None and len(fr["segs"]):
            d = point_segments_min_distance(plan[:, 0], plan[:, 1], fr["segs"])
            best = max(best, float(np.max((D_SAFE_HARD + V_TARGET * t) - d)))
        for c0, c1, r_c in _dyn_bubbles(fr.get("dyn_boxes")):
            d = np.hypot(plan[:, 0] - c0, plan[:, 1] - c1)
            best = max(best, float(np.max((D_SAFE_HARD + r_c + V_TARGET * t) - d)))
        return best

    # Prefer a solve that actually had a keep-out set — a frame without one has nothing
    # expanding to show. Among those, the tightest one.
    occ_ok = [f for f in frames
              if (f.get("segs") is not None and len(f["segs"]))
              or (f.get("dyn_boxes") is not None and len(f["dyn_boxes"]))]
    if solve_t is not None:
        fr = min(frames, key=lambda f: abs(f["t"] - solve_t))
        if abs(fr["t"] - solve_t) > DT:
            print(f"[traj] no solve recorded at t={solve_t:.1f}s — using the nearest, "
                  f"t={fr['t']:.1f}s (recorded {frames[0]['t']:.1f}..{frames[-1]['t']:.1f}s)")
    else:
        fr = max(occ_ok, key=_tightest) if occ_ok else max(frames, key=_curvature)
    plan = fr["plan"]
    tk = (np.arange(len(plan)) + 1) * DT
    segs = (np.asarray(fr["segs"], float).reshape(-1, 2, 2)
            if fr.get("segs") is not None and len(fr["segs"]) else np.empty((0, 2, 2)))
    # Movers as Unity publishes them on /dynamic_obstacles, at THIS solve. Each seeds an
    # expanding CIRCLE of radius D_SAFE_HARD + r_obj + V_TARGET*t_k — the same growth law
    # as the occlusion capsules, so they are drawn at the same sampled t_k, same colour.
    dyn_set = _dyn_bubbles(fr.get("dyn_boxes"))

    fig, ax = plt.subplots(figsize=(11, 11))

    # Static obstacles as Unity reports them (StaticObstaclePublisher.cs).
    sb = (np.asarray(static_boxes, float).reshape(-1, 5)
          if static_boxes is not None and len(static_boxes) else np.empty((0, 5)))
    for i, (bx, by, bsx, bsy, byaw) in enumerate(sb):
        poly = box_polygon(bx, by, bsx, bsy, byaw)
        ax.fill(poly[:, 0], poly[:, 1], color="0.80", ec="0.30", lw=1.0, zorder=0,
                label="static obstacles (Unity)" if i == 0 else None)

    # Ground-truth movers where they were AT the drawn solve (t_k = 0), not at the end.
    db = (np.asarray(fr["dyn_boxes"], float).reshape(-1, 5)
          if fr.get("dyn_boxes") is not None and len(fr["dyn_boxes"]) else np.empty((0, 5)))
    for i, (bx, by, bsx, bsy, byaw) in enumerate(db):
        poly = box_polygon(bx, by, bsx, bsy, byaw)
        ax.fill(poly[:, 0], poly[:, 1], color="tab:orange", alpha=0.45, ec="tab:orange",
                lw=1.2, zorder=2,
                label="dynamic objects at $t_k$ = 0 (Unity)" if i == 0 else None)

    ax.plot(x, y, "-", color="0.65", lw=1.2, zorder=1, label="executed trajectory")
    ax.plot(x[0], y[0], "o", color="tab:green", ms=9, zorder=3, label="start")
    ax.plot(x[-1], y[-1], "s", color="tab:red", ms=9, zorder=3, label="end")
    if goal_xy is not None:
        ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=18, mec="k", zorder=3,
                label="goal")

    ax.plot(plan[:, 0], plan[:, 1], "-", color="k", lw=2.0, zorder=4,
            label=("braking rollout — no feasible plan "
                   f"(solve at t = {fr['t']:.1f} s)" if fr.get("infeasible") else
                   f"predicted rollout (solve at t = {fr['t']:.1f} s)"))

    # Sampled predicted timestamps along that rollout.
    stage_colors = ["#00e5ff", "#ff00a0", "#ffd400", "#00ff7f",
                    "#7c4dff", "#ff6d00", "#00b0ff", "#c6ff00"]
    # t_k = 0: the ego pose at this solve, with the un-expanded keep-out.
    for seg in segs:
        poly = capsule_polygon(seg[0], seg[1], D_SAFE_HARD)
        ax.plot(poly[:, 0], poly[:, 1], "-", color="w", lw=1.8, alpha=0.95, zorder=1)
    for c0, c1, r_c in dyn_set:
        poly = capsule_polygon((c0, c1), (c0, c1), D_SAFE_HARD + r_c)
        ax.plot(poly[:, 0], poly[:, 1], "--", color="w", lw=1.6, alpha=0.95, zorder=1)
    ax.plot(fr["ex"], fr["ey"], "o", mfc="w", mec="k", ms=6, mew=0.9, zorder=5,
            label=f"$t_k$ = 0.0 s   r = {D_SAFE_HARD:.0f} m")

    stages = np.linspace(0, len(plan) - 1, min(max_frames, len(plan))).round().astype(int)
    for si, k in enumerate(dict.fromkeys(stages.tolist())):
        col = stage_colors[si % len(stage_colors)]
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
        for c0, c1, r_c in dyn_set:
            poly = capsule_polygon((c0, c1), (c0, c1), D_SAFE_HARD + r_c + V_TARGET * t_k)
            ax.plot(poly[:, 0], poly[:, 1], "--", color=col, lw=1.6, alpha=0.95, zorder=1)
        ax.plot(plan[k, 0], plan[k, 1], "o", mfc=col, mec="k", ms=6, mew=0.9, zorder=5,
                label=f"$t_k$ = {t_k:.1f} s   r = {r_k:.0f} m")

    for seg in segs:
        ax.plot(seg[:, 0], seg[:, 1], "-", color="k", lw=2.5, zorder=6)
        ax.plot(seg[0, 0], seg[0, 1], ".", color="k", ms=8, zorder=6)

    for i, (c0, c1, _r) in enumerate(dyn_set):
        ax.plot(c0, c1, "x", color="k", ms=7, mew=1.5, zorder=6,
                label="dynamic object centre (bubble seed)" if i == 0 else None)

    # Zoom on the EGO's own extent (start → end) plus the plan it is executing. Static
    # obstacles and the outer keep-out capsules are deliberately left out of the bounds:
    # a 700 m-away wall or a 40 m radius ring would zoom the ego down to a few pixels.
    # Anything outside the window simply gets clipped.
    allx = np.concatenate([x, plan[:, 0]])
    ally = np.concatenate([y, plan[:, 1]])
    cx, cy = 0.5 * (allx.min() + allx.max()), 0.5 * (ally.min() + ally.max())
    half_x = 0.5 * (allx.max() - allx.min()) + TRAJ_MARGIN
    half_y = max(0.5 * (ally.max() - ally.min()) + TRAJ_MARGIN, 0.25 * half_x)
    ax.set_xlim(cx - half_x, cx + half_x); ax.set_ylim(cy - half_y, cy + half_y)
    # Equal aspect on a SQUARE canvas would force the y window out to the x window's
    # size, undoing the zoom. Match the canvas to the data instead: metres stay square
    # while the window stays tight around the ego.
    ax.set_aspect("equal", adjustable="box")
    fig.set_size_inches(12.0, float(np.clip(12.0 * half_y / half_x, 4.0, 12.0)))
    ax.set_xlabel("x  (Unity Z) [m]"); ax.set_ylabel("y  (Unity X) [m]")
    _what = " + ".join(([ "occlusion"] if len(segs) else []) +
                       (["dynamic-obstacle"] if len(dyn_set) else []))
    ax.set_title(f"{verdict} — predicted rollout at t = {fr['t']:.1f} s with the "
                 f"expanding {_what} keep-out per sampled $t_k$"
                 if (len(segs) or len(dyn_set)) else
                 f"{verdict} — executed trajectory + predicted rollout at "
                 f"t = {fr['t']:.1f} s")
    # Legend OUTSIDE the axes: the zoomed window is short, so "best" placement lands it
    # on top of the rollout every time.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=120)
    plt.close(fig)

    _save_state_plot(stem, verdict, traj, goal_xy, t_mark=fr["t"],
                     solve_ts=[f["t"] for f in frames])

    print(f"[traj] saved {stem}.csv, {stem}.png and {stem}_state.png (rollout from the solve at "
          f"t={fr['t']:.1f}s, {len(set(stages.tolist()))} sampled stages, "
          f"{len(segs)} occlusion boundaries, {len(dyn_set)} dynamic objects)")


def run(unity_exec_path=None, port=5004, run_sysid=True,
        noise_std=0.0, detect_range=float("inf"),
        uncertainty=False, n_scenarios=N_SCEN, w_info=W_INFO,
        d_infl=D_INFL, d_safe=D_SAFE, info_range=INFO_RANGE,
        lidar_costmap=False, lidar_topic="/point_cloud", visibility_cost=False,
        occlusion_aware=False, dynamic_obstacles=False, dynamic_viz=False,
        occlusion_viz=False, save_traj=None, show_occlusion_plot=True, plot_solve_t=None,
        rviz_viz=False, rviz_rollouts=40):
    global DETECTION_RANGE, UNCERTAINTY, N_SCEN, W_INFO
    global D_INFL, D_SAFE, INFO_RANGE, LIDAR_COSTMAP, VISIBILITY_COST, STATIC_AVOID
    global OCCLUSION_AWARE, OCC_TRACKER, OCC_SEGS_NOW, OCC_PUB
    global DYNAMIC_AVOID, DYN_TRACKER, DYN_NOW, DYN_DBG, DYN_PUB, DYN_LAST_STAMP, DYN_CL_N
    global RVIZ_PUB, MPPI_KEEP_ROLLOUTS
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

            if DYNAMIC_AVOID and DYN_TRACKER is not None and LIDAR_COSTMAP.ready:
                _now = time.monotonic()
                _stamp = LIDAR_COSTMAP.stamp
                if _stamp != DYN_LAST_STAMP:
                    DYN_LAST_STAMP = _stamp
                    _cl = LIDAR_COSTMAP.clusters(cell=DYN_CELL, min_points=DYN_MIN_POINTS,
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
  
                if DYN_NOW is not None and len(DYN_NOW):
                    _vv = DYN_TRACKER.velocities()
                    _sg = DYN_TRACKER.vel_sigma()
                    _order = np.argsort(np.hypot(DYN_NOW[:, 0] - s[0],
                                                    DYN_NOW[:, 1] - s[1]))
                    for _rank, _i in enumerate(_order[:5]):
                        _c0, _c1, _r, _age = DYN_NOW[_i]
                        _d = float(np.hypot(_c0 - s[0], _c1 - s[1]))
                        _mark = "<-USED" if _rank < (0 if DYN_USED is None
                                                        else len(DYN_USED)) else ""
                        print(f"           #{_rank} @({_c0:7.1f},{_c1:7.1f}) "
                                f"r={_r:5.1f} d_ego={_d:6.1f} "
                                f"v=({_vv[_i][0]:5.1f},{_vv[_i][1]:5.1f}) "
                                f"|v|={np.hypot(*_vv[_i]):4.1f}+-{_sg[_i]:4.1f} "
                                f"age={_age:4.1f} {_mark}")
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
                               "ex": float(s[0]), "ey": float(s[1]),
                               "plan": OCC_PLAN.copy(),
                               # The occlusion boundaries this solve actually constrained
                               # against (post-gating) — what the cost saw, so the drawn
                               # keep-outs are the enforced ones.
                               "segs": (np.array(_segs, dtype=float)
                                        if _segs is not None and len(_segs) else None),
                               # Ground-truth movers as of THIS solve.
                               "dyn_boxes": (None if dyn_obs is None else dyn_obs.boxes()),
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
                         show_occlusion=show_occlusion_plot, solve_t=plot_solve_t)

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
        rviz_viz=args.rviz_viz,
        rviz_rollouts=args.rviz_rollouts)
