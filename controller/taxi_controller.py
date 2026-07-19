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
from value_net import ValueFunction, terminal_features
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
import quadprog

# ── Parameters — keep in sync with TaxiAgent.cs inspector fields ─────────────
DT        = 0.1          # Fixed Timestep in Unity (Project Settings → Time)
L         = 6.0          # wheelbase [m]
V_DES     = 8.0          # desired taxi speed [m/s]
# Goal-approach deceleration: at V_DES the aircraft's minimum turn radius can exceed the
# remaining distance once it's close to the goal, so it geometrically can't curve tightly enough
# to hit the small arrival capture radius (TaxiAgent's 2 m "reached" threshold) — it sweeps past
# and has to loop back around, which looks like circling/overshoot right at arrival. Taper the
# MPPI speed TARGET down as remaining goal distance shrinks below GOAL_SLOWDOWN_DIST so the turn
# radius shrinks enough for a precise final approach.
GOAL_SLOWDOWN_DIST = 15.0   # [m] remaining distance at which speed target starts tapering
GOAL_MIN_SPEED     = 1.5    # [m/s] speed target floor right at the goal (not 0 — avoid stalling
                            # the bicycle model's steering authority, which rolls off toward 0 speed)
W_HALF    = 10.0         # taxiway half-width [m]
D_SAFE    = 14.0          # keep-out radius [m]
D_INFL    = 30.0         # MPPI obstacle influence radius [m]
UNC_GROWTH  = 1.0        # influence-ring inflation rate [m/s of prediction time], applied ONLY to
                         # frontal blockers being passed (see the deterministic obstacle branch).
                         # The constant-velocity prediction degrades with t, so the pass clearance
                         # grows mildly with prediction time. NOT applied to crossers/convergers:
                         # an inflated ring covers the whole lane and its lateral gradient shoves
                         # the ego off-road when braking in-lane is the right response.
UNC_GROWTH_MAX = 4.0     # cap on the inflation [m] so the ring can't grow unbounded on long horizons
D_INFL_PASS = 15.0       # influence ring for a FRONTAL blocker being passed (go-around) [m]. Must sit
                         # well ABOVE D_SAFE so there's a wide soft band that builds lateral offset
                         # early; if it's only ~1 m above D_SAFE the ego feels the agent only at the
                         # last metre and clips the keep-out. (The capped braking + relaxed lane cost
                         # keep the wider ring from re-causing the freeze.)
HEADON_GIVEWAY = 6.0     # lateral give-way target [m]: the moment an oncoming agent is detected (out
                         # to D_BYPASS), the ego's lane target shifts this far to the side AWAY from
                         # it, so it eases off-centre EARLY and holds that side through the pass —
                         # proactive clearance, rather than reacting only when the ring bites.
A_MIN     = -4.0
A_MAX     =  1.5
DELTA_LIM = 0.5
ALPHA1    = 1.8          # HOCBF first-order gain
ALPHA2    = 1.8          # HOCBF second-order gain
ALPHA_W   = 6.0          # QP acceleration vs steering weight

# CBF Constraint
CBF_MOVER_MIN   = 1.0    # [m/s] below this the obstacle is a static blocker → always engage
CBF_COS_GATE    = 0.5    # engage only if |v_obs·ê|/|v_obs| >= this (i.e. within ~60° of the
                         #   heading axis — head-on/along-track); pure crossers fall below it

# ── Realistic kinematic extensions — must match TaxiAgent.cs inspector values ─
DRAG_COEFF        = 0.04   # aerodynamic + rolling drag [1/s]
ACCEL_TAU         = 0.5    # thrust/brake lag time constant [s]
MAX_STEER_RATE    = 0.6    # nose-wheel steering rate limit [rad/s]
STEER_ROLLOFF_SPD = 15.0   # speed at which steering authority starts rolling off [m/s]
STEER_ROLLOFF_MIN = 0.25   # minimum steering authority fraction at high speed

H_MPPI    = 60           # planning horizon (steps)
K_MPPI    = 1500         # rollout samples
LAMBDA    = 1.0          # MPPI temperature
SIG_A     = 1.0          # noise std for acceleration samples
SIG_D     = 0.70         # noise std for steering samples

# MPPI stage costs
W_LAT, W_HEAD, W_V, W_CTRL = 0.05, 4.0, 1.2, 0.05
# Action cost   ℓact = ||u||²_R + ||Δu||²_RΔ,  u = [a, delta], Δu_k = u_k - u_{k-1}.
#   R  (R_ACT)  — diagonal effort weight: penalises large commands (energy/effort).
#   RΔ (R_DACT) — diagonal rate weight: penalises command CHANGE (smoothness, no jerk).
# Stored as the diagonals of the R / RΔ matrices, per channel [a, delta].
R_ACT  = np.array([0.05, 0.20])
R_DACT = np.array([0.10, 0.40])
# Velocity cost  ℓvel = W_VEL · exp(-C_VEL · d_k²) · ||v_k||²,  d_k = distance to goal, v_k = speed.
# The exp(-C_VEL·d_k²) envelope is ~0 far from the goal (speed unpenalised while transiting) and
# →1 as d_k→0, so the ego is driven to bleed off speed and arrive stopped at the goal. C_VEL sets
# the envelope's half-strength RADIUS: d_half = sqrt(ln2 / C_VEL). 0.02 gives ~5.9m — too late to
# brake a fast-moving plane; 0.003 gives ~15m, matching GOAL_SLOWDOWN_DIST so braking starts with
# enough room. W_VEL scales overall strength — must be large enough to outweigh ℓprogress, which
# unconditionally rewards continued movement (including flying past the goal).
C_VEL  = 0.003
W_VEL  = 3.0
# W_OBS lowered and W_PROG raised (was 12.0 / 0.4) so forward progress competes with the obstacle
# penalty. On a head-on pass the ego can't keep D_SAFE either way (the oncoming agent comes to it),
# so an over-weighted W_OBS made "stop and wait" the cheapest option; a stronger progress pull makes
# driving through win. Trade-off: the ego is slightly less conservative around crossers/convergers.
W_OBS, W_OFF, W_PROG        = 2.0, 2.0, 1.0
# Goal-approach reward (paper's ℓgoal). Per stage k:  ℓgoal = -C_GOAL * max(0, d0 - d_k), with
# d0 = Euclidean distance to the goal at the rollout start and d_k = ||p_k - p_goal|| the true
# Euclidean distance to the goal at stage k. A heavier TERMINAL copy at the last stage H-1
# (ℓgoal,H-1 = -C_GOAL_TERM * max(0, d0 - d_{H-1})) weights the END position of the rollout, so
# a rollout is allowed to detour around an obstacle as long as it ENDS closer to the goal — this
# is what prevents greedy stop-short behaviour. Paper values: C_GOAL 5.0 (goal in line-of-sight)
# or 0.125 (goal occluded → encourage exploration); C_GOAL_TERM 10.0.
C_GOAL                      = 20.0
C_GOAL_TERM                 = 10.0
C_PROGRESS                  = 10.0

# ── Occlusion-aware safety: forward-reachable-set keep-out + RSS sightline speed cap ──
# SHARED with taxi_controller_mpc.py (the MPC imports these), so both controllers use the SAME
# occlusion model and constants. A worst-case hidden agent sits on the FREE↔UNKNOWN frontier (a
# blind corner; LidarCostmap.distance_to_unknown gives the distance to it) and could be anywhere
# within V_TARGET·t_k of it after t_k = (k+1)·DT. Two coupled terms, gated on --occlusion-aware:
#   * expanding keep-out circle:  r_keep = D_SAFE_OCC + V_TARGET·t_k  (soft ring at D_INFL_OCC,
#     steep exact penalty inside r_keep) — the ego swings wide around the blind corner;
#   * RSS sightline speed cap:     v_safe = sqrt(2·A_BRAKE_SIGHT·d_vis), floored at V_SIGHT_FLOOR —
#     the ego slows so it can always stop before the nearest occlusion.
# Keep V_TARGET realistic — too high and the bubble swallows the horizon and the ego freezes.
V_TARGET       = 3.0    # assumed max speed of a hidden agent emerging from occlusion [m/s]
D_SAFE_OCC     = 16.0   # base (t=0) hard keep-out radius around an occlusion boundary [m]
D_INFL_OCC     = 24.0   # base (t=0) soft-influence radius (early deflection). Must stay >= D_SAFE_OCC.
W_OCC          = 20.0   # soft-influence ring penalty weight for occlusion boundaries
W_SIGHT        = 15.0    # sightline over-speed² penalty weight
A_BRAKE_SIGHT  = 0.5    # assumed braking decel for the RSS stopping distance [m/s²] (gentle ⇒ slows early)
V_SIGHT_FLOOR  = 2.5    # never cap the sightline speed below this [m/s]
GOAL_OCC_CLEAR = 31.0   # within this distance of the goal, fade the (repelling) keep-out to 0 — the
                        # goal is known-safe, so the expanding circle must not push the ego off it

N_SCEN = 10
W_INFO = 10
INFO_RANGE = 10
W_TERM = 10

# Lister obstacles cost
LIDAR_COSTMAP  = None    # LidarCostmap instance when --lidar-costmap / --visibility-cost is set
STATIC_AVOID   = False   # set True by --lidar-costmap: adds the static keep-out/soft-ring cost
                         # below to MPPI. Independent of VISIBILITY_COST (--visibility-cost),
                         # which only adds the "peek around blind corners" incentive and does
                         # NOT by itself avoid a collision with a static surface.
W_STATIC       = 20.0     # weight of the static-surface soft influence ring (graded, gives a gradient)
D_SAFE_STATIC  = 10.0     # hard keep-out from any observed static surface [m] — ~1 aircraft width
D_INFL_STATIC  = 14.0      # soft influence ring around static surfaces [m]
# Exact-penalty weights on a keep-out VIOLATION (d < D_SAFE_STATIC), mirroring the MPC's
# RHO_SLACK/RHO_SLACK2 on (d_safe² − d²)_+. Unlike the old flat `BIG`, this is graded and
# grows unbounded (quadratically) as d→0, so it (a) always has a gradient the sampler can
# follow away from the wall — even when every rollout is inside D_INFL — and (b) cannot be
# outweighed by the accumulated goal/progress reward. This is what stops the corner clip.
RHO_STATIC     = 10.0
RHO_STATIC2    = 5.0

OCCLUSION_AWARE = False   # set True by --occlusion-aware (needs --lidar-costmap): enables the
                          # FRS keep-out + sightline speed cap above (mirrors the MPC controller).
VISIBILITY_COST = False
W_VIS           = 20.0   # weight on the per-step hidden-fraction ∈[0,1]. Summed over H_MPPI steps
                         # its worst case is ~H·W_VIS, kept comparable to the goal terms so the pull
                         # to reveal a blind corner competes without overriding goal-seeking. 0=off.

DETECTION_RANGE = float("inf")   # default: oracle (off). Set via --detect-range.

# Number of obstacles packed in the observation vector (must match TaxiAgent K_OBS)
K_OBS    = 3
OBS_SIZE = 4 + K_OBS * 4 + 2 + 2   # 20: ego(4) + K_OBS*4 + goal(1) + cbf_h(1) + tangent(2)

# Scenario types — int value pushed as 'scenario_type' side channel. Also used as the
# RL task_id in the recorded dataset. IDs 0-4 are the original scenarios (shared with
# Unity's ScenarioType enum); 5-7 are appended tasks whose Unity behaviours are added
# in a follow-up (the Python plumbing here is forward-compatible with the append plan).
SCENARIO_STANDARD         = 0   # difficulty-based perpendicular crossing conflict
SCENARIO_HEADON           = 1   # oncoming traffic on the ego's own taxiway
SCENARIO_FOLLOW           = 2   # follow a lead vehicle to the same goal; lead randomly stops/accelerates
SCENARIO_INTERSECTION     = 3   # ego on one taxiway branch, crosser on the other — meet at node
SCENARIO_RUNWAY_INCURSION = 4   # ego drives on a runway; a vehicle holds short and may incur onto it
SCENARIO_CONVERGING       = 5   # agents spawn FAR on a ring and home toward the ego from multiple
                                # bearings — a long-range converging threat (tests early/long-range
                                # planning via D_INFL, not close-in avoidance)
SCENARIO_NAMES = ['standard', 'headon', 'follow_vehicle', 'intersection',
                  'runway_incursion', 'converging']

rng = np.random.default_rng(42)



W_STRAIGHT = 1e10
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

    # ── TEMP sightline diagnostic: is the cap even active at the ego's current pose? ──
    if OCCLUSION_AWARE and LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
        _dv = float(LIDAR_COSTMAP.distance_to_unknown(
            np.array([s0_fwd]), np.array([s0_lat]))[0])
        _vs = max((2.0 * A_BRAKE_SIGHT * _dv) ** 0.5, V_SIGHT_FLOOR)

        print(f"[sightline] ready d_vis(ego)={_dv:8.2f}  v_safe={_vs:5.2f}  v={s0[3]:5.2f}")
    else:
        print(f"[sightline] INACTIVE  ready="
              f"{None if LIDAR_COSTMAP is None else LIDAR_COSTMAP.ready}")

    for k in range(H_MPPI):
        st = _rollout_step(st, na[:, k, 0], na[:, k, 1])
        fwd, lat, th, vv = st[:, 0], st[:, 1], st[:, 2], st[:, 3]

        # Keeping only heading cost at the moment, no lane cost used
        #cost += W_HEAD * th**2

        # ℓgoal = -C_GOAL * max(0, d0 - d_k): reward net closure toward the goal by this stage,
        # d_k = ||p_k - p_goal|| the true Euclidean distance to the goal at stage k.
        d_k = np.hypot(goal_fwd - fwd, goal_lat - lat)
        cost += -C_GOAL * np.maximum(0.0, d0 - d_k)

        u_k   = na[:, k, :]                                  # (K, 2)
        u_km1 = u_prev[None, :] if k == 0 else na[:, k - 1, :]
        du    = u_k - u_km1
        cost += (R_ACT  * u_k**2).sum(axis=1)
        cost += (R_DACT * du**2 ).sum(axis=1)

        # ℓprogress = -C_PROGRESS · ||p_k - p_{k+1}||: reward the distance travelled between
        # consecutive positions (p_k = prev step, p_{k+1} = this step) — rewards making ground.
        cost += -C_PROGRESS * np.hypot(fwd - prev_fwd, lat - prev_lat)
        prev_fwd, prev_lat = fwd, lat

        # ℓvel = W_VEL · exp(-C_VEL · d_k²) · ||v_k||².  d_k already computed above; v_k = speed vv.
        cost += np.exp(-C_VEL * d_k**2) * vv**2

        # ── Occlusion-aware safety (mirrors the MPC): FRS keep-out + RSS sightline cap ──
        # d_occ = distance to the nearest occlusion frontier (FREE↔UNKNOWN edge) at this rollout
        # pose. A worst-case hidden agent could be anywhere within V_TARGET·t_k of that edge by
        # t_k, so an EXPANDING keep-out circle r_keep = D_SAFE_OCC + V_TARGET·t_k (soft ring at
        # D_INFL_OCC) pushes the ego wide, and the sightline cap v_safe = sqrt(2·A_BRAKE_SIGHT·d_occ)
        # slows it so it can stop before the corner. Both use the shared constants (== the MPC).
        if OCCLUSION_AWARE and LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
            d_occ  = LIDAR_COSTMAP.distance_to_unknown(fwd, lat)   # (K,) dist to nearest occlusion [m]
            t_k    = (k + 1) * DT
            r_grow = V_TARGET * t_k
            r_infl = D_INFL_OCC + r_grow
            r_keep = D_SAFE_OCC + r_grow
            # Goal is known-safe: fade the (repelling) keep-out to 0 within GOAL_OCC_CLEAR of the
            # goal so the expanding circle can't push the ego off its own goal (MPPI analogue of
            # the MPC's goal-masking of occlusion points). d_k = dist to goal, computed above.
            goal_fade = np.clip(d_k / GOAL_OCC_CLEAR, 0.0, 1.0)
            cost += goal_fade * W_OCC * np.maximum(0.0, r_infl - d_occ)**2   # soft ring
            violc = np.maximum(0.0, r_keep**2 - d_occ**2)                     # steep keep-out
            cost += goal_fade * (RHO_STATIC * violc + RHO_STATIC2 * violc**2)
            # RSS sightline speed cap (not faded — slowing never repels the ego from the goal).
            v_safe = np.maximum(np.sqrt(2.0 * A_BRAKE_SIGHT * d_occ), V_SIGHT_FLOOR)
            cost  += W_SIGHT * np.maximum(0.0, vv - v_safe)**2

        if STATIC_AVOID and LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
            d_static = LIDAR_COSTMAP.distance(fwd, lat)
            # Soft influence ring: smooth deflection BEFORE the keep-out. Graded, so
            # even when every rollout sits inside D_INFL (rounding a concave corner)
            # the samples closer to the wall cost more and the sampler prefers the
            # ones farther out — the steering signal the old flat `BIG` washed out.
            cost += W_STATIC * np.maximum(0.0, D_INFL_STATIC - d_static)**2
            # Steep exact-penalty keep-out inside D_SAFE (linear + quadratic on the
            # squared-distance violation): unbounded as d→0, so no rollout buys its
            # way through the surface. Mirrors the MPC's static keep-out.
            viol = np.maximum(0.0, D_SAFE_STATIC**2 - d_static**2)
            cost += RHO_STATIC * viol + RHO_STATIC2 * viol**2

        if VISIBILITY_COST and LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready:
            rel_pts = np.stack([fwd - s0_fwd, lat - s0_lat], axis=1)   # (K, 2)
            cost += W_VIS * LIDAR_COSTMAP.hidden_fraction(rel_pts)
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
    d_H1 = np.hypot(goal_fwd - st[:, 0], goal_lat - st[:, 1])
    cost += -C_GOAL_TERM * np.maximum(0.0, d0 - d_H1)

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


# ── Scenario sweep ────────────────────────────────────────────────────────────

DT_SPAN        = 0.8    # was 3.0 — episodes with |Δt|>~1.3s are self-clearing (the
                        # crosser vacates the conflict point before the ego arrives) and
                        # can never collide regardless of controller. Narrowing to ±0.8s
                        # concentrates the sweep on the timing window where genuine
                        # conflicts live, shrinking the "dud" tail that dilutes the rate.
JITTER_DT      = 0.15
SPEED_JIT      = 0.10
BASE_SEED      = 1234
CONFLICT_Z_MAX = 20.0   # max Z shift of conflict point along taxiway [m]

# An episode is a "dud" (no genuine conflict) if no obstacle ever came within this
# distance of the ego — the paths never intersected in space/time, so a collision
# was physically impossible and the episode only dilutes the reported rate. The
# conditional rate (collisions / genuine-conflict episodes) is the meaningful metric.
DUD_DIST = 30.0   # [m] — min_dist above this ⇒ episode excluded from conditional rate

# Lever 1 — speed scales with difficulty ONLY for head-on (longitudinal) conflicts,
# where higher closing speed genuinely compresses the reaction window. For
# PERPENDICULAR crossers (standard) a faster mover spends LESS time in
# the conflict zone (t_window ~ vehicle_size / speed), so speed scaling makes
# crossings EASIER, not harder — confirmed empirically. Crossers stay slow so they
# linger in the conflict zone; difficulty is raised via Lever 2 (compound conflicts)
# instead.
SPEED_CROSS_BASE   = 5.0    # perpendicular crosser speed [m/s] — kept low on purpose
SPEED_HEADON_MIN   = 5.0    # head-on closing speed at difficulty 0 [m/s]
SPEED_HEADON_MAX   = 12.0   # head-on closing speed at difficulty 1 [m/s]
# Distance ahead of the ego where the head-on meet point sits (the oncoming agent spawns further
# ahead still and drives back to it). Larger = agents appear further away, giving the ego more
# reaction room before the pass. Pushed to Unity as 'head_on_gap'; only the head-on scenario reads it.
HEADON_APPROACH_GAP = 90.0

# Converging scenario — homing agents that spawn on a ring around the ego and drive at it.
# Spawn distance is deliberately large so the ego reacts at long range (raise D_INFL to match);
# closing speed ramps modestly with difficulty (faster convergence = tighter reaction window).
CONVERGE_SPAWN_DIST = 70.0  # ring radius [m] agents spawn on around the ego (pushed to Unity)
SPEED_CONVERGE_MIN  = 4.0   # homing speed at difficulty 0 [m/s]
SPEED_CONVERGE_MAX  = 8.0   # homing speed at difficulty 1 [m/s]

# Lever 3 — concentrate Δt near the conflict (was a uniform linspace).
# A power-warped grid keeps the full [-DT_SPAN, DT_SPAN] coverage at the
# extremes but packs most episodes near Δt=0 where genuine conflicts live,
# raising difficulty density and shrinking the no-conflict "dud" tail.
DT_CONCENTRATION = 1.0  # with the narrowed ±0.8s span the whole grid is already in
                        # the conflict window, so no extra warping is needed (1.0=uniform)

# Fix A — episode reach: agents are placed so they arrive at the conflict point
# within egoTtc + dtOffset seconds. Reducing reach shrinks how far ahead
# Unity searches for intersections, so agents spawn CLOSE to the conflict
# (short upstream arc) and actually arrive during the episode.
# At V_DES=8 m/s, 20 s caps intersections to ~160 m ahead — agents at most
# 100-120 m upstream at spd=5 m/s, arriving in ~20-24 s well within timeout.
EPISODE_REACH_SECONDS = 20.0   # pushed to Unity as "episode_reach_seconds" each episode
MAX_GOAL_DIST         = 180.0  # cap the ego goal [m] so long (runway-length) paths don't taxi to
                               # timeout after the encounter. Pushed to Unity as "max_goal_dist".


def make_scenarios(n_episodes, base_seed=BASE_SEED, min_difficulty=0.0, max_difficulty=1.0,
                   scenario_type=SCENARIO_STANDARD, spawn_heading_deg=None,
                   spawn_heading_jitter_deg=0.0):
    """
    Build a reproducible list of per-episode scenario parameter dicts.

    Each dict is pushed to Unity via EnvironmentParametersChannel before env.reset().
    Keys:
      incursion_dt    — Δt offset for the primary (agent[0]) incursion [s]
      ambulance_speed — crossing speed for agent[0] [m/s]
      difficulty      — [0, 1] curriculum difficulty sent to ScenarioManager

    min_difficulty / max_difficulty clamp the ramp so you can test a specific
    difficulty band. E.g. min_difficulty=0.85 forces 3-agent Erratic scenarios.

    spawn_heading_deg (optional): fixed ego spawn heading [deg, world yaw] at
    the start marker. When None (default), the key is left out of the dict
    entirely and Unity falls back to its own spawn behaviour (identity heading
    in the two-point-sandbox / scenario-manager path). When set, every episode
    spawns the ego rotated to this heading, +/- a per-episode random jitter of
    spawn_heading_jitter_deg (still seeded off episode_seed, so reproducible).
    """
    # Lever 3: power-warp a uniform [-1, 1] grid so density concentrates near 0
    # while the extremes still reach ±DT_SPAN (full timing coverage preserved).
    u    = np.linspace(-1.0, 1.0, n_episodes)
    grid = DT_SPAN * np.sign(u) * np.abs(u) ** DT_CONCENTRATION
    scenarios = []
    for i, dt in enumerate(grid):
        r = np.random.default_rng(base_seed + i)
        # Difficulty is drawn INDEPENDENTLY of Δt so the (Δt × difficulty) plane is
        # actually sampled, not walked along its diagonal. (Previously difficulty
        # ramped with the episode index, perfectly correlating it with Δt and making
        # it impossible to attribute a failure to timing vs. scenario complexity.)
        # The Δt grid still spans [-DT_SPAN, DT_SPAN] for full timing coverage.
        difficulty = float(r.uniform(min_difficulty, max_difficulty))
        # Scenario type: forced when scenario_type >= 0, else drawn per-episode from the
        # active list ("mixed" mode, scenario_type = -1). Runway incursion is excluded from
        # mixed mode — it needs runway geometry that only some episodes can place — so run it
        # explicitly via --scenario runway_incursion. Converging is likewise explicit-only: it is
        # a distinct long-range regime best run with an enlarged D_INFL (--d-infl).
        _active_types = [SCENARIO_STANDARD, SCENARIO_HEADON,
                         SCENARIO_FOLLOW, SCENARIO_INTERSECTION]
        stype = float(scenario_type) if scenario_type >= 0 else float(r.choice(_active_types))
        desired_spd = -1.0
        # Lever 1: only head-on closing speed ramps with difficulty; perpendicular
        # crossers stay slow (so they linger in the conflict zone).
        if stype == SCENARIO_HEADON:
            spd_base = SPEED_HEADON_MIN + (SPEED_HEADON_MAX - SPEED_HEADON_MIN) * difficulty
        elif stype == SCENARIO_CONVERGING:
            spd_base = SPEED_CONVERGE_MIN + (SPEED_CONVERGE_MAX - SPEED_CONVERGE_MIN) * difficulty
        else:
            spd_base = SPEED_CROSS_BASE
        scenarios.append({
            "episode_seed":      float(base_seed + i),   # deterministic Unity path/obstacle selection
            "incursion_dt":      float(dt + r.uniform(-JITTER_DT, JITTER_DT)),
            "ambulance_speed":   float(spd_base * (1.0 + r.uniform(-SPEED_JIT, SPEED_JIT))),
            "difficulty":        difficulty,
            "conflict_z_offset": float(r.uniform(-CONFLICT_Z_MAX, CONFLICT_Z_MAX)),
            "cross_dir_sign":    float(r.choice([-1.0, 1.0])),
            "scenario_type":     stype,
            "desired_speed":     desired_spd,
            "head_on_prob":           0.3 if stype == SCENARIO_HEADON else 0.0,
            "episode_reach_seconds":  EPISODE_REACH_SECONDS,
            "max_goal_dist":          MAX_GOAL_DIST,         # cap ego goal so long paths don't run to timeout
            "converge_spawn_dist":    CONVERGE_SPAWN_DIST,   # ring radius; only used by SCENARIO_CONVERGING
            "head_on_gap":            HEADON_APPROACH_GAP,   # meet/spawn distance; only used by SCENARIO_HEADON
        })
        if spawn_heading_deg is not None:
            jitter = float(r.uniform(-spawn_heading_jitter_deg, spawn_heading_jitter_deg))
            scenarios[-1]["spawn_heading_deg"] = float(spawn_heading_deg) + jitter
    return scenarios


# ── Main control loop ─────────────────────────────────────────────────────────

def _save_trajectory(out_dir, ep, ep_log, goal_xy, traj, obs_track=None, occ_pts=None,
                     occ_boundary=None, occ_horizon_s=None, occ_max_circles=16,
                     occ_n_rings=4, occ_cluster_m=D_SAFE_OCC):
    """Persist one episode's ego trajectory as CSV + a top-down PNG plot.

    traj columns: t, x, y, theta, v, a_cmd, delta_cmd  (x=Unity Z, y=Unity X).
    obs_track (optional): (N, 3) array of per-step dynamic-obstacle world
    positions (t, x, y), same frame as traj — overlaid on the plot.
    occ_pts (optional): (M, 2) array of static occluder (OCC) cell centres
    (x, y) from the LiDAR costmap, same frame — the buildings/walls.
    occ_boundary (optional): (K, 2) array of occlusion-boundary (blind-spot)
    points the occlusion-aware controller actually constrained (x, y), same
    frame — each seeds an expanding forward-reachable-set keep-out.
    occ_horizon_s (optional): planning horizon [s] (N_steps * DT) used to grow
    the keep-out — pass this to also draw the EXPANDING circle r_keep =
    D_SAFE_OCC + V_TARGET*t at occ_n_rings times spread across [0, occ_horizon_s].
    The dense occlusion-boundary cloud is first clustered onto a coarse grid of
    cell size occ_cluster_m and ONE ring-set is drawn per cluster centroid (capped
    at occ_max_circles), so the plot reads like the paper's per-obstacle capsule
    sets instead of a solid band of overlapping circles along the wall.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    verdict = "collision" if ep_log["collided"] else ("reached" if ep_log["reached"] else "timeout")
    stem = os.path.join(out_dir, f"ep{ep+1:03d}_{ep_log['scenario']}_{verdict}")

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
    if occ_boundary is not None and len(occ_boundary):
        ax.scatter(occ_boundary[:, 0], occ_boundary[:, 1], c="tab:purple", s=14,
                   marker="D", alpha=0.5, label="occlusion boundaries", zorder=2)
        if occ_horizon_s:
            from matplotlib.patches import Circle
            # The occlusion-boundary set is a DENSE cloud (hundreds of cells strung
            # along the wall), so drawing rings on every point — or a raw decimation —
            # smears into overlapping tubes. Cluster onto a coarse grid (one cell ≈
            # occ_cluster_m) and draw ONE concentric ring-set per cluster centroid, so
            # the plot reads like the paper's per-obstacle capsule sets rather than a
            # solid band. (When the detection is fixed to fire only at the true corner,
            # this collapses to a single clean ring-set there.)
            pts = np.asarray(occ_boundary)
            cell = max(1e-3, occ_cluster_m)
            keys = np.round(pts / cell).astype(np.int64)
            uniq, inv = np.unique(keys, axis=0, return_inverse=True)
            cents = np.array([pts[inv == u].mean(axis=0) for u in range(len(uniq))])
            if len(cents) > occ_max_circles:
                sel = np.linspace(0, len(cents) - 1, occ_max_circles).astype(int)
                cents = cents[sel]
            pts = cents
            times = np.linspace(0.0, occ_horizon_s, occ_n_rings)
            cmap = plt.get_cmap("autumn_r")
            for ti, t in enumerate(times):
                r = D_SAFE_OCC + V_TARGET * t
                color = cmap(0.15 + 0.85 * ti / max(1, occ_n_rings - 1))
                for px, py in pts:
                    ax.add_patch(Circle((px, py), r, fill=False, lw=0.6, alpha=0.35,
                                         edgecolor=color, zorder=1.5))
                ax.plot([], [], "-", color=color, lw=1.5,
                        label=f"occ keep-out t={t:.1f}s (r={r:.0f}m)")
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
    ax.set_title(f"Ep {ep+1} — {ep_log['scenario']} — {verdict}")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=120)
    plt.close(fig)
    print(f"[traj] saved {stem}.csv and {stem}.png")


def run(unity_exec_path=None, port=5004, run_sysid=True, n_episodes=20,
        min_difficulty=0.0, max_difficulty=1.0,
        noise_std=0.0, scenario_type=SCENARIO_STANDARD, detect_range=float("inf"), dataset_path=None,
        uncertainty=False, n_scenarios=N_SCEN, w_info=W_INFO,
        d_infl=D_INFL, d_safe=D_SAFE, info_range=INFO_RANGE,
        value_net_path=None, w_term=W_TERM, pin_episode=None,
        lidar_costmap=False, lidar_topic="/point_cloud", visibility_cost=False,
        occlusion_aware=False, save_traj=None,
        spawn_heading_deg=None, spawn_heading_jitter_deg=0.0):
    global DETECTION_RANGE, UNCERTAINTY, N_SCEN, W_INFO
    global D_INFL, D_SAFE, INFO_RANGE, VALUE_NET, W_TERM, LIDAR_COSTMAP, VISIBILITY_COST, STATIC_AVOID
    global OCCLUSION_AWARE
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
    if lidar_costmap or visibility_cost:
        VISIBILITY_COST = visibility_cost
        STATIC_AVOID    = lidar_costmap
        OCCLUSION_AWARE = occlusion_aware and lidar_costmap   # occlusion terms need the map
        from lidar_costmap import LidarCostmap
        # max_age raised from the 0.5s default: Unity's PointCloudPublisher is configured well
        # below 10Hz (measured ~1Hz cloud_age via [DEBUG lidar]), so 0.5s made `ready` permanently
        # False and the static-avoidance/collision cost never activated. 1.5s covers ~1Hz publish
        # with margin; lower it again if the Unity publish rate is raised instead.
        cm = LidarCostmap(max_age=1.5, enable_visibility=visibility_cost)
        if cm.start(topic=lidar_topic):
            LIDAR_COSTMAP = cm
            feats = []
            if lidar_costmap:    feats.append(f"static(D_SAFE={D_SAFE_STATIC:.1f}m)")
            if OCCLUSION_AWARE:  feats.append(f"occlusion(D_SAFE_OCC={D_SAFE_OCC:.1f}m, "
                                             f"v_target={V_TARGET:.1f}m/s, sightline)")
            if visibility_cost:  feats.append(f"visibility(W_VIS={W_VIS:.1f})")
            print(f"[Controller] LiDAR map     : ON  (topic={lidar_topic}) — {' + '.join(feats)} "
                  f"in the MPPI cost; persistent 3-state map")
        else:
            VISIBILITY_COST = False
            STATIC_AVOID    = False
            OCCLUSION_AWARE = False
            print("[Controller] LiDAR map     : requested but unavailable (no rclpy) — "
                  "running WITHOUT LiDAR-based costs")
    elif occlusion_aware:
        print("[Controller] Occlusion-aware : requested but needs --lidar-costmap — DISABLED")
    env_params = EnvironmentParametersChannel()
    env = UnityEnvironment(
        file_name=unity_exec_path,
        base_port=port,
        seed=42,
        no_graphics=unity_exec_path is not None,  # headless when running a build
        side_channels=[env_params],
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

    scenarios     = make_scenarios(n_episodes, min_difficulty=min_difficulty,
                                              max_difficulty=max_difficulty,
                                              scenario_type=scenario_type,
                                              spawn_heading_deg=spawn_heading_deg,
                                              spawn_heading_jitter_deg=spawn_heading_jitter_deg)
    # --pin-episode: replay ONE fixed scenario every episode. The pinned episode_seed makes
    # Unity's per-episode RNG (ego path pick, obstacle assignment, timings) fully deterministic,
    # so the ego spawns at the SAME map location each episode — for hand-placing occluders /
    # buildings in the editor and iterating on one repeatable situation.
    if pin_episode is not None:
        pinned = make_scenarios(1, base_seed=pin_episode,
                                min_difficulty=min_difficulty,
                                max_difficulty=max_difficulty,
                                scenario_type=scenario_type,
                                spawn_heading_deg=spawn_heading_deg,
                                spawn_heading_jitter_deg=spawn_heading_jitter_deg)[0]
        scenarios = [dict(pinned) for _ in range(n_episodes)]
        print(f"[Controller] Pinned episode  : seed={pin_episode} — identical scenario every "
              f"episode (ego path, obstacles, timings all repeat)")
    episode_stats = []
    # Unity (Editor or build) can exit mid-run — most often the Editor drops Play mode because a
    # script recompiled, or the window was closed. That surfaces as UnityCommunicatorStoppedException
    # from env.step(); catch it so we still print the summary for the episodes that DID complete
    # instead of dying with a traceback and losing the recorded dataset.
    unity_stopped = False
    for ep in range(n_episodes):
        mean   = np.zeros((H_MPPI, 2))
        sc     = scenarios[ep]
        task_id = int(sc["scenario_type"])
        ep_log = {"min_h": np.inf, "min_dist": np.inf,
                  "collided": False, "reached": False, "steps": 0,
                  "incursion_dt": sc["incursion_dt"],
                  "difficulty": sc["difficulty"],
                  "scenario": SCENARIO_NAMES[task_id]}
        traj = []   # per-step ego pose log for this episode: (t, x, y, theta, v, a_cmd, delta_cmd)
        obs_track = []   # per-step obstacle world positions: list of (t, x, y) rows
        # Union across the episode (deduped by rounded world cell) of the static
        # occluders and occlusion boundaries the perception produced. The OA-MPC
        # perception is frame-based (no memory), so a single end-of-run snapshot only
        # shows what is visible at the goal — accumulate to show the full extent the
        # ego actually reacted to, matching the MPC controller's trajectory plot.
        occ_seen = {}    # static occluder points  → plotted as "static occluders"
        occ_track = {}   # occlusion boundary points → plotted as "occlusion boundaries"
        prev_obs = None
        prev_act = None

        for key, val in sc.items():
            env_params.set_float_parameter(key, val)

        sname = SCENARIO_NAMES[int(sc["scenario_type"])]
        print(f"\n[Ep {ep+1:3d}] [{sname}] Δt={sc['incursion_dt']:+.2f}s  "
              f"v_amb={sc['ambulance_speed']:.2f} m/s  "
              f"diff={sc['difficulty']:.2f}  "
              f"noise={noise_std:.3f}  "
              f"dir={'L' if sc['cross_dir_sign'] < 0 else 'R'}")

        try:
            env.reset()
        except UnityCommunicatorStoppedException:
            unity_stopped = True
            break
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        ep_steps     = 0
        delta_actual = 0.0   # tracked Python-side to feed into MPPI state
        accel_actual = 0.0
        u_prev       = np.zeros(2)   # last applied [a, delta], for the MPPI Δu smoothness cost
        episode_done = False
        # Re-seed MPPI per episode so CBF vs no-CBF runs are directly comparable.
        rng = np.random.default_rng(BASE_SEED + ep)

        while not episode_done:
            # ── Terminal step (episode just ended) ──────────────────────────────
            if len(terminal_steps) > 0:
                ep_log["collided"] = ep_log["min_h"] < 0.
                ep_log["reached"]  = not terminal_steps.interrupted[0] and not ep_log["collided"]
                # Close out the final transition with the terminal obs + terminal reward.
                if prev_obs is not None:
                    term_reward = -20.0 if ep_log["collided"] else 10.0
                    final_obs   = terminal_steps.obs[0][0]
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
                # Path-relevance gate: keep only occlusion shadow within a corridor of
                # the ego→goal route (drops the wall driven alongside / beside the goal).
                if goal_xy is not None:
                    LIDAR_COSTMAP.set_goal(goal_xy[0], goal_xy[1])
                LIDAR_COSTMAP.update(ego_fwd)
                # Accumulate the persistent-grid perception into the episode-wide union
                # (deduped by rounded world cell) for the trajectory plot: OCC cells as
                # static occluders, occlusion_points() as blind-corner boundaries.
                if save_traj is not None and LIDAR_COSTMAP.ready:
                    g = LIDAR_COSTMAP.grid
                    oi, oj = np.where(g.state == 2)
                    for wx, wy in zip(g._o0 + (oi + 0.5) * g.res, g._o1 + (oj + 0.5) * g.res):
                        occ_seen[(round(float(wx), 1), round(float(wy), 1))] = None
                    op = LIDAR_COSTMAP.occlusion_points()
                    if op is not None:
                        for wx, wy in op:
                            occ_track[(round(float(wx), 1), round(float(wy), 1))] = None
            u_nom, mean = mppi(s, mean, obstacles, goal_xy, u_prev)

            u_cmd      = u_nom
            cbf_engaged = False

            a_cmd     = float(np.clip(u_cmd[0], A_MIN, A_MAX))
            delta_cmd = float(np.clip(u_cmd[1], -DELTA_LIM, DELTA_LIM))
            u_prev    = np.array([a_cmd, delta_cmd])   # feed the Δu smoothness cost next step

            if LIDAR_COSTMAP is not None and LIDAR_COSTMAP.ready and ep_steps % 20 == 0:
                g = LIDAR_COSTMAP.grid
                oi, oj = np.where(g.state == 2)                     # OCC cells
                op = LIDAR_COSTMAP.occlusion_points()
                n_occ = 0 if op is None else len(op)
                if len(oi):
                    w0 = g._o0 + (oi + 0.5) * g.res
                    w1 = g._o1 + (oj + 0.5) * g.res
                    d_true  = np.min(np.hypot(w0 - s[0], w1 - s[1]))
                    d_field = LIDAR_COSTMAP.distance(np.array([s[0]]), np.array([s[1]]))[0]
                    k = np.argmin(np.hypot(w0 - s[0], w1 - s[1]))
                    print(f"[CHK] ego=({s[0]:.1f},{s[1]:.1f})  nearest OCC=({w0[k]:.1f},{w1[k]:.1f})  "
                        f"d_true={d_true:.1f}  d_field={d_field:.1f}  nOCC={len(oi)}  nOcc_bnd={n_occ}")


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
            ep_log["min_h"]    = min(ep_log["min_h"], h_val)
            ep_log["min_dist"] = min(ep_log["min_dist"], dist)
            ep_log["steps"]    = ep_steps


            # Record ego pose this step (world frame: x=Unity Z, y=Unity X).
            traj.append((ep_steps * DT, s[0], s[1], s[2], s[3], a_cmd, delta_cmd))
            # Record obstacle world positions this step (rel_xy is ego-relative world coords).
            for rel, _ov in obstacles:
                obs_track.append((ep_steps * DT, s[0] + rel[0], s[1] + rel[1]))

            action = ActionTuple(
                continuous=np.array([[a_cmd, delta_cmd]], dtype=np.float32)
            )
            prev_obs = obs_n.copy()
            prev_act = np.array([a_cmd, delta_cmd], dtype=np.float32)


            env.set_actions(behavior_name, action)
            try:
                env.step()
            except UnityCommunicatorStoppedException:
                unity_stopped = True
                break
            decision_steps, terminal_steps = env.get_steps(behavior_name)

        if unity_stopped:      # Unity exited mid-episode — don't log a partial episode
            break
        episode_stats.append(ep_log)

        if save_traj is not None and traj:
            # Overlay the episode-wide UNION of static occluders and occlusion
            # boundaries the perception produced (not a single decaying end-of-run
            # frame), so the plot shows every wall/blind corner the ego reacted to —
            # matching the MPC controller's trajectory plot. obs_track dynamic slots
            # are often empty when the crosser never enters range.
            occ_pts      = np.array(list(occ_seen.keys()))  if occ_seen  else None
            occ_boundary = np.array(list(occ_track.keys())) if occ_track else None
            _save_trajectory(save_traj, ep, ep_log, goal_xy, np.asarray(traj),
                             np.asarray(obs_track) if obs_track else None,
                             occ_pts, occ_boundary=occ_boundary,
                             occ_horizon_s=H_MPPI * DT)

        verdict = "COLLISION" if ep_log["collided"] else "safe"
        print(f"[Ep {ep+1:3d}] Δt={ep_log['incursion_dt']:+.2f}s  "
              f"diff={ep_log['difficulty']:.2f}  "
              f"steps={ep_log['steps']:4d}  "
              f"min_dist={ep_log['min_dist']:5.2f}m  "
              f"min_h={ep_log['min_h']:8.2f}  → {verdict}")

    if unity_stopped:
        print(f"\n[Controller] Unity stopped early (Editor left Play mode, or the app closed). "
              f"Completed {len(episode_stats)}/{n_episodes} episodes — reporting those.\n"
              f"             If this was unexpected: check the Unity Console for an exception, and "
              f"avoid editing C# scripts while the Editor is in Play mode (that triggers a recompile "
              f"and drops the connection).")
    try:
        env.close()
    except Exception:
        pass

    print("\n=== Summary ===")
    n   = len(episode_stats)
    if n == 0:
        print("No episodes completed — nothing to summarise.")
        return
    col = sum(1 for e in episode_stats if e["collided"])
    dists = [e["min_dist"] for e in episode_stats]
    print(f"Collision rate : {col}/{n} = {col/n:.1%}")

    # ── Conditional rate: exclude duds (paths never intersected) ───────────────
    genuine = [e for e in episode_stats if e["min_dist"] <= DUD_DIST]
    n_gen   = len(genuine)
    col_gen = sum(1 for e in genuine if e["collided"])
    n_dud   = n - n_gen
    if n_gen:
        print(f"Conditional    : {col_gen}/{n_gen} = {col_gen/n_gen:.1%}  "
              f"(genuine conflicts only; {n_dud} duds with min_dist>{DUD_DIST:.0f}m excluded)")
    else:
        print(f"Conditional    : n/a (all {n} episodes were duds, min_dist>{DUD_DIST:.0f}m)")

    print(f"Mean min_dist  : {np.mean(dists):.2f} m  "
          f"(median {np.median(dists):.2f} m — use median; no-conflict duds skew the mean)")
    print(f"Worst min_dist : {np.min(dists):.2f} m "
          f"(target >= {D_SAFE:.1f} m)")
    print(f"Worst min_h    : {np.min([e['min_h'] for e in episode_stats]):.2f} "
          "(target >= 0)")

    # ── Per-scenario-type breakdown ───────────────────────────────────────────
    print("\n  scenario      episodes  collisions   rate    genuine  cond.rate   median min_dist[m]")
    for name in SCENARIO_NAMES:
        grp = [e for e in episode_stats if e["scenario"] == name]
        if not grp:
            continue
        g_col   = sum(1 for e in grp if e["collided"])
        g_med   = np.median([e["min_dist"] for e in grp])
        grp_gen = [e for e in grp if e["min_dist"] <= DUD_DIST]
        gc_col  = sum(1 for e in grp_gen if e["collided"])
        cond    = f"{gc_col/len(grp_gen):5.1%}" if grp_gen else "  n/a"
        print(f"  {name:<12s}  {len(grp):8d}  {g_col:10d}   {g_col/len(grp):5.1%}   "
              f"{len(grp_gen):7d}    {cond}   {g_med:14.2f}")

    print("\n  Δt[s]  scenario      diff  min_dist[m]   min_h     result")
    for e in sorted(episode_stats, key=lambda d: d["incursion_dt"]):
        print(f"  {e['incursion_dt']:+5.2f}  {e['scenario']:<12s}  {e['difficulty']:.2f}  "
              f"{e['min_dist']:8.2f}   {e['min_h']:8.2f}   "
              f"{'COLLISION' if e['collided'] else 'safe'}")

    return episode_stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exec",           default=None)
    p.add_argument("--port",           default=5004,  type=int)
    p.add_argument("--sysid",          default=False,  type=lambda x: x.lower() == "true")
    p.add_argument("--episodes",       default=20,    type=int)
    p.add_argument("--min-difficulty", default=0.0,   type=float)
    p.add_argument("--max-difficulty", default=1.0,   type=float)
    p.add_argument("--noise-std",      default=0.0,   type=float,
                   help="Std-dev of Gaussian noise injected into obstacle obs [m]. 0=off.")
    p.add_argument("--scenario",       default="standard",
                   choices=SCENARIO_NAMES + ["mixed"],
                   help="Force a specific scenario type for all episodes, or "
                        "'mixed' to randomise across the active crossing-conflict types "
                        "(standard, headon, follow_vehicle, intersection) per episode.")
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
                        "planning, e.g. the 'converging' scenario. Must stay >= --d-safe.")
    p.add_argument("--d-safe",         default=D_SAFE, type=float,
                   help="Hard keep-out radius [m] for the CBF barrier and the BIG close-in penalty. "
                        "Governs close-in avoidance; leave at default unless you want a larger "
                        "physical standoff. Must stay <= --d-infl.")
    p.add_argument("--info-range",     default=INFO_RANGE, type=float,
                   help="Radius [m] of the uncertainty (belief-entropy) caution term. Raise it to "
                        "match --d-infl so the ego slows for ambiguous distant threats. Only used "
                        "with --uncertainty.")
    p.add_argument("--value-net",      default=None,
                   help="Path to a value_net checkpoint (.npz from train_value.py). Enables the "
                        "learned terminal cost-to-go V(s,b) at MPPI rollout endpoints.")
    p.add_argument("--w-term",         default=W_TERM, type=float,
                   help="Weight of the learned terminal value term. 0 disables it even with "
                        "--value-net loaded; sweep upward from 0.5. Only used with --value-net.")
    p.add_argument("--pin-episode",    default=None, type=int, metavar="SEED",
                   help="Replay one fixed scenario every episode (same ego spawn/path, same "
                        "obstacles, same timings). Use to hand-place occluders/buildings at a "
                        "known map location in the Unity editor and test against it repeatedly. "
                        "Try different SEED values to pick a location you like.")
    p.add_argument("--lidar-costmap",  action="store_true",
                   help="Build a static-obstacle distance field from the published PointCloud2 "
                        "each control step and add it to the MPPI cost — walls/parked aircraft/"
                        "buildings get a uniform margin (D_SAFE_STATIC). Needs ROS 2 sourced "
                        "(rclpy) and the ros_tcp_endpoint running.")
    p.add_argument("--lidar-topic",    default="/point_cloud",
                   help="PointCloud2 topic for the LiDAR map. Default: /point_cloud.")
    p.add_argument("--visibility-cost", action="store_true",
                   help="Active-perception term: from the persistent 3-state map, penalise "
                        "rollout endpoints from which occluded path-relevant space stays hidden, "
                        "so the ego arcs wider to see into blind corners (self-terminating via "
                        "memory + decay). Enables the LiDAR map; needs ROS 2 sourced.")
    p.add_argument("--occlusion-aware", action="store_true",
                   help="Add occlusion-aware forward-reachable-set keep-outs + an RSS sightline "
                        "speed cap (same model/constants as the MPC controller): a worst-case "
                        "hidden agent on each blind-corner frontier seeds an EXPANDING keep-out "
                        "(radius grows as v_target·t) and the ego is speed-capped so it can stop "
                        "before the nearest occlusion. Requires --lidar-costmap.")
    p.add_argument("--save-traj", default=None, metavar="DIR",
                   help="Save each episode's ego trajectory as CSV + a top-down PNG plot into DIR.")
    p.add_argument("--spawn-heading-deg", default=None, type=float, metavar="DEG",
                   help="Fixed ego spawn heading [deg, world yaw] at the start marker. Omit to keep "
                        "Unity's default spawn behaviour (identity heading).")
    p.add_argument("--spawn-heading-jitter-deg", default=0.0, type=float, metavar="DEG",
                   help="+/- random jitter added to --spawn-heading-deg each episode (reproducible, "
                        "seeded off episode_seed). No effect unless --spawn-heading-deg is set.")
    args = p.parse_args()

    if args.d_infl < args.d_safe:
        p.error(f"--d-infl ({args.d_infl}) must be >= --d-safe ({args.d_safe}): "
                "the soft influence ring cannot be inside the hard keep-out radius.")

    sc_int = -1 if args.scenario == "mixed" else SCENARIO_NAMES.index(args.scenario)
    run(unity_exec_path=args.exec if args.exec != "None" else None,
        port=args.port,
        run_sysid=args.sysid,
        n_episodes=args.episodes,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        noise_std=args.noise_std,
        scenario_type=sc_int,
        detect_range=args.detect_range,
        n_scenarios=args.n_scenarios,
        w_info=args.w_info,
        d_infl=args.d_infl,
        d_safe=args.d_safe,
        info_range=args.info_range,
        value_net_path=args.value_net,
        w_term=args.w_term,
        pin_episode=args.pin_episode,
        lidar_costmap=args.lidar_costmap,
        lidar_topic=args.lidar_topic,
        visibility_cost=args.visibility_cost,
        occlusion_aware=args.occlusion_aware,
        save_traj=args.save_traj,
        spawn_heading_deg=args.spawn_heading_deg,
        spawn_heading_jitter_deg=args.spawn_heading_jitter_deg)
