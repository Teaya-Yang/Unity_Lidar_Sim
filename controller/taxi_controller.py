"""
taxi_controller.py
==================
External Python controller for the Unity taxiing environment.
Connects via the ML-Agents Python API, runs MPPI + HOCBF-QP, and sends
[a, delta] actions back to Unity each decision step.

Observation contract (must match TaxiAgent.cs CollectObservations, OBS_SIZE=20):

  WITHOUT TaxiwayNetwork (global frame):
    obs[0]   x_ego    — Unity Z position [m]
    obs[1]   y_ego    — Unity X position [m]
    obs[2]   theta    — heading [rad]
    obs[18]  0.0      (tangent_x stub)
    obs[19]  1.0      (tangent_z stub)

  WITH TaxiwayNetwork (Frenet frame):
    obs[0]   s        — arc-length along ego path [m]
    obs[1]   d        — signed cross-track error [m]  (+ = left)
    obs[2]   theta_e  — heading error vs path tangent [rad]
    obs[18]  tangent_x — world X component of path tangent (Unity X)
    obs[19]  tangent_z — world Z component of path tangent (Unity Z)

  Both modes (common):
    obs[3]   v        — speed [m/s]
    obs[4..15]        3 × obstacle (dx_global, dy_global, vx, vy) — 12 floats
    obs[16]  goal     — remaining distance/arc to goal [m]
    obs[17]  cbf_h    — barrier value h = dist^2 - D^2 of nearest obstacle

  FRENET MODE DETECTION: abs(obs[18]) + abs(obs[19]) > 0.01 AND obs[19] != 1.0

Action contract:
  act[0]  a_cmd     — acceleration [m/s^2], clipped to [A_MIN, A_MAX]
  act[1]  delta_cmd — steering [rad],       clipped to [-DELTA_LIM, DELTA_LIM]

MPPI IN FRENET MODE:
  Uses a local linear path approximation (zero curvature between steps).
  State [0] = arc-length progress Δs, [1] = cross-track error d.
  Valid for airport taxiway curvatures over a 2.5 s horizon.
  CBF uses Euclidean obstacle distances and is frame-independent.
"""

import numpy as np
import argparse
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
import quadprog

# ── Parameters — keep in sync with TaxiAgent.cs inspector fields ─────────────
DT        = 0.1          # Fixed Timestep in Unity (Project Settings → Time)
L         = 6.0          # wheelbase [m]
V_DES     = 8.0          # desired taxi speed [m/s] 
W_HALF    = 10.0         # taxiway half-width [m]
D_SAFE    = 14.0          # keep-out radius [m]
D_INFL    = 16.0         # MPPI obstacle influence radius [m]
A_MIN     = -4.0
A_MAX     =  1.5
DELTA_LIM = 0.5
ALPHA1    = 1.8          # HOCBF first-order gain
ALPHA2    = 1.8          # HOCBF second-order gain
ALPHA_W   = 6.0          # QP acceleration vs steering weight

# Geometry gate for the CBF constraint — engage braking only when it can actually help.
# The discriminator is the OBSTACLE VELOCITY direction relative to the ego heading ê,
# which identifies the hazard TYPE and is stable across the whole encounter (unlike the
# instantaneous position, which for a crosser points straight ahead at closest approach
# — opening a position-based gate at exactly the wrong instant):
#   • CROSSER  — |v_obs·ê|/|v_obs| small (velocity ⟂ heading): self-clearing, braking is
#                impotent and only phase-shifts the ego INTO the crossing → SKIP.
#   • HEAD-ON  — v_obs·ê strongly negative (closing along heading): braking opens
#                distance → ENGAGE.
#   • BLOCKER  — |v_obs| ≈ 0 and obstacle ahead (parked/stationary): persistent → ENGAGE.
CBF_MOVER_MIN   = 1.0    # [m/s] below this the obstacle is a static blocker → always engage
CBF_COS_GATE    = 0.5    # engage only if |v_obs·ê|/|v_obs| >= this (i.e. within ~60° of the
                         #   heading axis — head-on/along-track); pure crossers fall below it

# ── Realistic kinematic extensions — must match TaxiAgent.cs inspector values ─
DRAG_COEFF        = 0.04   # aerodynamic + rolling drag [1/s]
ACCEL_TAU         = 0.5    # thrust/brake lag time constant [s]
MAX_STEER_RATE    = 0.6    # nose-wheel steering rate limit [rad/s]
STEER_ROLLOFF_SPD = 15.0   # speed at which steering authority starts rolling off [m/s]
STEER_ROLLOFF_MIN = 0.25   # minimum steering authority fraction at high speed

H_MPPI    = 25           # planning horizon (steps)
K_MPPI    = 1500         # rollout samples
LAMBDA    = 1.0          # MPPI temperature
SIG_A     = 1.0          # noise std for acceleration samples
SIG_D     = 0.35         # noise std for steering samples

# MPPI stage costs
W_LAT, W_HEAD, W_V, W_CTRL = 3.0, 6.0, 1.2, 0.05
W_OBS, W_OFF, W_PROG        = 12.0, 2.0, 0.4
W_OFF_BYPASS = 2.0    # relaxed lane penalty when a blocking obstacle is ahead
D_BYPASS     = 35.0   # trigger range: on-lane obstacle within this distance [m]
LAT_GOAROUND = 1.5    # lateral offset [m] at which ego is considered committed to a go-around
BIG = 300.0

SIG_D_BYPASS  = 0.35  # wide steering noise to sample go-around rollouts
ALPHA_W_GOAROUND = 0.2  # QP steering weight during go-around (vs ALPHA_W=6 normally)

# Detection range — obstacles beyond this Euclidean distance [m] are masked from
# MPPI and CBF, simulating finite LiDAR/sensor range. Oracle = inf (old behaviour).
# At V_DES=8 m/s the MPPI horizon covers 20 m; a 25 m range gives the planner ~0.6 s
# of preview beyond the horizon — reaction becomes tight for fast/late detections.
DETECTION_RANGE = float("inf")   # default: oracle (off). Set via --detect-range.

# Number of obstacles packed in the observation vector (must match TaxiAgent K_OBS)
K_OBS    = 3
OBS_SIZE = 4 + K_OBS * 4 + 2 + 2   # 20: ego(4) + K_OBS*4 + goal(1) + cbf_h(1) + tangent(2)

# Scenario types — int value pushed as 'scenario_type' side channel. Also used as the
# RL task_id in the recorded dataset. IDs 0-4 are the original scenarios (shared with
# Unity's ScenarioType enum); 5-7 are appended tasks whose Unity behaviours are added
# in a follow-up (the Python plumbing here is forward-compatible with the append plan).
SCENARIO_STANDARD    = 0
SCENARIO_STATIONARY  = 1
SCENARIO_HEADON      = 2
SCENARIO_HIGHSPEED   = 3
SCENARIO_ACCELERATING= 4
SCENARIO_FOLLOW      = 5   # follow a lead vehicle to the same goal; lead randomly stops/accelerates
SCENARIO_RECOVERY    = 6   # bad-spawn: ego injected with lateral/heading error, no obstacle
SCENARIO_BLIND       = 7   # occluded crosser revealed late (building blocks line-of-sight)
SCENARIO_INTERSECTION= 8   # ego on one taxiway branch, one crosser on the other — meet at node
SCENARIO_NAMES = ['standard', 'stationary', 'headon', 'highspeed', 'accelerating',
                  'follow_vehicle', 'recovery', 'blind_corner', 'intersection']

V_DES_HIGH = 14.0   # high-speed scenario [m/s] (~27 knots)

rng = np.random.default_rng(42)


# ── Offline-RL dataset recorder ───────────────────────────────────────────────

class RLDataRecorder:
    """
    Accumulates (obs, action, reward, next_obs, done, task_id) transitions across all
    episodes and writes them to a single compressed .npz for offline RL / TD-MPC2.

    The MPPI+CBF controller is the expert policy; every control step it takes is one
    recorded transition. Rewards are a light shaping signal (progress per step, plus a
    terminal bonus/penalty) — downstream training can ignore or overwrite them since the
    dataset is primarily behaviour-cloning / model-learning fodder.
    """

    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.next_obs, self.terminals, self.task_ids = [], [], []

    def store_step(self, obs, action, reward, next_obs, done, task_id):
        self.obs.append(np.asarray(obs, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.next_obs.append(np.asarray(next_obs, dtype=np.float32).copy())
        self.terminals.append(bool(done))
        self.task_ids.append(int(task_id))

    def __len__(self):
        return len(self.obs)

    def save(self, filename="taxi_expert_data.npz"):
        if not self.obs:
            print("[Recorder] No transitions collected — nothing to save.")
            return
        print(f"[Recorder] Saving {len(self.obs)} transitions to {filename} ...")
        np.savez_compressed(
            filename,
            observations=np.array(self.obs,      dtype=np.float32),
            actions=np.array(self.actions,       dtype=np.float32),
            rewards=np.array(self.rewards,       dtype=np.float32),
            next_observations=np.array(self.next_obs, dtype=np.float32),
            terminals=np.array(self.terminals,   dtype=bool),
            task_ids=np.array(self.task_ids,     dtype=np.int32),
        )
        # Per-task transition counts help spot under-represented tasks in the dataset.
        uniq, counts = np.unique(self.task_ids, return_counts=True)
        summary = ", ".join(
            f"{SCENARIO_NAMES[t] if t < len(SCENARIO_NAMES) else t}={c}"
            for t, c in zip(uniq, counts))
        print(f"[Recorder] Per-task transitions: {summary}")


# ── Observation unpacking ────────────────────────────────────────────────────

def _is_frenet_mode(obs: np.ndarray) -> bool:
    """
    Detect whether Unity is sending Frenet-frame observations.
    The stub values when no network is assigned are tangent=(0,1) exactly.
    A real path tangent will have tangent_z != 1.0 or tangent_x != 0.0.
    """
    tan_x = float(obs[18])
    tan_z = float(obs[19])
    # Stub: (0.0, 1.0).  Real tangent: anything else (path tangent is unit-length).
    return not (abs(tan_x) < 1e-4 and abs(tan_z - 1.0) < 1e-4)


def obs_to_state(obs: np.ndarray, prev_delta: float = 0.0, prev_accel: float = 0.0):
    """
    Unpack the Unity 20-D observation vector.

    Returns
    -------
    s          : np.ndarray shape (6,)
                 Global mode   — [x_fwd, y_lat, theta, v, delta_actual, accel_actual]
                 Frenet mode   — [s_arc, d_cross, theta_e, v, delta_actual, accel_actual]
    obstacles  : list of (rel_xy, obs_v)
                 rel_xy is ego-relative (dx, dy) in whichever frame.
                 CBF uses Euclidean distance so it's frame-independent.
    goal       : float  — remaining distance / arc-length to goal
    frenet_mode: bool   — True when TaxiwayNetwork is active in Unity
    tangent    : np.ndarray (2,) — (tan_x, tan_z) in Unity world space;
                 use to rotate CBF safe-set or path-following cost

    A zero-padded obstacle slot has dy == 999; those are skipped.
    """
    ego0    = float(obs[0])   # x_ego (global) or s (Frenet)
    ego1    = float(obs[1])   # y_ego (global) or d (Frenet)
    ego2    = float(obs[2])   # theta (global) or theta_e (Frenet)
    v       = float(obs[3])
    goal    = float(obs[16])
    tan_x   = float(obs[18])
    tan_z   = float(obs[19])
    frenet  = _is_frenet_mode(obs)

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

    return s, obstacles, goal, frenet, np.array([tan_x, tan_z])


def inject_sensor_noise(obs: np.ndarray, noise_std: float, rng_local) -> np.ndarray:
    """
    Add Gaussian noise to the obstacle observation slots [4..15].
    Ego state [0..3], goal/cbf [16..17], and tangent [18..19] are not corrupted.
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


def mppi(s0, mean, obstacles, goal, frenet_mode=False, tangent=None):
    """
    Sample K_MPPI rollouts with realistic 6D state dynamics.

    Global mode (frenet_mode=False):
      s0 = [x, y, theta_global, v, delta, accel]. Rollout in world frame.
      Lane cost penalises lateral position y; obstacles in world coords.

    Frenet mode (frenet_mode=True, tangent=(tan_x, tan_z)):
      s0 = [s_arc, d, theta_e, v, delta, accel].
      We RECONSTRUCT the global heading from the path tangent and roll out in
      the WORLD frame so that obstacle deltas/velocities (which Unity always
      sends in world coords) stay in one consistent frame as the ego heading.
      The lane-following cost is then recovered by projecting the world-frame
      displacement onto the path tangent (progress) and normal (cross-track).

      tangent points along the path in world space: heading angle is
      atan2(tan_x, tan_z) because the rollout uses cos(theta)->Z, sin(theta)->X.

    Returns (u_nom, new_mean).
    """
    # Widen steering noise and relax lane penalty when a stationary/slow obstacle
    # blocks the lane ahead — gives MPPI enough samples to find a go-around path.
    sig_d_eff = SIG_D
    w_off_eff = W_OFF
    for rel_xy, obs_v in obstacles:
        obs_spd = float(np.hypot(obs_v[0], obs_v[1]))
        if 0 < rel_xy[0] < D_BYPASS and abs(rel_xy[1]) < W_HALF and obs_spd < 2.0:
            sig_d_eff = SIG_D_BYPASS
            w_off_eff = W_OFF_BYPASS
            break

    noise = rng.normal(0, [SIG_A, sig_d_eff], (K_MPPI, H_MPPI, 2))
    na    = mean + noise
    na[:, :, 0] = np.clip(na[:, :, 0], A_MIN, A_MAX)
    na[:, :, 1] = np.clip(na[:, :, 1], -DELTA_LIM, DELTA_LIM)

    cost = np.zeros(K_MPPI)
    st   = np.tile(s0, (K_MPPI, 1)).astype(float)

    if frenet_mode:
        # Reconstruct world heading and roll out from a zeroed world origin.
        tan_x, tan_z = float(tangent[0]), float(tangent[1])
        th_tan = np.arctan2(tan_x, tan_z)          # path heading in world frame
        d0     = s0[1]                             # initial cross-track error
        th_g0  = th_tan - s0[2]                     # theta_e = th_tan - th_global
        st[:, 0] = 0.0                              # world Z displacement from start
        st[:, 1] = 0.0                              # world X displacement from start
        st[:, 2] = th_g0                            # world heading
        s0_fwd = 0.0
        s0_lat = 0.0
    else:
        s0_fwd = s0[0]
        s0_lat = s0[1]

    for k in range(H_MPPI):
        st = _rollout_step(st, na[:, k, 0], na[:, k, 1])
        fwd, lat, th, vv = st[:, 0], st[:, 1], st[:, 2], st[:, 3]

        if frenet_mode:
            # Cross-track error d(t) = d0 + (Tz*ΔX - Tx*ΔZ); matches GetRelativeState sign.
            d_t     = d0 + tan_z * lat - tan_x * fwd
            theta_e = th_tan - th
            cost += W_LAT  * d_t**2
            cost += W_HEAD * theta_e**2
            cost += np.where(np.abs(d_t) > W_HALF, w_off_eff * (np.abs(d_t) - W_HALF)**2, 0.)
        else:
            cost += W_LAT  * lat**2
            cost += W_HEAD * th**2
            cost += np.where(np.abs(lat) > W_HALF, w_off_eff * (np.abs(lat) - W_HALF)**2, 0.)

        cost += W_V    * (vv - V_DES)**2
        cost += W_CTRL * (na[:, k, 0]**2 + 4. * na[:, k, 1]**2)

        # Obstacles — world frame in both modes. rel_xy is ego-relative (world Z, X).
        t_elapsed = (k + 1) * DT
        for rel_xy, obs_v in obstacles:
            obs_z = s0_fwd + rel_xy[0] + obs_v[0] * t_elapsed
            obs_x = s0_lat + rel_xy[1] + obs_v[1] * t_elapsed
            d_obs = np.hypot(fwd - obs_z, lat - obs_x)
            cost += np.where(d_obs < D_INFL, W_OBS * (D_INFL - d_obs)**2, 0.)
            cost += np.where(d_obs < D_SAFE, BIG, 0.)

    # Progress
    if frenet_mode:
        prog = tan_z * st[:, 0] + tan_x * st[:, 1]   # displacement along tangent
        cost += W_PROG * (goal - prog)
    else:
        cost += W_PROG * (goal - (st[:, 0] - s0_fwd))

    w   = np.exp(-(cost - cost.min()) / LAMBDA)
    w  /= w.sum()
    opt = (w[:, None, None] * na).sum(axis=0)

    u_nom    = opt[0].copy()
    new_mean = np.vstack([opt[1:], opt[-1]])
    return u_nom, new_mean


# ── HOCBF-QP ─────────────────────────────────────────────────────────────────

def hocbf_constraint(s, u_nom, rel_xy, obs_v):
    """
    Compute one HOCBF constraint row for a single obstacle.
    Returns (A_row, b_row) for the QP inequality A @ u >= b.

    rel_xy : ego-relative obstacle position (dx_fwd, dy_lat) — works in both frames
             because h = dist² - D² is Euclidean and frame-independent for constraint geometry.
    """
    x, y, th, v  = s
    a_nom  = float(np.clip(u_nom[0], A_MIN, A_MAX))
    d_nom  = float(np.clip(u_nom[1], -DELTA_LIM, DELTA_LIM))
    # dx, dy are relative to ego: obstacle is at (x+rel_xy[0], y+rel_xy[1])
    dx     = -rel_xy[0]   # ego → obstacle: ego minus obstacle = -(obs - ego)
    dy     = -rel_xy[1]
    vx, vy = obs_v
    rel_vx = v * np.cos(th) - vx
    rel_vy = v * np.sin(th) - vy

    h    = dx**2 + dy**2 - D_SAFE**2
    hdot = 2. * (dx * rel_vx + dy * rel_vy)

    tand    = np.tan(d_nom)
    sec2d   = 1. + tand**2
    thdot_n = v / L * tand

    hh_kin = 2. * (rel_vx**2 + rel_vy**2)
    hh_th  = 2.*dx*(-v*np.sin(th)*thdot_n) + 2.*dy*(v*np.cos(th)*thdot_n)
    hh_a   = (2.*dx*np.cos(th) + 2.*dy*np.sin(th)) * a_nom
    hh_nom = hh_kin + hh_th + hh_a

    dHH_da  = 2.*dx*np.cos(th) + 2.*dy*np.sin(th)
    dthd_dd = v / L * sec2d
    dHH_dd  = 2.*dx*(-v*np.sin(th))*dthd_dd + 2.*dy*(v*np.cos(th))*dthd_dd

    rhs = (-(ALPHA1 + ALPHA2)*hdot - ALPHA1*ALPHA2*h
           - hh_nom + dHH_da*a_nom + dHH_dd*d_nom)

    return np.array([dHH_da, dHH_dd]), rhs


def cbf_qp(s, u_nom, obstacles, lat=0.0):
    """
    Solve the safety QP with one constraint row per obstacle:
        min  (u - u_nom)^T W (u - u_nom)
        s.t. A_cbf[i] @ u >= b_cbf[i]  for each obstacle i
             A_MIN <= a <= A_MAX
             |delta| <= DELTA_LIM

    lat: current lateral offset [m] — used to detect committed go-around.
    Returns (u_cmd, cbf_engaged).
    """
    a_nom = float(np.clip(u_nom[0], A_MIN, A_MAX))
    d_nom = float(np.clip(u_nom[1], -DELTA_LIM, DELTA_LIM))
    u_n   = np.array([a_nom, d_nom])

    # During a go-around (large lateral offset), strongly prefer steering over braking
    alpha_w = ALPHA_W_GOAROUND if abs(lat) > LAT_GOAROUND else ALPHA_W
    W = np.diag([1., alpha_w])
    c = W @ u_n

    # Box constraints: a >= A_MIN, a <= A_MAX, delta >= -DELTA_LIM, delta <= DELTA_LIM
    C_box = np.array([[ 1., 0.], [-1., 0.], [0.,  1.], [0., -1.]]).T   # (2, 4)
    b_box = np.array([A_MIN, -A_MAX, -DELTA_LIM, -DELTA_LIM])

    th   = float(s[2])
    ehat = np.array([np.cos(th), np.sin(th)])   # ego heading unit vector

    cbf_rows = []  # obstacles is list of (rel_xy, obs_v)
    cbf_rhs  = []
    for obs_xy, obs_v in obstacles:
        # Velocity-based geometry gate (see CBF_MOVER_MIN / CBF_COS_GATE note).
        # Static blockers always engage; movers engage only when their velocity is
        # aligned with the heading axis (head-on/along-track), not for pure crossers.
        v_obs = np.asarray(obs_v, dtype=float)
        speed = float(np.hypot(v_obs[0], v_obs[1]))
        if speed >= CBF_MOVER_MIN:
            cos_align = abs(float(v_obs @ ehat)) / speed
            if cos_align < CBF_COS_GATE:
                continue   # perpendicular crosser — braking is impotent, skip
        row, rhs = hocbf_constraint(s, u_nom, obs_xy, obs_v)
        cbf_rows.append(row)
        cbf_rhs.append(rhs)

    if cbf_rows:
        C_cbf = np.array(cbf_rows).T          # (2, n_active)
        b_cbf = np.array(cbf_rhs)
        C = np.hstack([C_cbf, C_box])          # (2, n_active + 4)
        b = np.concatenate([b_cbf, b_box])
    else:
        C = C_box
        b = b_box

    try:
        u_star  = quadprog.solve_qp(W, c, C, b, 0)[0]
        engaged = bool(np.linalg.norm(u_star - u_n) > 1e-3)
        return u_star, engaged
    except Exception:
        return np.array([A_MIN, d_nom]), True


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
# PERPENDICULAR crossers (standard/highspeed) a faster mover spends LESS time in
# the conflict zone (t_window ~ vehicle_size / speed), so speed scaling makes
# crossings EASIER, not harder — confirmed empirically. Crossers stay slow so they
# linger in the conflict zone; difficulty is raised via Lever 2 (compound conflicts)
# instead. Stationary ignores speed entirely (parked).
SPEED_CROSS_BASE   = 5.0    # perpendicular crosser speed [m/s] — kept low on purpose
SPEED_HEADON_MIN   = 5.0    # head-on closing speed at difficulty 0 [m/s]
SPEED_HEADON_MAX   = 12.0   # head-on closing speed at difficulty 1 [m/s]

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


def make_scenarios(n_episodes, base_seed=BASE_SEED, min_difficulty=0.0, max_difficulty=1.0,
                   scenario_type=SCENARIO_STANDARD):
    """
    Build a reproducible list of per-episode scenario parameter dicts.

    Each dict is pushed to Unity via EnvironmentParametersChannel before env.reset().
    Keys:
      incursion_dt    — Δt offset for the primary (agent[0]) incursion [s]
      ambulance_speed — crossing speed for agent[0] [m/s]
      difficulty      — [0, 1] curriculum difficulty sent to ScenarioManager

    min_difficulty / max_difficulty clamp the ramp so you can test a specific
    difficulty band. E.g. min_difficulty=0.85 forces 3-agent Erratic scenarios.
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
        # Scenario type: forced when scenario_type >= 0, else drawn per-episode
        # across all five types ("mixed" mode, scenario_type = -1).
        # Mixed mode draws from the active scenario list (stationary excluded by user).
        # Tasks 5-7 (follow/recovery/blind) are included so the offline-RL dataset covers
        # all behaviours; drop them from this list if you only want the crossing-conflict
        # scenarios for a CBF vs no-CBF collision-rate comparison.
        _active_types = [SCENARIO_STANDARD, SCENARIO_HEADON,
                         SCENARIO_HIGHSPEED, SCENARIO_ACCELERATING,
                         SCENARIO_FOLLOW, SCENARIO_RECOVERY, SCENARIO_BLIND]
        stype = float(scenario_type) if scenario_type >= 0 else float(r.choice(_active_types))
        desired_spd = V_DES_HIGH if stype == SCENARIO_HIGHSPEED else -1.0
        # Lever 1: only head-on closing speed ramps with difficulty; perpendicular
        # crossers stay slow (so they linger in the conflict zone). Stationary mode
        # ignores this Unity-side; the MPPI bypass keys on the observed velocity, so
        # a parked obstacle stays parked regardless.
        if stype == SCENARIO_HEADON:
            spd_base = SPEED_HEADON_MIN + (SPEED_HEADON_MAX - SPEED_HEADON_MIN) * difficulty
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
        })
    return scenarios


# ── Main control loop ─────────────────────────────────────────────────────────

def run(unity_exec_path=None, port=5004, run_sysid=True, n_episodes=20,
        min_difficulty=0.0, max_difficulty=1.0,
        noise_std=0.0, scenario_type=SCENARIO_STANDARD, no_cbf=False,
        detect_range=float("inf"), dataset_path=None):
    global DETECTION_RANGE
    DETECTION_RANGE = detect_range
    print(f"[Controller] Connecting to Unity on port {port} ...")
    if detect_range < float("inf"):
        print(f"[Controller] Detection range : {detect_range:.1f} m  (obstacles beyond masked)")
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
                                              scenario_type=scenario_type)
    episode_stats = []
    recorder      = RLDataRecorder()   # accumulates (s,a,r,s') transitions for offline RL

    for ep in range(n_episodes):
        mean   = np.zeros((H_MPPI, 2))
        sc     = scenarios[ep]
        task_id = int(sc["scenario_type"])
        ep_log = {"min_h": np.inf, "min_dist": np.inf,
                  "collided": False, "reached": False, "steps": 0,
                  "incursion_dt": sc["incursion_dt"],
                  "difficulty": sc["difficulty"],
                  "scenario": SCENARIO_NAMES[task_id]}
        # RL transition bookkeeping: hold the previous (obs, action) until the next
        # obs arrives, so we can emit a complete (obs, action, reward, next_obs) tuple.
        prev_obs = None
        prev_act = None

        for key, val in sc.items():
            env_params.set_float_parameter(key, val)

        sname = SCENARIO_NAMES[int(sc["scenario_type"])]
        print(f"\n[Ep {ep+1:3d}] [{sname}] Δt={sc['incursion_dt']:+.2f}s  "
              f"v_amb={sc['ambulance_speed']:.2f} m/s  "
              f"diff={sc['difficulty']:.2f}  "
              f"noise={noise_std:.3f}  "
              f"dir={'L' if sc['cross_dir_sign'] < 0 else 'R'}  "
              f"{'[NO-CBF]' if no_cbf else '[CBF]'}")

        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        ep_steps     = 0
        delta_actual = 0.0   # tracked Python-side to feed into MPPI state
        accel_actual = 0.0
        frenet_mode  = False
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
                    recorder.store_step(prev_obs, prev_act, term_reward,
                                        final_obs, True, task_id)
                episode_done = True
                break

            if len(decision_steps) == 0:
                env.step()
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                continue

            obs   = decision_steps.obs[0][0]          # shape (OBS_SIZE,)
            ep_steps += 1

            obs_n = inject_sensor_noise(obs, noise_std, rng)
            s, obstacles, goal, frenet_mode, tangent = obs_to_state(obs_n, delta_actual, accel_actual)

            # Emit the previous transition now that its next_obs (obs_n) is available.
            # Reward: light progress shaping (forward speed); terminal bonus is added at
            # episode end. Downstream RL can ignore/overwrite this — it's mainly BC data.
            if prev_obs is not None:
                step_reward = 0.05 * float(s[3])   # s[3] = current forward speed
                recorder.store_step(prev_obs, prev_act, step_reward,
                                    obs_n, False, task_id)

            if ep_steps % 20 == 0:
                mode_str = f"[Frenet tan=({tangent[0]:.2f},{tangent[1]:.2f})]" if frenet_mode else "[global]"
                print(f"[DEBUG] {mode_str} fwd={s[0]:.1f} lat={s[1]:.2f} "
                      f"th={s[2]:.3f} v={s[3]:.2f} δ={s[4]:.3f} a_act={s[5]:.2f}  "
                      f"{len(obstacles)} obs  goal={goal:.1f}")

            u_nom, mean = mppi(s, mean, obstacles, goal, frenet_mode, tangent)

            if no_cbf:
                u_cmd      = u_nom
                cbf_engaged = False
            else:
                # CBF must run in the WORLD frame: obstacle deltas/velocities are world-frame,
                # so the ego heading fed to the CBF must be world-frame too. In Frenet mode the
                # observed heading is theta_e (path-relative), so reconstruct global heading.
                if frenet_mode:
                    th_tan   = np.arctan2(tangent[0], tangent[1])
                    th_world = th_tan - s[2]
                    s_cbf    = np.array([0.0, 0.0, th_world, s[3]])
                else:
                    s_cbf    = s[:4]
                u_cmd, cbf_engaged = cbf_qp(s_cbf, u_nom, obstacles, lat=float(s[1]))
            a_cmd     = float(np.clip(u_cmd[0], A_MIN, A_MAX))
            delta_cmd = float(np.clip(u_cmd[1], -DELTA_LIM, DELTA_LIM))

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

            # Log nearest obstacle distance
            h_val = float(obs[17])
            dist  = float(np.sqrt(max(0., h_val + D_SAFE**2)))
            ep_log["min_h"]    = min(ep_log["min_h"], h_val)
            ep_log["min_dist"] = min(ep_log["min_dist"], dist)
            ep_log["steps"]    = ep_steps

            if cbf_engaged:
                print(f"  [CBF] t={ep_steps*DT:.1f}s  h={h_val:.1f}  dist={dist:.2f}m  "
                      f"a_nom={u_nom[0]:.2f}→{a_cmd:.2f}  "
                      f"d_nom={u_nom[1]:.3f}→{delta_cmd:.3f}  "
                      f"n_obs={len(obstacles)}")

            action = ActionTuple(
                continuous=np.array([[a_cmd, delta_cmd]], dtype=np.float32)
            )
            # Stash this (obs, action) so the next iteration can close the transition.
            prev_obs = obs_n.copy()
            prev_act = np.array([a_cmd, delta_cmd], dtype=np.float32)

            env.set_actions(behavior_name, action)
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
        episode_stats.append(ep_log)

        verdict = "COLLISION" if ep_log["collided"] else "safe"
        print(f"[Ep {ep+1:3d}] Δt={ep_log['incursion_dt']:+.2f}s  "
              f"diff={ep_log['difficulty']:.2f}  "
              f"steps={ep_log['steps']:4d}  "
              f"min_dist={ep_log['min_dist']:5.2f}m  "
              f"min_h={ep_log['min_h']:8.2f}  → {verdict}")

    env.close()

    print("\n=== Summary ===")
    n   = len(episode_stats)
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

    # ── Persist the offline-RL dataset ────────────────────────────────────────
    if dataset_path:
        recorder.save(dataset_path)

    return episode_stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exec",           default=None)
    p.add_argument("--port",           default=5004,  type=int)
    p.add_argument("--sysid",          default=True,  type=lambda x: x.lower() == "true")
    p.add_argument("--episodes",       default=20,    type=int)
    p.add_argument("--min-difficulty", default=0.0,   type=float)
    p.add_argument("--max-difficulty", default=1.0,   type=float)
    p.add_argument("--noise-std",      default=0.0,   type=float,
                   help="Std-dev of Gaussian noise injected into obstacle obs [m]. 0=off.")
    p.add_argument("--scenario",       default="standard",
                   choices=SCENARIO_NAMES + ["mixed"],
                   help="Force a specific scenario type for all episodes, or "
                        "'mixed' to randomise across all five types per episode.")
    p.add_argument("--no-cbf",         action="store_true",
                   help="Disable the HOCBF-QP safety filter (MPPI only). "
                        "Use for ablation baseline.")
    p.add_argument("--detect-range",   default=float("inf"), type=float,
                   help="Euclidean detection range [m]. Obstacles beyond this distance "
                        "are masked from MPPI and CBF, simulating finite sensor range. "
                        "Default=inf (oracle). Try 25 for a tight reaction window.")
    p.add_argument("--dataset",        default=None,
                   help="Path to save the recorded offline-RL dataset (.npz). "
                        "Omit to skip saving (e.g. for quick ablation runs).")
    args = p.parse_args()

    sc_int = -1 if args.scenario == "mixed" else SCENARIO_NAMES.index(args.scenario)
    run(unity_exec_path=args.exec if args.exec != "None" else None,
        port=args.port,
        run_sysid=args.sysid,
        n_episodes=args.episodes,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        noise_std=args.noise_std,
        scenario_type=sc_int,
        no_cbf=args.no_cbf,
        detect_range=args.detect_range,
        dataset_path=args.dataset)
