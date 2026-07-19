"""
taxi_controller_mpc.py
======================
External Python controller for the Unity taxiing environment — MPC variant.

This is a drop-in alternative to `taxi_controller.py`. Instead of the sampling-based
MPPI planner, it drives the aircraft with a **nonlinear Model Predictive Controller
(NMPC)** that solves a constrained optimal-control problem every decision step with
CasADi + IPOPT:

    minimize   Σ_k  ℓ_stage(x_k, u_k)   +   ℓ_terminal(x_N)
    subject to x_0     = s0                         (current measured state)
               x_{k+1} = f(x_k, u_k)                (realistic bicycle dynamics)
               A_MIN <= a_k     <= A_MAX            (control box constraints)
               -DELTA_LIM <= delta_k <= DELTA_LIM

The prediction model f() is a SMOOTH kinematic bicycle matching the physically
important parts of the MPPI rollout / `TaxiAgent.ApplyBicycleDynamics` — first-order
acceleration lag, aerodynamic/rolling drag, speed-dependent steering roll-off — but
the hard clamps of that rollout (command saturation, the steering-rate limit, v≥0)
are expressed as the NLP's box bounds / linear constraints instead of in-model
min/max, so the model stays C¹ and IPOPT converges in a handful of iterations.
Closed-loop re-solves every DT correct any small mismatch against Unity's exact
(clamped) dynamics. State is [x, y, theta, v, accel]; the rate-limited delta_actual
of the MPPI rollout is not a model state here.

Everything else — observation contract (OBS_SIZE=20), action contract ([a, delta]),
the scenario sweep, sys-id probe, and per-episode trajectory logging — is shared with
`taxi_controller.py` via direct import, so the two controllers stay in lock-step.

Observation / action contract: see taxi_controller.py (unchanged).

Requirements (beyond taxi_controller's): `pip install casadi`.

Run (same as the MPPI controller, e.g.):
    python3 taxi_controller_mpc.py --scenario headon --episodes 20 --save-traj out_mpc
"""

import argparse
import time

import numpy as np
import casadi as ca


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.exception import UnityCommunicatorStoppedException
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

# ── Reuse the shared machinery + parameters from the MPPI controller ──────────
# Importing keeps the dynamics constants, observation unpacking, scenario sweep,
# sys-id probe and trajectory logging identical across both controllers.
import taxi_controller as tc
from taxi_controller import (
    DT, L, V_DES, GOAL_SLOWDOWN_DIST, GOAL_MIN_SPEED,
    A_MIN, A_MAX, DELTA_LIM,
    DRAG_COEFF, ACCEL_TAU, MAX_STEER_RATE, STEER_ROLLOFF_SPD, STEER_ROLLOFF_MIN,
    D_SAFE, D_INFL, K_OBS, OBS_SIZE, DUD_DIST,
    D_SAFE_STATIC, D_INFL_STATIC,
    V_TARGET, D_SAFE_OCC, D_INFL_OCC, W_OCC,
    W_SIGHT, A_BRAKE_SIGHT, V_SIGHT_FLOOR,
    SCENARIO_STANDARD, SCENARIO_NAMES,
    obs_to_state, inject_sensor_noise, make_scenarios,
    identify_bicycle_model, _save_trajectory,
)

# ── MPC-specific tuning ───────────────────────────────────────────────────────
# Horizon: MPPI uses H=60 (6 s) because sampling is cheap. IPOPT solves a real NLP
# each step, so a shorter horizon keeps the per-step solve well under the DT budget
# while still seeing far enough ahead for taxi-speed avoidance. 25 steps = 2.5 s.
N_MPC   = 40

# Goal capture zone. The imported speed taper floors the target at GOAL_MIN_SPEED so
# the ego never stalls mid-course — but right at the goal that floor means it can't
# brake to a halt, so it overshoots the small arrival radius and orbits back around.
# Inside GOAL_STOP_DIST we multiply the target down to 0 at the goal so the MPC brakes
# to a stop on it. Continuous with the outer taper (factor = 1 beyond this distance).
GOAL_STOP_DIST = 5.0    # [m] remaining distance within which the speed target ramps to 0

# Stage-cost weights. Everything is kept well-scaled (O(1) gradients) so the NLP
# is well-conditioned. The driving objective uses BOUNDED-gradient terms — heading
# alignment, speed tracking, and a LINEAR distance-to-goal pull (not an unbounded
# ‖p − p_goal‖², whose gradient blows up far from the goal and dwarfs the obstacle
# terms) — so that near an obstacle the keep-out reliably wins and the ego brakes
# or steers rather than buying its way through.
W_GOAL_RUN  = 0.05    # gentle running pull toward the goal (LINEAR in distance)
W_GOAL_TERM = 2.0     # terminal pull on the FINAL state (LINEAR in distance)
W_HEAD      = 6.0     # heading alignment toward the goal bearing  (1 − cos(θ − ψ)); raised to
                      # commit to the goal bearing and damp the lateral S-oscillation
W_V         = 2.0     # speed tracking toward the (goal-tapered) desired speed
R_ACT       = np.array([0.05, 0.20])   # control effort  ‖u‖²_R
R_DACT      = np.array([0.10, 0.7])   # control rate  ‖Δu‖²_RΔ; steering-rate weight raised
                                       # (0.40→1.0) to smooth steering reversals (anti-oscillation)

# Obstacle avoidance. Predicted with a constant-velocity ray (same as MPPI's
# deterministic branch). The keep-out is a HARD constraint  d ≥ D_SAFE, softened
# with a per-(stage,obstacle) slack so the NLP is always feasible; the slack is
# hit with a well-scaled exact (linear + quadratic) penalty that dominates the
# bounded goal pull, so no rollout buys its way through the keep-out. A soft
# influence ring gives smooth early deflection before the hard radius is reached.
W_OBS      = 8.0      # soft influence ring:  W_OBS · max(0, D_INFL − d)²
RHO_SLACK  = 7.0     # linear exact-penalty weight on a keep-out violation [per m²]
RHO_SLACK2 = 3.0      # quadratic penalty on the violation (curves the exact penalty)

# Static-obstacle (LiDAR costmap) avoidance. Enabled with --lidar-costmap. Each
# control step the K_STATIC nearest occupied (OCC) cell centres to the ego are fed
# as fixed keep-out points into dedicated NLP slots, reusing the slack machinery
# above with the static radii D_SAFE_STATIC / D_INFL_STATIC (imported from the MPPI
# controller). K_STATIC trades coverage of nearby walls against solve time.
K_STATIC       = 12       # number of nearest static OCC points constrained per step
W_STATIC_RING  = 15.0      # soft influence ring weight for static surfaces
STATIC_QUERY_R = 35.0     # only consider OCC cells within this radius of the ego [m]

# Dead-ahead symmetry breaking (warm-start only — constraints are untouched, so
# full D_SAFE is preserved). A frontal obstacle within SYM_AHEAD_RANGE whose lateral
# offset from the ego→goal line is under SYM_LAT_THRESH is treated as "on the line";
# the initial guess is then curved to one side with SYM_BIAS steering for the first
# SYM_BIAS_FRAC of the horizon so IPOPT descends into a go-around basin.
SYM_LAT_THRESH  = 5.0    # |lateral offset| below which an obstacle counts as dead-ahead [m]
SYM_AHEAD_RANGE = 22.0   # only bias for obstacles this close ahead [m]
SYM_BIAS        = 0.35   # steering bias applied in the seeded guess [rad]
SYM_BIAS_FRAC   = 0.5    # fraction of the horizon over which the bias is applied

# ── Occlusion-aware MPC (Firoozi et al., forward reachable sets) ──────────────
# Undetected agents may emerge from occluded regions. Following the paper's
# alternative "direct circle-sampling" formulation (the drop-in fit for a
# monolithic IPOPT NLP, versus the bilevel capsule-projection method), each
# occlusion BOUNDARY point (a blind-spot cell from the LiDAR costmap, see
# LidarCostmap.occlusion_points) seeds an EXPANDING keep-out circle: a worst-case
# hidden agent starts on the boundary and advances at up to V_TARGET, so the
# forbidden radius grows along the prediction horizon as V_TARGET · t_k. The
# ego therefore gives occluded corners a wider berth the further ahead it plans.
# Enabled only with --lidar-costmap (needs the map); off ⇒ k_occ=0, zero cost.
#
# The occlusion model + constants (V_TARGET, D_SAFE_OCC, D_INFL_OCC, W_OCC, and the
# sightline cap W_SIGHT / A_BRAKE_SIGHT / V_SIGHT_FLOOR) are imported from
# taxi_controller so the MPC and MPPI controllers stay in lock-step. Only the two
# NLP-specific knobs below are local: K_OCC (number of solver slots) and OCC_QUERY_R.
K_OCC       = 10        # nearest occlusion-boundary points constrained per step (0 ⇒ off)
OCC_QUERY_R = 60.0     # only consider occlusion boundaries within this radius of the ego [m]


class TaxiMPC:
    """Nonlinear MPC for the taxiing aircraft, built once and re-solved each step.

    The NLP is assembled symbolically in __init__ with the current state, goal,
    desired speed, last applied control and the (up to K_OBS) obstacle predictions
    as *parameters*, so each control step only updates numbers and re-solves —
    the standard receding-horizon MPC pattern. Warm-started from the previous
    solution (shifted one step) for fast convergence.
    """

    # State drops the rate-limited `delta_actual` used by the MPPI rollout: the
    # steering rate limit is enforced instead as a LINEAR constraint on Δdelta_cmd,
    # which keeps the model smooth (IPOPT-friendly) rather than clamped.
    NX = 5   # state  [x, y, theta, v, accel]
    NU = 2   # control [a_cmd, delta_cmd]

    def __init__(self, N=N_MPC, k_obs=K_OBS, d_infl=D_INFL, d_safe=D_SAFE,
                 k_static=0, d_infl_static=D_INFL_STATIC, d_safe_static=D_SAFE_STATIC,
                 k_occ=0, d_infl_occ=D_INFL_OCC, d_safe_occ=D_SAFE_OCC, v_target=V_TARGET,
                 verbose=False):
        self.N     = int(N)
        self.k_obs = int(k_obs)
        self.d_infl = float(d_infl)
        self.d_safe = float(d_safe)
        self.k_static = int(k_static)          # 0 ⇒ no static (LiDAR) keep-out
        self.d_infl_static = float(d_infl_static)
        self.d_safe_static = float(d_safe_static)
        self.k_occ = int(k_occ)                # 0 ⇒ no occlusion-aware keep-out
        self.d_infl_occ = float(d_infl_occ)
        self.d_safe_occ = float(d_safe_occ)
        self.v_target   = float(v_target)

        nx, nu, Nh = self.NX, self.NU, self.N

        # ── Symbolic one-step dynamics f(x, u) ────────────────────────────────
        st = ca.MX.sym("st", nx)
        u  = ca.MX.sym("u",  nu)
        self.f = ca.Function("f", [st, u], [self._dynamics(st, u)])

        # ── Decision variables: states X (N+1), controls U (N) ───────────────
        # No keep-out slack variables: the hard keep-outs (d ≥ D_SAFE) are folded
        # into pure SMOOTH cost penalties below instead of inequality constraints +
        # slacks. This is algebraically identical to the old slack formulation at the
        # optimum (an optimal slack equals the keep-out violation), but removes the
        # (k_obs+k_static+k_occ)·Nh inequality constraints and slack vars — the source
        # of the feasibility-restoration churn that drove Maximum_Iterations_Exceeded
        # once the LiDAR/occlusion keep-outs switched on. The problem is now far
        # smaller and better conditioned, so IPOPT converges inside the iter budget.
        X = [ca.MX.sym(f"x_{k}", nx) for k in range(Nh + 1)]
        U = [ca.MX.sym(f"u_{k}", nu) for k in range(Nh)]

        # ── Parameters (updated every solve) ──────────────────────────────────
        P_s0    = ca.MX.sym("s0",    nx)          # current measured state
        P_goal  = ca.MX.sym("goal",  2)           # goal world position (x_fwd, y_lat)
        P_vdes  = ca.MX.sym("vdes",  1)           # goal-tapered desired speed
        P_uprev = ca.MX.sym("uprev", nu)          # last applied control (Δu_0 + rate limit)
        P_opos  = [ca.MX.sym(f"op_{i}", 2) for i in range(self.k_obs)]  # abs obstacle pos now
        P_ovel  = [ca.MX.sym(f"ov_{i}", 2) for i in range(self.k_obs)]  # obstacle velocity
        P_spos  = [ca.MX.sym(f"sp_{i}", 2) for i in range(self.k_static)]  # static OCC points
        P_occ   = [ca.MX.sym(f"occ_{i}", 2) for i in range(self.k_occ)]    # occlusion boundary points

        R_act_dm  = ca.DM(R_ACT)
        R_dact_dm = ca.DM(R_DACT)
        d_safe2      = self.d_safe ** 2
        d_safe_stat2 = self.d_safe_static ** 2
        max_dstep = MAX_STEER_RATE * DT           # per-step steering-rate cap [rad]

        g_eq   = [X[0] - P_s0]   # equalities: initial state + multiple-shooting defects
        g_rate = []              # inequalities: |Δdelta_cmd| ≤ max_dstep  (linear)
        cost   = 0
        v_safe_list = []         # per-stage sightline speed cap, for debug printing

        for k in range(Nh):
            xk, uk = X[k], U[k]
            g_eq.append(X[k + 1] - self.f(xk, uk))

            xn = X[k + 1]
            px, py, th, v = xn[0], xn[1], xn[2], xn[3]

            # Gentle goal-position pull (LINEAR distance → bounded gradient) +
            # heading alignment toward the goal bearing.
            d_goal = ca.sqrt((px - P_goal[0]) ** 2 + (py - P_goal[1]) ** 2 + 1.0)
            cost += W_GOAL_RUN * d_goal
            psi   = ca.atan2(P_goal[1] - py, P_goal[0] - px)
            cost += W_HEAD * (1 - ca.cos(th - psi))

            # Speed tracking toward the (goal-tapered) desired speed.
            cost += W_V * (v - P_vdes) ** 2

            # Control effort + rate (smoothness). Δu_0 uses the last applied control.
            u_km1 = P_uprev if k == 0 else U[k - 1]
            du    = uk - u_km1
            cost += ca.dot(R_act_dm  * uk, uk)
            cost += ca.dot(R_dact_dm * du, du)

            # Steering-rate limit as a linear inequality: −Δ ≤ delta_k − delta_{k−1} ≤ Δ.
            g_rate.append(uk[1] - u_km1[1])

            # Obstacle terms — constant-velocity prediction to time (k+1)·DT.
            t_k = (k + 1) * DT
            for i in range(self.k_obs):
                op   = P_opos[i] + P_ovel[i] * t_k
                dx_o = px - op[0]
                dy_o = py - op[1]
                d2   = dx_o * dx_o + dy_o * dy_o
                d    = ca.sqrt(d2 + 1e-6)
                # Soft influence ring (smooth early deflection).
                cost += W_OBS * ca.fmax(0.0, self.d_infl - d) ** 2
                # Keep-out d ≥ D_SAFE as a smooth exact penalty on the violation
                # (d_safe² − d²)₊ — identical to the old slack cost at the optimum.
                viol = ca.fmax(0.0, d_safe2 - d2)
                cost += RHO_SLACK * viol + RHO_SLACK2 * viol * viol

            # Static-obstacle terms (LiDAR OCC points) — fixed, no velocity.
            for i in range(self.k_static):
                sp   = P_spos[i]
                dx_s = px - sp[0]
                dy_s = py - sp[1]
                d2s  = dx_s * dx_s + dy_s * dy_s
                ds   = ca.sqrt(d2s + 1e-6)
                cost += W_STATIC_RING * ca.fmax(0.0, self.d_infl_static - ds) ** 2
                viols = ca.fmax(0.0, d_safe_stat2 - d2s)
                cost += RHO_SLACK * viols + RHO_SLACK2 * viols * viols

            # Occlusion forward-reachable-set terms (Firoozi et al.). A worst-case
            # hidden agent starts on the occlusion boundary and advances at up to
            # v_target, so the keep-out circle EXPANDS along the horizon by
            # v_target · t_k. The influence ring expands identically. (The paper's
            # d_target = v_target/Δt is a typo; distance = speed · time is used.)
            r_grow = self.v_target * t_k
            d_vis  = ca.MX(1e6)          # distance to the NEAREST occlusion boundary this stage
            for i in range(self.k_occ):
                oc   = P_occ[i]
                dx_c = px - oc[0]
                dy_c = py - oc[1]
                d2c  = dx_c * dx_c + dy_c * dy_c
                dc   = ca.sqrt(d2c + 1e-6)
                d_vis = ca.fmin(d_vis, dc)
                r_infl_occ = self.d_infl_occ + r_grow
                r_keep_occ = self.d_safe_occ + r_grow
                # Soft influence ring (smooth early deflection around the corner).
                cost += W_OCC * ca.fmax(0.0, r_infl_occ - dc) ** 2
                # Expanding keep-out d ≥ r_keep_occ as a smooth exact penalty.
                violc = ca.fmax(0.0, r_keep_occ * r_keep_occ - d2c)
                cost += RHO_SLACK * violc + RHO_SLACK2 * violc * violc

            # Sightline (RSS) speed cap: slow so the ego can brake to a stop before the
            # nearest occlusion boundary. Widening the arc alone would let it round a
            # blind corner at cruise speed; this makes it slow into the corner. Padded
            # (far-parked) occlusion slots keep d_vis huge ⇒ v_safe huge ⇒ no cap when
            # no real boundary is near.
            if self.k_occ:
                v_safe = ca.fmax(ca.sqrt(2.0 * A_BRAKE_SIGHT * d_vis + 1e-6), V_SIGHT_FLOOR)
                v_safe_list.append(v_safe)
                cost += W_SIGHT * ca.fmax(0.0, v - v_safe) ** 2

        # Terminal goal pull (LINEAR distance) — anchors the END of the horizon at
        # the goal so a rollout may detour around an obstacle yet still be rewarded
        # for finishing close.
        xn = X[Nh]
        cost += W_GOAL_TERM * ca.sqrt((xn[0] - P_goal[0]) ** 2 + (xn[1] - P_goal[1]) ** 2 + 1.0)

        # ── Assemble the NLP ──────────────────────────────────────────────────
        # Only two constraint blocks remain: the dynamics equalities and the linear
        # steering-rate inequalities. All keep-outs are now cost penalties.
        w = ca.vertcat(*X, *U)
        p = ca.vertcat(P_s0, P_goal, P_vdes, P_uprev, *P_opos, *P_ovel, *P_spos, *P_occ)
        g = ca.vertcat(*g_eq, *g_rate)
        nlp = {"x": w, "p": p, "f": cost, "g": g}

        # Debug helper: evaluate the per-stage sightline speed cap numerically
        # after solving (v_safe is symbolic here and can't be printed directly).
        if v_safe_list:
            self.v_safe_fn = ca.Function("v_safe_fn", [w, p], [ca.vertcat(*v_safe_list)])
        else:
            self.v_safe_fn = None

        opts = {
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "max_iter": 60,
                "tol": 1e-4,
                # Stop early at a good-enough point instead of grinding to max_iter.
                "acceptable_tol": 5e-3,
                "acceptable_iter": 3,
                # L-BFGS Hessian: far cheaper per iteration and better conditioned
                # near the penalty kinks than the exact Hessian on a problem this size.
                "hessian_approximation": "limited-memory",
                "warm_start_init_point": "yes",
                "mu_strategy": "adaptive",
                "sb": "yes",
            },
        }
        self.solver = ca.nlpsol("taxi_mpc", "ipopt", nlp, opts)

        # ── Sizes / bounds ────────────────────────────────────────────────────
        self.n_x  = nx * (Nh + 1)
        self.n_u  = nu * Nh
        n_eq    = nx * (Nh + 1)              # initial + Nh dynamics defects
        n_rate  = Nh                         # steering-rate inequalities

        # States: v ≥ 0 (bound), rest free; controls box-constrained.
        lbx, ubx = [], []
        for _ in range(Nh + 1):
            lbx += [-ca.inf, -ca.inf, -ca.inf, 0.0, -ca.inf]   # v ≥ 0
            ubx += [ ca.inf,  ca.inf,  ca.inf, ca.inf, ca.inf]
        for _ in range(Nh):
            lbx += [A_MIN, -DELTA_LIM]
            ubx += [A_MAX,  DELTA_LIM]
        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)
        # Equalities = 0; steering-rate in [−max_dstep, +max_dstep].
        self.lbg = np.concatenate([np.zeros(n_eq),
                                   np.full(n_rate, -max_dstep)])
        self.ubg = np.concatenate([np.zeros(n_eq),
                                   np.full(n_rate,  max_dstep)])

        # Warm-start caches (previous optimal trajectory + duals).
        self._Xopt = None    # (Nh+1, nx)
        self._Uopt = None    # (Nh, nu)
        self._lam_x0 = None
        self._lam_g0 = None
        self.verbose = verbose

    # ── Smooth symbolic dynamics used for MPC prediction ──────────────────────
    # A smooth kinematic bicycle: first-order acceleration lag + drag + speed-
    # dependent steering roll-off. The hard clamps of the MPPI rollout (command
    # saturation, steering-rate limit, v≥0) are handled by the NLP's box bounds /
    # linear constraints instead of in-model max/min, so the model stays C¹ and
    # IPOPT converges in a handful of iterations. Closed-loop re-solves every DT
    # correct any small mismatch against Unity's exact ApplyBicycleDynamics.
    def _dynamics(self, st, u):
        x, y, th, v, accel = st[0], st[1], st[2], st[3], st[4]
        a_cmd, delta_cmd   = u[0], u[1]

        # First-order acceleration lag (a_cmd is box-bounded to [A_MIN, A_MAX]).
        accel_new = accel + (a_cmd - accel) * (DT / ACCEL_TAU)

        # Drag + speed integration (v ≥ 0 enforced by the state's lower bound).
        drag  = DRAG_COEFF * v
        v_new = v + (accel_new - drag) * DT

        # Speed-dependent steering authority (smooth roll-off; delta_cmd box-bounded).
        speed_frac = v / max(STEER_ROLLOFF_SPD, 1e-3)
        authority  = 1.0 - speed_frac * (1.0 - STEER_ROLLOFF_MIN)
        delta_eff  = delta_cmd * authority

        # Bicycle geometry.
        dtheta = v_new / L * ca.tan(delta_eff) * DT
        x_new  = x + v_new * ca.cos(th) * DT
        y_new  = y + v_new * ca.sin(th) * DT
        th_new = th + dtheta

        return ca.vertcat(x_new, y_new, th_new, v_new, accel_new)

    # ── Build the numeric parameter vector for one solve ──────────────────────
    def _pack_params(self, s0, goal_xy, v_des, u_prev, obstacles,
                     static_pts=None, occ_pts=None):
        nx = self.NX
        p = list(s0[:nx]) + [goal_xy[0], goal_xy[1], v_des, u_prev[0], u_prev[1]]

        opos, ovel = [], []
        for i in range(self.k_obs):
            if i < len(obstacles):
                rel_xy, obs_v = obstacles[i]
                # Absolute obstacle position NOW = ego pos + ego-relative offset.
                opos.append([s0[0] + rel_xy[0], s0[1] + rel_xy[1]])
                ovel.append([obs_v[0], obs_v[1]])
            else:
                # Empty slot: parked far away so its keep-out is never active.
                opos.append([1e6, 1e6])
                ovel.append([0.0, 0.0])
        for v2 in opos: p += v2
        for v2 in ovel: p += v2

        # Static (LiDAR) keep-out points: the k_static nearest OCC cells to the ego.
        self._spos_used = None
        if self.k_static:
            sp = self._nearest_points(s0, static_pts, self.k_static, STATIC_QUERY_R)
            self._spos_used = sp                 # kept for per-step diagnostics
            for row in sp:
                p += [float(row[0]), float(row[1])]

        # Occlusion boundary points: the k_occ nearest blind-spot cells to the ego.
        # Drop any that sit within the (fully expanded) keep-out radius of the GOAL —
        # the goal is a known-safe target, so a phantom agent must not be assumed to
        # hide on it; otherwise the widened, expanding occlusion circle would cover the
        # goal and repel the ego from its own destination (the orbit failure mode).
        if self.k_occ:
            occ_src = occ_pts
            if occ_src is not None and len(occ_src) and goal_xy is not None:
                r_goal_clear = self.d_safe_occ + self.v_target * self.N * DT
                gd = np.hypot(np.asarray(occ_src)[:, 0] - goal_xy[0],
                              np.asarray(occ_src)[:, 1] - goal_xy[1])
                occ_src = np.asarray(occ_src)[gd > r_goal_clear]
            op = self._nearest_points(s0, occ_src, self.k_occ, OCC_QUERY_R)
            for row in op:
                p += [float(row[0]), float(row[1])]
        return np.array(p, dtype=float)

    @staticmethod
    def _nearest_points(s0, pts, k, query_r):
        """Pick the k nearest points (within query_r of the ego); pad far-parked if fewer."""
        out = np.full((k, 2), 1e6)
        if pts is not None and len(pts):
            pts = np.asarray(pts, float)
            d   = np.hypot(pts[:, 0] - s0[0], pts[:, 1] - s0[1])
            near = pts[d <= query_r]
            dn   = d[d <= query_r]
            if len(near):
                order = np.argsort(dn)[:k]
                out[:len(order)] = near[order]
        return out

    def _maybe_bias_guess(self, s0, goal_xy, obstacles, X0, U0):
        """Curve the warm-start guess to one side around a dead-ahead obstacle.

        Finds the nearest slow obstacle that sits ahead and nearly on the ego→goal
        line; if one exists, overwrites the guess with a one-sided curved rollout so
        the solver escapes the symmetric straight-through ridge. Returns (X0, U0).
        """
        ex, ey = float(s0[0]), float(s0[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        gvec   = np.array([gx - ex, gy - ey])
        gnorm  = np.hypot(*gvec)
        if gnorm < 1e-3:
            return X0, U0
        ghat = gvec / gnorm
        nhat = np.array([-ghat[1], ghat[0]])   # left-hand normal to the goal line

        best = None   # (along_dist, lateral_signed)
        for rel_xy, obs_v in obstacles:
            r = np.asarray(rel_xy, float)
            along = float(r @ ghat)
            if not (0.0 < along < SYM_AHEAD_RANGE):
                continue
            lateral = float(r @ nhat)
            if abs(lateral) > SYM_LAT_THRESH:
                continue
            if best is None or along < best[0]:
                best = (along, lateral)
        if best is None:
            return X0, U0

        # Steer AWAY from the side the obstacle leans (default left for a perfect tie).
        side  = -np.sign(best[1]) if abs(best[1]) > 0.1 else 1.0
        n_bias = max(1, int(self.N * SYM_BIAS_FRAC))
        Ub = U0.copy()
        Ub[:n_bias, 1] = side * SYM_BIAS
        Ub[:, 0] = np.clip(Ub[:, 0], A_MIN, A_MAX)
        Ub[:, 1] = np.clip(Ub[:, 1], -DELTA_LIM, DELTA_LIM)

        # Roll the seeded controls through the model to get a consistent state guess.
        Xb = np.empty_like(X0)
        Xb[0] = np.asarray(s0, float)
        st = np.asarray(s0, float)
        for k in range(self.N):
            st = np.asarray(self.f(st, Ub[k])).flatten()
            Xb[k + 1] = st
        return Xb, Ub

    def _flatten(self, Xarr, Uarr):
        """Pack (Nh+1, nx) states + (Nh, nu) controls into w."""
        Nh = self.N
        parts  = [Xarr[k] for k in range(Nh + 1)]
        parts += [Uarr[k] for k in range(Nh)]
        return np.concatenate(parts)

    # ── Solve the receding-horizon problem, return the first control ──────────
    def solve(self, s0, goal_xy, obstacles, u_prev, static_pts=None, occ_pts=None):
        """Return (u0, info). u0 = [a_cmd, delta_cmd] to apply this step.

        static_pts: optional (M, 2) array of static OCC world points (from the LiDAR
        costmap); the k_static nearest to the ego are added as keep-out points.
        occ_pts: optional (M, 2) array of occlusion-boundary world points (blind-spot
        cells from the LiDAR costmap); the k_occ nearest seed expanding keep-outs.
        """
        nx, nu, Nh = self.NX, self.NU, self.N

        # Goal-approach speed taper (identical to the MPPI controller).
        d0        = float(np.hypot(goal_xy[0] - s0[0], goal_xy[1] - s0[1]))
        slow_frac = float(np.clip(d0 / GOAL_SLOWDOWN_DIST, 0.0, 1.0))
        v_des_eff = GOAL_MIN_SPEED + (V_DES - GOAL_MIN_SPEED) * slow_frac
        # Inside the capture zone, ramp the target down to 0 at the goal so the ego
        # brakes to a stop instead of orbiting at the GOAL_MIN_SPEED floor. Continuous:
        # the factor is 1 beyond GOAL_STOP_DIST, so the outer taper is unaffected.
        v_des_eff *= float(np.clip(d0 / GOAL_STOP_DIST, 0.0, 1.0))

        p = self._pack_params(s0, goal_xy, v_des_eff, u_prev, obstacles, static_pts, occ_pts)

        # ── Warm start: shift previous solution, else roll out from s0 ─────────
        if self._Xopt is None:
            X0 = np.tile(np.asarray(s0, float), (Nh + 1, 1))
            U0 = np.zeros((Nh, nu))
        else:
            # Shift: drop the executed stage, duplicate the last.
            X0 = np.vstack([self._Xopt[1:], self._Xopt[-1:]])
            U0 = np.vstack([self._Uopt[1:], self._Uopt[-1:]])

        # Break the dead-ahead symmetry: if a slow obstacle sits nearly on the
        # ego→goal line, a straight guess sits on a zero-lateral-gradient ridge and
        # IPOPT can converge to plowing straight through. Re-seed the guess with a
        # one-sided curved rollout so it commits to a go-around. Constraints (the
        # true obstacle, full D_SAFE) are untouched, so safety is preserved.
        X0, U0 = self._maybe_bias_guess(s0, goal_xy, obstacles, X0, U0)
        w0 = self._flatten(X0, U0)

        args = dict(x0=w0, p=p,
                    lbx=self.lbx, ubx=self.ubx,
                    lbg=self.lbg, ubg=self.ubg)
        if self._lam_x0 is not None:
            args["lam_x0"] = self._lam_x0
            args["lam_g0"] = self._lam_g0

        t0  = time.perf_counter()
        sol = self.solver(**args)
        dt_solve = time.perf_counter() - t0

        stats = self.solver.stats()
        ok    = stats.get("success", False)
        w_opt = np.asarray(sol["x"]).flatten()

        if self.v_safe_fn is not None:
            v_safe_vals = np.asarray(self.v_safe_fn(sol["x"], p)).flatten()
            #print("v_safe:", v_safe_vals)

        # Unpack states/controls from the flat solution.
        Xopt = w_opt[:self.n_x].reshape(Nh + 1, nx)
        Uopt = w_opt[self.n_x:self.n_x + self.n_u].reshape(Nh, nu)

        # A converged, finite solution is committed and cached for warm-starting.
        # Otherwise fall back to a SAFE finite command — never emit NaN/inf to Unity
        # (a non-finite action crashes the agent and drops the communicator). The
        # fallback is the next command of the previous plan (the shifted guess U0[0],
        # finite by construction: a prior optimum or zeros), so a transient solver
        # hiccup just replays the last good plan instead of injecting garbage.
        u0        = Uopt[0]
        finite    = bool(np.all(np.isfinite(u0)))
        used_fallback = False
        if ok and finite:
            self._Xopt   = Xopt
            self._Uopt   = Uopt
            self._lam_x0 = sol["lam_x"]
            self._lam_g0 = sol["lam_g"]
        else:
            u0 = U0[0]                       # shifted previous plan (or zeros on step 1)
            if not np.all(np.isfinite(u0)):
                u0 = np.zeros(self.NU)       # last-resort: coast (drag decelerates)
            used_fallback = True
            # Even on a non-converged solve, cache the (finite) PRIMAL iterate as the
            # next warm start so the following solve resumes from it instead of
            # re-seeding from a stale shifted plan. But DROP the multipliers: warm-
            # starting the duals from a diverged iterate was sustaining a fallback
            # cascade (every subsequent solve inherited the bad point and also hit the
            # iter cap, permanently). Letting IPOPT recompute the multipliers from the
            # primal guess lets the next solve recover.
            if np.all(np.isfinite(Xopt)) and np.all(np.isfinite(Uopt)):
                self._Xopt   = Xopt
                self._Uopt   = Uopt
                self._lam_x0 = None
                self._lam_g0 = None
            else:
                # Non-finite iterate: discard it entirely and re-seed clean next step.
                self._Xopt = self._Uopt = self._lam_x0 = self._lam_g0 = None

        a_cmd     = float(np.clip(u0[0], A_MIN, A_MAX))
        delta_cmd = float(np.clip(u0[1], -DELTA_LIM, DELTA_LIM))

        # ── Static keep-out diagnostics ──────────────────────────────────────
        # max_sst: largest static keep-out violation (D_SAFE_STATIC² − d²)₊ [m²] the
        #   PLANNED trajectory incurs — how much the penalised keep-out is breached
        #   (0 ⇒ plan clears every constrained static point).
        # min_pred_stat: the SMALLEST clearance to the constrained static points the
        #   PLANNED trajectory achieves [m] — compare to the measured nearStatic to see
        #   whether the plan intends to keep D_SAFE_STATIC but can't, or plans to breach.
        max_sst = 0.0
        min_pred_stat = np.inf
        if self._spos_used is not None:
            sp = self._spos_used[self._spos_used[:, 0] < 1e5]     # drop far-parked pads
            if len(sp):
                dxy = (Xopt[1:, None, :2] - sp[None, :, :])        # (Nh, K, 2)
                d   = np.hypot(dxy[..., 0], dxy[..., 1])
                min_pred_stat = float(d.min())
                max_sst = float(np.max(np.maximum(0.0,
                                                  self.d_safe_static ** 2 - d ** 2)))

        info = {"solve_ms": dt_solve * 1e3, "success": bool(ok and finite),
                "fallback": used_fallback,
                "iter": stats.get("iter_count", -1),
                "status": stats.get("return_status", "?"),
                "max_sst": max_sst, "min_pred_stat": min_pred_stat}
        return np.array([a_cmd, delta_cmd]), info


# ── Main control loop (MPC) ───────────────────────────────────────────────────

def run(unity_exec_path=None, port=5004, run_sysid=False, n_episodes=20,
        min_difficulty=0.0, max_difficulty=1.0, noise_std=0.0,
        scenario_type=SCENARIO_STANDARD, detect_range=float("inf"),
        d_infl=D_INFL, d_safe=D_SAFE, horizon=N_MPC, pin_episode=None,
        lidar_costmap=False, lidar_topic="/point_cloud", save_traj=None,
        occlusion_aware=False, spawn_heading_deg=None, spawn_heading_jitter_deg=0.0):
    # Detection range lives as a module global inside taxi_controller (obs_to_state
    # reads it), so set it there rather than shadowing a local copy.
    tc.DETECTION_RANGE = detect_range

    print(f"[MPC] Connecting to Unity on port {port} ...")
    print(f"[MPC] Horizon N={horizon} ({horizon * DT:.1f} s)  "
          f"D_INFL={d_infl:.1f} m  D_SAFE={d_safe:.1f} m")
    if detect_range < float("inf"):
        print(f"[MPC] Detection range : {detect_range:.1f} m  (obstacles beyond masked)")

    # ── Optional LiDAR static-obstacle keep-out (buildings / parked aircraft) ──
    # These surfaces never appear in the observation vector; they come only from
    # the published PointCloud2 via the persistent LidarCostmap grid. Enabling this
    # adds the K_STATIC nearest OCC-cell keep-out points to the MPC each step; with
    # --occlusion-aware, the grid's occlusion_points() (blind-corner cells) seed the
    # K_OCC expanding forward-reachable-set keep-outs.
    costmap  = None
    k_static = 0
    k_occ    = 0
    if lidar_costmap:
        from lidar_costmap import LidarCostmap
        cm = LidarCostmap(max_age=1.5, enable_visibility=False)
        if cm.start(topic=lidar_topic):
            costmap  = cm
            k_static = K_STATIC
            print(f"[MPC] LiDAR map     : ON  (topic={lidar_topic}) — static keep-out "
                  f"K_STATIC={K_STATIC}, D_SAFE_STATIC={D_SAFE_STATIC:.1f} m")
            if occlusion_aware:
                k_occ = K_OCC
                print(f"[MPC] Occlusion-aware: ON  — forward reachable sets K_OCC={K_OCC}, "
                      f"v_target={V_TARGET:.1f} m/s, D_SAFE_OCC={D_SAFE_OCC:.1f} m "
                      f"(expands +{V_TARGET*horizon*DT:.1f} m over the horizon)")
        else:
            print("[MPC] LiDAR map     : requested but unavailable (no rclpy) — "
                  "running WITHOUT static keep-out")
    elif occlusion_aware:
        print("[MPC] Occlusion-aware: requested but needs --lidar-costmap — DISABLED")

    mpc = TaxiMPC(N=horizon, k_obs=K_OBS, d_infl=d_infl, d_safe=d_safe,
                  k_static=k_static, k_occ=k_occ)

    env_params = EnvironmentParametersChannel()
    env = UnityEnvironment(
        file_name=unity_exec_path,
        base_port=port,
        seed=42,
        no_graphics=unity_exec_path is not None,
        side_channels=[env_params],
    )
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec          = env.behavior_specs[behavior_name]
    print(f"[MPC] Behavior  : {behavior_name}")
    print(f"[MPC] Obs shape : {spec.observation_specs[0].shape}")
    print(f"[MPC] Act size  : {spec.action_spec.continuous_size}")

    obs_size = spec.observation_specs[0].shape[0]
    act_size = spec.action_spec.continuous_size
    assert obs_size == OBS_SIZE, (
        f"Expected {OBS_SIZE} observations, got {obs_size}. "
        f"Set BehaviorParameters Vector Observations = {OBS_SIZE} in Unity Inspector "
        f"and ensure TaxiAgent.cs K_OBS = {K_OBS}."
    )
    assert act_size == 2, f"Expected 2 continuous actions, got {act_size}."

    if run_sysid:
        identify_bicycle_model(env, behavior_name)
        env.reset()

    scenarios = make_scenarios(n_episodes, min_difficulty=min_difficulty,
                               max_difficulty=max_difficulty,
                               scenario_type=scenario_type,
                               spawn_heading_deg=spawn_heading_deg,
                               spawn_heading_jitter_deg=spawn_heading_jitter_deg)
    if pin_episode is not None:
        pinned = make_scenarios(1, base_seed=pin_episode,
                                min_difficulty=min_difficulty,
                                max_difficulty=max_difficulty,
                                scenario_type=scenario_type,
                                spawn_heading_deg=spawn_heading_deg,
                                spawn_heading_jitter_deg=spawn_heading_jitter_deg)[0]
        scenarios = [dict(pinned) for _ in range(n_episodes)]
        print(f"[MPC] Pinned episode  : seed={pin_episode} — identical scenario every episode")

    episode_stats = []
    unity_stopped = False

    for ep in range(n_episodes):
        sc      = scenarios[ep]
        task_id = int(sc["scenario_type"])
        ep_log  = {"min_h": np.inf, "min_dist": np.inf,
                   "collided": False, "reached": False, "steps": 0,
                   "incursion_dt": sc["incursion_dt"],
                   "difficulty": sc["difficulty"],
                   "scenario": SCENARIO_NAMES[task_id]}
        traj      = []
        obs_track = []
        # Occlusion-boundary points actually constrained by the MPC, deduped by rounded
        # world cell (a mouth that sweeps as the ego moves would otherwise paint a whole
        # line across many steps even when each step is corner-only).
        occ_track = {}
        # Union of every OCC cell seen across the episode. The costmap is a rolling,
        # DECAYING, ego-centered grid (occ_ttl≈12s), so a single end-of-run snapshot
        # only shows walls still in view at the goal — walls the ego passed earlier
        # have decayed/rolled out. Accumulate (dedup by cell) to show the real extent.
        occ_seen = {}

        for key, val in sc.items():
            env_params.set_float_parameter(key, val)

        sname = SCENARIO_NAMES[int(sc["scenario_type"])]
        print(f"\n[Ep {ep+1:3d}] [{sname}] Δt={sc['incursion_dt']:+.2f}s  "
              f"v_amb={sc['ambulance_speed']:.2f} m/s  diff={sc['difficulty']:.2f}  "
              f"noise={noise_std:.3f}  dir={'L' if sc['cross_dir_sign'] < 0 else 'R'}")

        try:
            env.reset()
        except UnityCommunicatorStoppedException:
            unity_stopped = True
            break
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        ep_steps     = 0
        delta_actual = 0.0
        accel_actual = 0.0
        u_prev       = np.zeros(2)
        # Fresh MPC warm-start each episode so runs are comparable.
        mpc._Xopt = mpc._Uopt = mpc._lam_x0 = mpc._lam_g0 = None
        episode_done = False
        solve_ms_acc = 0.0
        solve_fail   = 0

        while not episode_done:
            if len(terminal_steps) > 0:
                ep_log["collided"] = ep_log["min_h"] < 0.0
                ep_log["reached"]  = (not terminal_steps.interrupted[0]
                                      and not ep_log["collided"])
                episode_done = True
                break

            if len(decision_steps) == 0:
                env.step()
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                continue

            obs = decision_steps.obs[0][0]
            ep_steps += 1

            obs_n = inject_sensor_noise(obs, noise_std, tc.rng)
            s, obstacles, goal_xy = obs_to_state(obs_n, delta_actual, accel_actual)

            # ── Refresh the LiDAR static-obstacle costmap and extract OCC points ──
            static_pts = None
            occ_pts    = None
            n_static   = 0
            n_occ      = 0
            if costmap is not None:
                ego_fwd = np.array([np.cos(s[2]), np.sin(s[2])])
                # Path-relevance gate: keep only occlusion shadow within a corridor of
                # the ego→goal route (drops the wall driven alongside / beside the goal).
                costmap.set_goal(goal_xy[0], goal_xy[1])
                costmap.update(ego_fwd)
                if costmap.ready:
                    g = costmap.grid
                    oi, oj = np.where(g.state == 2)              # OCC cells
                    if len(oi):
                        static_pts = np.column_stack((
                            g._o0 + (oi + 0.5) * g.res,          # world a0 (Unity Z)
                            g._o1 + (oj + 0.5) * g.res))         # world a1 (Unity X)
                        n_static = len(static_pts)
                        if save_traj is not None:
                            # Dedup by rounded world cell so the union stays bounded.
                            for wx, wy in static_pts:
                                occ_seen[(round(float(wx), 1), round(float(wy), 1))] = None
                    if k_occ:
                        occ_pts = costmap.occlusion_points()     # blind-spot boundary cells
                        n_occ   = 0 if occ_pts is None else len(occ_pts)

            # ── MPC solve ─────────────────────────────────────────────────────
            # MPC state is [x, y, theta, v, accel]; delta_actual is not a state
            # (the steering-rate limit is a constraint), so it isn't passed.
            s_mpc = np.array([s[0], s[1], s[2], s[3], accel_actual])
            u_cmd, info = mpc.solve(s_mpc, goal_xy, obstacles, u_prev, static_pts, occ_pts)
            solve_ms_acc += info["solve_ms"]
            solve_fail   += (0 if info["success"] else 1)

            a_cmd     = float(np.clip(u_cmd[0], A_MIN, A_MAX))
            delta_cmd = float(np.clip(u_cmd[1], -DELTA_LIM, DELTA_LIM))
            u_prev    = np.array([a_cmd, delta_cmd])

            # Per-step diagnostic: how many obstacles the MPC sees, the nearest one,
            # and the solve status. Prints on step 1, every 10 steps, and on any
            # solver fallback — so it's obvious whether obstacles are reaching the
            # planner and whether IPOPT is converging.
            n_obs   = len(obstacles)
            near    = min((np.hypot(r[0], r[1]) for r, _ in obstacles), default=np.inf)
            near_st = np.inf
            if static_pts is not None and len(static_pts):
                near_st = float(np.min(np.hypot(static_pts[:, 0] - s[0],
                                                static_pts[:, 1] - s[1])))
            if ep_steps == 1 or ep_steps % 10 == 0 or info["fallback"]:
                tag = " FALLBACK" if info["fallback"] else ""
                mps = info.get("min_pred_stat", np.inf)
                print(f"  [step {ep_steps:4d}] n_obs={n_obs} near={near:6.1f}m  "
                      f"nStatic={n_static} nearStatic={near_st:6.1f}m  "
                      f"minPredStat={mps:6.1f}m maxSlack={info.get('max_sst', 0.0):7.1f}  "
                      f"nOcc={n_occ}  "
                      f"v={s[3]:5.2f}  a={a_cmd:+.2f} δ={delta_cmd:+.3f}  "
                      f"solve={info['solve_ms']:5.1f}ms iter={info['iter']:3d} "
                      f"{info['status']}{tag}")

            # Advance Python-side kinematic state to match what Unity computes,
            # so the next MPC solve starts from the right delta/accel actuals.
            v            = s[3]
            speed_frac   = min(v / max(STEER_ROLLOFF_SPD, 1e-3), 1.0)
            eff_limit    = DELTA_LIM * (1.0 - speed_frac * (1.0 - STEER_ROLLOFF_MIN))
            delta_target = float(np.clip(delta_cmd, -eff_limit, eff_limit))
            delta_actual = delta_actual + float(np.clip(
                delta_target - delta_actual, -MAX_STEER_RATE * DT, MAX_STEER_RATE * DT))
            accel_actual += (float(np.clip(a_cmd, A_MIN, A_MAX)) - accel_actual) * (DT / ACCEL_TAU)

            h_val = float(obs[17])
            dist  = float(min((np.hypot(rel[0], rel[1]) for rel, _ in obstacles),
                              default=np.inf))
            ep_log["min_h"]    = min(ep_log["min_h"], h_val)
            ep_log["min_dist"] = min(ep_log["min_dist"], dist)
            ep_log["steps"]    = ep_steps

            traj.append((ep_steps * DT, s[0], s[1], s[2], s[3], a_cmd, delta_cmd))
            for rel, _ov in obstacles:
                obs_track.append((ep_steps * DT, s[0] + rel[0], s[1] + rel[1]))
            # Record the occlusion-boundary points the MPC actually constrained this
            # step (the k_occ nearest within OCC_QUERY_R) so the plot shows what is
            # pushing the ego. _nearest_points pads unused slots to 1e6 — drop those.
            if k_occ and occ_pts is not None and len(occ_pts):
                near_occ = mpc._nearest_points(s_mpc, occ_pts, k_occ, OCC_QUERY_R)
                for row in near_occ:
                    if row[0] < 1e5:
                        occ_track[(round(float(row[0]), 1),
                                   round(float(row[1]), 1))] = None

            action = ActionTuple(
                continuous=np.array([[a_cmd, delta_cmd]], dtype=np.float32))
            env.set_actions(behavior_name, action)
            try:
                env.step()
            except UnityCommunicatorStoppedException:
                unity_stopped = True
                break
            decision_steps, terminal_steps = env.get_steps(behavior_name)

        if unity_stopped:
            break
        episode_stats.append(ep_log)

        if save_traj is not None and traj:
            # Overlay the UNION of static occluders seen across the whole episode
            # (not the decaying end-of-run snapshot) so the plot shows every wall the
            # ego actually reacted to, including ones it has since driven past.
            occ_pts = np.array(list(occ_seen.keys())) if occ_seen else None
            occ_boundary = (np.array(list(occ_track.keys())) if occ_track else None)
            _save_trajectory(save_traj, ep, ep_log, goal_xy, np.asarray(traj),
                             np.asarray(obs_track) if obs_track else None, occ_pts,
                             occ_boundary=occ_boundary, occ_horizon_s=mpc.N * DT)

        verdict   = "COLLISION" if ep_log["collided"] else "safe"
        mean_ms   = solve_ms_acc / max(ep_log["steps"], 1)
        print(f"[Ep {ep+1:3d}] Δt={ep_log['incursion_dt']:+.2f}s  "
              f"diff={ep_log['difficulty']:.2f}  steps={ep_log['steps']:4d}  "
              f"min_dist={ep_log['min_dist']:5.2f}m  min_h={ep_log['min_h']:8.2f}  "
              f"solve≈{mean_ms:4.1f}ms  fails={solve_fail}  → {verdict}")

    if unity_stopped:
        print(f"\n[MPC] Unity stopped early. Completed {len(episode_stats)}/{n_episodes} "
              f"episodes — reporting those.")
    try:
        env.close()
    except Exception:
        pass

    # ── Summary (same layout as the MPPI controller) ──────────────────────────
    print("\n=== Summary (MPC) ===")
    n = len(episode_stats)
    if n == 0:
        print("No episodes completed — nothing to summarise.")
        return
    col   = sum(1 for e in episode_stats if e["collided"])
    dists = [e["min_dist"] for e in episode_stats]
    print(f"Collision rate : {col}/{n} = {col/n:.1%}")

    genuine = [e for e in episode_stats if e["min_dist"] <= DUD_DIST]
    n_gen   = len(genuine)
    col_gen = sum(1 for e in genuine if e["collided"])
    n_dud   = n - n_gen
    if n_gen:
        print(f"Conditional    : {col_gen}/{n_gen} = {col_gen/n_gen:.1%}  "
              f"(genuine conflicts only; {n_dud} duds excluded)")
    else:
        print(f"Conditional    : n/a (all {n} episodes were duds)")

    print(f"Mean min_dist  : {np.mean(dists):.2f} m  (median {np.median(dists):.2f} m)")
    print(f"Worst min_dist : {np.min(dists):.2f} m  (target >= {d_safe:.1f} m)")
    print(f"Worst min_h    : {np.min([e['min_h'] for e in episode_stats]):.2f}  (target >= 0)")

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

    return episode_stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MPC (CasADi/IPOPT) taxiing controller.")
    p.add_argument("--exec",           default=None)
    p.add_argument("--port",           default=5004, type=int)
    p.add_argument("--sysid",          default=False, type=lambda x: x.lower() == "true")
    p.add_argument("--episodes",       default=20, type=int)
    p.add_argument("--min-difficulty", default=0.0, type=float)
    p.add_argument("--max-difficulty", default=1.0, type=float)
    p.add_argument("--noise-std",      default=0.0, type=float,
                   help="Std-dev of Gaussian noise injected into obstacle obs [m]. 0=off.")
    p.add_argument("--scenario",       default="standard",
                   choices=SCENARIO_NAMES + ["mixed"],
                   help="Force a scenario type for all episodes, or 'mixed' to randomise.")
    p.add_argument("--detect-range",   default=float("inf"), type=float,
                   help="Euclidean detection range [m]; obstacles beyond are masked. Default=inf.")
    p.add_argument("--horizon",        default=N_MPC, type=int,
                   help=f"MPC prediction horizon in steps (default {N_MPC}; {N_MPC*DT:.1f} s).")
    p.add_argument("--d-infl",         default=D_INFL, type=float,
                   help="Obstacle soft-influence radius [m]. Must stay >= --d-safe.")
    p.add_argument("--d-safe",         default=D_SAFE, type=float,
                   help="Hard keep-out radius [m] (steep penalty inside). Must stay <= --d-infl.")
    p.add_argument("--pin-episode",    default=None, type=int, metavar="SEED",
                   help="Replay one fixed scenario every episode.")
    p.add_argument("--lidar-costmap",  action="store_true",
                   help="Build a static-obstacle keep-out from the published PointCloud2 each "
                        "step and add the K_STATIC nearest OCC points to the MPC — walls/parked "
                        "aircraft/buildings get a D_SAFE_STATIC margin. Needs ROS 2 sourced (rclpy) "
                        "and the ros_tcp_endpoint running. WITHOUT this the MPC only avoids the "
                        "dynamic obstacles in the observation vector.")
    p.add_argument("--lidar-topic",    default="/point_cloud",
                   help="PointCloud2 topic for the LiDAR map. Default: /point_cloud.")
    p.add_argument("--occlusion-aware", action="store_true",
                   help="Add occlusion-aware forward-reachable-set keep-outs (Firoozi et al.): "
                        "the K_OCC nearest blind-spot cells behind occluders each seed an "
                        "EXPANDING keep-out circle (radius grows as v_target·t over the horizon), "
                        "so the ego gives occluded corners a wider berth. Requires --lidar-costmap.")
    p.add_argument("--save-traj",      default=None, metavar="DIR",
                   help="Save each episode's ego trajectory as CSV + a top-down PNG plot into DIR.")
    p.add_argument("--spawn-heading-deg", default=None, type=float, metavar="DEG",
                   help="Fixed ego spawn heading [deg, world yaw] at the start marker. Omit to keep "
                        "Unity's default spawn behaviour (identity heading).")
    p.add_argument("--spawn-heading-jitter-deg", default=0.0, type=float, metavar="DEG",
                   help="+/- random jitter added to --spawn-heading-deg each episode (reproducible, "
                        "seeded off episode_seed). No effect unless --spawn-heading-deg is set.")
    args = p.parse_args()

    if args.d_infl < args.d_safe:
        p.error(f"--d-infl ({args.d_infl}) must be >= --d-safe ({args.d_safe}).")

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
        d_infl=args.d_infl,
        d_safe=args.d_safe,
        horizon=args.horizon,
        pin_episode=args.pin_episode,
        lidar_costmap=args.lidar_costmap,
        lidar_topic=args.lidar_topic,
        save_traj=args.save_traj,
        occlusion_aware=args.occlusion_aware,
        spawn_heading_deg=args.spawn_heading_deg,
        spawn_heading_jitter_deg=args.spawn_heading_jitter_deg)
