import argparse
import time

import numpy as np
import casadi as ca


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.exception import UnityCommunicatorStoppedException

import taxi_controller_mppi as tc
from taxi_controller_mppi import (
    DT, L, V_DES, GOAL_SLOWDOWN_DIST, GOAL_MIN_SPEED,
    A_MIN, A_MAX, DELTA_LIM,
    DRAG_COEFF, ACCEL_TAU, MAX_STEER_RATE, STEER_ROLLOFF_SPD, STEER_ROLLOFF_MIN,
    D_SAFE, D_INFL, OBS_SIZE,
    D_SAFE_HARD, W_HARD,
    V_TARGET, K_OCC, OCC_QUERY_R, OCC_FWD_HALF_ANGLE,
    SCAN_FOV_H, SCAN_FOV_V, SCAN_RES_H, SCAN_RES_V, SCAN_MAX_RANGE,
    OCC_TRACK_ASSOC, OCC_TRACK_ALPHA, OCC_TRACK_TTL, OCC_TRACK_HITS,
    W_SIGHT, A_BRAKE_SIGHT, V_SIGHT_FLOOR, OCC_HORIZON, OCC_T_GROW_MAX,
    obs_to_state,
    identify_bicycle_model, _save_trajectory,
)
from occlusion_capsules import (point_segment_distance_sym, point_segment_distance_np,
                                OcclusionCornerTracker, occlusion_stage_cost)

from taxi_config import CFG
_mpc, _occ, _goal = CFG["mpc"], CFG["occlusion"], CFG["goal"]

N_MPC   = _mpc["horizon"]


GOAL_STOP_DIST = _goal["stop_dist"]   # [m] distance within which the speed target ramps to 0


from taxi_cost import (W_GOAL_RUN, W_GOAL_TERM, W_HEAD, W_V, R_ACT, R_DACT)


from taxi_cost import W_OBS, RHO_SLACK, RHO_SLACK2


K_OBS = CFG["dynamic_obstacles"]["k_obs"]


RHO_HARD       = CFG["keepout"]["rho_hard"]
RHO_HARD2      = CFG["keepout"]["rho_hard2"]

K_STATIC       = _mpc["k_static"]        # nearest static OCC points constrained per step
STATIC_QUERY_R = _mpc["static_query_r"]  # only consider OCC cells within this radius [m]

SYM_LAT_THRESH  = _mpc["sym_lat_thresh"]
SYM_AHEAD_RANGE = _mpc["sym_ahead_range"]
SYM_BIAS        = _mpc["sym_bias"]
SYM_BIAS_FRAC   = _mpc["sym_bias_frac"]

OCC_USE_CAPSULES = _occ["use_capsules"]


OCC_SINGLE_CIRCLE     = _occ["single_circle"]
OCC_D_TARGET_HORIZON  = N_MPC * DT     # [s] only used when OCC_SINGLE_CIRCLE is True

# Per-step enlargement increment, for reference/printing: 3.0 m/s · 0.1 s = 0.3 m.
D_TARGET_PER_STEP     = V_TARGET * DT

class TaxiMPC:
    """Nonlinear MPC for the taxiing aircraft, built once and re-solved each step.

    The NLP is assembled symbolically in __init__ with the current state, goal,
    desired speed, last applied control and the (up to K_OBS) obstacle predictions
    as *parameters*, so each control step only updates numbers and re-solves —
    the standard receding-horizon MPC pattern. Warm-started from the previous
    solution (shifted one step) for fast convergence.
    """

    # which keeps the model smooth (IPOPT-friendly) rather than clamped.
    NX = 5   # state  [x, y, theta, v, accel]
    NU = 2   # control [a_cmd, delta_cmd]

    def __init__(self, N=N_MPC, k_obs=K_OBS, d_infl=D_INFL, d_safe=D_SAFE,
                 k_static=0, d_safe_hard=D_SAFE_HARD, w_hard=W_HARD,
                 k_occ=0, v_target=V_TARGET,
                 occ_d_target=None, verbose=False, terminal_stop=True):
        self.N     = int(N)
        self.k_obs = int(k_obs)
        self.d_infl = float(d_infl)
        self.d_safe = float(d_safe)
        self.k_static = int(k_static)
        self.d_safe_hard = float(d_safe_hard)
        self.w_hard      = float(w_hard)
        self.k_occ = int(k_occ)
        self.occ_d_target = None if occ_d_target is None else float(occ_d_target)
        self.v_target   = float(v_target)
        self.terminal_stop = bool(terminal_stop)   # z_N = z_{N-1} — see the constraint below

        nx, nu, Nh = self.NX, self.NU, self.N

        # ── Symbolic one-step dynamics f(x, u) ────────────────────────────────
        st = ca.MX.sym("st", nx)
        u  = ca.MX.sym("u",  nu)
        self.f = ca.Function("f", [st, u], [self._dynamics(st, u)])

        X = [ca.MX.sym(f"x_{k}", nx) for k in range(Nh + 1)]
        U = [ca.MX.sym(f"u_{k}", nu) for k in range(Nh)]
        S = [ca.MX.sym(f"s_{k}", 1)  for k in range(Nh)]

        P_s0    = ca.MX.sym("s0",    nx)          # current measured state
        P_goal  = ca.MX.sym("goal",  2)           # goal world position (x_fwd, y_lat)
        P_vdes  = ca.MX.sym("vdes",  1)           # goal-tapered desired speed
        P_uprev = ca.MX.sym("uprev", nu)          # last applied control (Δu_0 + rate limit)
        P_opos  = [ca.MX.sym(f"op_{i}", 2) for i in range(self.k_obs)]  # abs obstacle pos now
        P_ovel  = [ca.MX.sym(f"ov_{i}", 2) for i in range(self.k_obs)]  # obstacle velocity

        P_spos  = [ca.MX.sym(f"sp_{i}", 3) for i in range(self.k_static)]

        P_occ   = [ca.MX.sym(f"occ_{i}", 4) for i in range(self.k_occ)]

        R_act_dm  = ca.DM(R_ACT)
        R_dact_dm = ca.DM(R_DACT)
        d_safe2      = self.d_safe ** 2
        max_dstep = MAX_STEER_RATE * DT           # per-step steering-rate cap [rad]

        g_eq   = [X[0] - P_s0]   # equalities: initial state + multiple-shooting defects
        g_rate = []              # inequalities: |Δdelta_cmd| ≤ max_dstep  (linear)
        g_hard = []              # inequalities: d − r_keep + s_k ≥ 0  (the keep-outs)
        cost   = 0

        for k in range(Nh):
            xk, uk = X[k], U[k]
            g_eq.append(X[k + 1] - self.f(xk, uk))

            xn = X[k + 1]
            px, py, th, v = xn[0], xn[1], xn[2], xn[3]

            # Gentle goal-position pull (LINEAR distance → bounded gradient) +
            # heading alignment toward the goal bearing.
            d_goal = ca.sqrt((px - P_goal[0]) ** 2 + (py - P_goal[1]) ** 2 + 1.0)
            cost += W_GOAL_RUN * d_goal
            # psi   = ca.atan2(P_goal[1] - py, P_goal[0] - px)
            # cost += W_HEAD * (1 - ca.cos(th - psi))

            # # Speed tracking toward the (goal-tapered) desired speed.
            # cost += W_V * (v - P_vdes) ** 2

            # Control effort + rate (smoothness). Δu_0 uses the last applied control.
            # u_km1 = P_uprev if k == 0 else U[k - 1]
            # du    = uk - u_km1
            # cost += ca.dot(R_act_dm  * uk, uk)
            # cost += ca.dot(R_dact_dm * du, du)

            # Steering-rate limit as a linear inequality: −Δ ≤ delta_k − delta_{k−1} ≤ Δ.
            # g_rate.append(uk[1] - u_km1[1])

            # This stage's slack: the largest keep-out violation tolerated at stage k.
            # THE PENALTY IS NOT OPTIONAL. sk enters every keep-out below as
            # d − r_keep + sk ≥ 0, so an UNPENALISED sk is free: the solver simply raises
            # it until every constraint holds and the keep-outs stop existing. Commenting
            # this line out made the plan run 2.5 m INSIDE a static circle whose margin is
            # 15 m. Exact-penalty weights, deliberately not w_hard — see RHO_HARD.
            sk = S[k]
            cost += RHO_HARD * sk + RHO_HARD2 * sk * sk

            t_k = (k + 1) * DT

            # Hard constraint on the sensed dynamic obstacles, on a constant-velocity
            # prediction to t_k: ‖p_ego − p_obs(t_k)‖ ≥ d_safe.
            for i in range(self.k_obs):
                op   = P_opos[i] + P_ovel[i] * t_k
                dx_o = px - op[0]
                dy_o = py - op[1]
                d_o  = ca.sqrt(dx_o * dx_o + dy_o * dy_o + 1e-6)
                g_hard.append(d_o - self.d_safe + sk)


            # Hard constraint on the static surfaces
            for i in range(self.k_static):
                sp   = P_spos[i]
                r_s  = sp[2]
                dx_s = px - sp[0]
                dy_s = py - sp[1]
                d2s  = dx_s * dx_s + dy_s * dy_s
                ds   = ca.sqrt(d2s + 1e-6)
                r_keep_s = self.d_safe_hard + r_s
                g_hard.append(ds - r_keep_s + sk)


            t_eff = (self.occ_d_target / self.v_target if self.occ_d_target is not None
                     else t_k)

            # Hard constraint on the occlusion
            r_keep_occ = self.d_safe_hard + self.v_target * min(t_eff, OCC_T_GROW_MAX)
            d_vis = ca.MX(1e6)           # distance to the NEAREST boundary, for the RSS cap
            for i in range(self.k_occ):
                oc = P_occ[i]
                d2c, dc = point_segment_distance_sym(
                    px, py, oc[0], oc[1], oc[2], oc[3],
                    ca.fmin, ca.fmax, ca.sqrt)
                d_vis = ca.fmin(d_vis, dc)
                g_hard.append(dc - r_keep_occ + sk)
                
        if self.terminal_stop:
            g_eq.append(X[Nh] - X[Nh - 1])

        # Terminal goal cost pull
        xn = X[Nh]
        cost += W_GOAL_TERM * ca.sqrt((xn[0] - P_goal[0]) ** 2 + (xn[1] - P_goal[1]) ** 2 + 1.0)

        # Slacks go LAST in the decision vector, so the existing X/U unpacking offsets in
        # solve() and _flatten() are unchanged.
        w = ca.vertcat(*X, *U, *S)
        p = ca.vertcat(P_s0, P_goal, P_vdes, P_uprev, *P_opos, *P_ovel, *P_spos, *P_occ)
        g = ca.vertcat(*g_eq, *g_rate, *g_hard)
        nlp = {"x": w, "p": p, "f": cost, "g": g}

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

        self.n_x  = nx * (Nh + 1)
        self.n_u  = nu * Nh
        self.n_s  = Nh                       # one keep-out slack per stage
        # initial + Nh dynamics defects (+ the terminal stationarity block when enabled).
        # Counted from the list for the same reason n_rate is — each entry is an nx-vector.
        n_eq    = nx * len(g_eq)
        # Sized from the LISTS, not from Nh: the stage loop's terms are individually
        # switchable (the steering-rate constraint is currently commented out), and a
        # hard-coded count silently desyncs g from lbg/ubg — IPOPT then rejects the call
        # with a "mismatching shape" error rather than anything that names the cause.
        n_rate  = len(g_rate)                # steering-rate inequalities
        n_hard  = len(g_hard)                # (k_obs + k_static + k_occ) · Nh keep-outs

        # States: v ≥ 0 (bound), rest free; controls box-constrained; slacks ≥ 0.
        lbx, ubx = [], []
        for _ in range(Nh + 1):
            lbx += [-ca.inf, -ca.inf, -ca.inf, 0.0, -ca.inf]   # v ≥ 0
            ubx += [ ca.inf,  ca.inf,  ca.inf, ca.inf, ca.inf]
        for _ in range(Nh):
            lbx += [A_MIN, -DELTA_LIM]
            ubx += [A_MAX,  DELTA_LIM]
        lbx += [0.0] * self.n_s
        ubx += [ca.inf] * self.n_s
        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)
        # Equalities = 0; steering-rate in [−max_dstep, +max_dstep]; keep-outs ≥ 0 (one-sided).
        self.lbg = np.concatenate([np.zeros(n_eq),
                                   np.full(n_rate, -max_dstep),
                                   np.zeros(n_hard)])
        self.ubg = np.concatenate([np.zeros(n_eq),
                                   np.full(n_rate,  max_dstep),
                                   np.full(n_hard, ca.inf)])

        # Warm-start caches (previous optimal trajectory + duals).
        self._Xopt = None    # (Nh+1, nx)
        self._Uopt = None    # (Nh, nu)
        self._lam_x0 = None
        self._lam_g0 = None
        self.verbose = verbose

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

    def _pack_params(self, s0, goal_xy, v_des, u_prev, obstacles,
                     static_pts=None, occ_pts=None, occ_segs=None):
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

        # Static keep-out CIRCLES: the k_static nearest covering circles to the ego,
        # each (x, y, radius).
        self._spos_used = None
        if self.k_static:
            sp = self._nearest_circles(s0, static_pts, self.k_static, STATIC_QUERY_R)
            self._spos_used = sp                 # kept for per-step diagnostics
            for row in sp:
                p += [float(row[0]), float(row[1]), float(row[2])]


        if self.k_occ:
            segs = self._as_segments(occ_segs, occ_pts)
            if segs is not None and len(segs) and goal_xy is not None:
                # Test the NEAR endpoint (the corner) against the goal: that is the point
                # the phantom is anchored to.
                gd = np.hypot(segs[:, 0, 0] - goal_xy[0], segs[:, 0, 1] - goal_xy[1])
                r_goal_clear = self.d_safe_hard + self.v_target * self.N * DT
                segs = segs[gd > r_goal_clear]
            os_ = self._nearest_segments(s0, segs, self.k_occ, OCC_QUERY_R)
            for row in os_:
                p += [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
        return np.array(p, dtype=float)

    @staticmethod
    def _as_segments(occ_segs, occ_pts):
        """Normalize either source to (M, 2, 2) segments. Points become DEGENERATE
        segments (a == b), whose capsule is exactly the old circle — so the legacy
        point-based costmap path keeps its previous behaviour bit for bit."""
        if occ_segs is not None and len(occ_segs):
            return np.asarray(occ_segs, float).reshape(-1, 2, 2)
        if occ_pts is not None and len(occ_pts):
            pts = np.asarray(occ_pts, float).reshape(-1, 2)
            return np.stack([pts, pts], axis=1)
        return None

    @staticmethod
    def _nearest_segments(s0, segs, k, query_r):
        """The k segments nearest the ego (by true point-to-segment distance), padded
        with far-parked degenerate segments so the parameter vector is fixed-size."""
        out = np.full((k, 4), 1e6)
        if segs is not None and len(segs):
            segs = np.asarray(segs, float).reshape(-1, 2, 2)
            ego = np.array([s0[0], s0[1]], float)
            d = np.array([point_segment_distance_np(ego[None, :], s[0], s[1])[0]
                          for s in segs])
            near = segs[d <= query_r]
            dn = d[d <= query_r]
            if len(near):
                order = np.argsort(dn)[:k]
                sel = near[order]
                out[:len(sel)] = sel.reshape(len(sel), 4)
        return out

    @staticmethod
    def _nearest_circles(s0, circles, k, query_r):
        """Pick the k circles nearest the ego by SURFACE distance (‖p−c‖ − r), within
        query_r; pad with far-parked zero-radius circles ([1e6, 1e6, 0]) if fewer, so
        the parameter vector is fixed-size and pads are never active."""
        out = np.full((k, 3), 1e6)
        out[:, 2] = 0.0
        if circles is not None and len(circles):
            circles = np.asarray(circles, float).reshape(-1, 3)
            surf = np.hypot(circles[:, 0] - s0[0], circles[:, 1] - s0[1]) - circles[:, 2]
            near = circles[surf <= query_r]
            dn   = surf[surf <= query_r]
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
        """Pack (Nh+1, nx) states + (Nh, nu) controls + zero slacks into w."""
        Nh = self.N
        parts  = [Xarr[k] for k in range(Nh + 1)]
        parts += [Uarr[k] for k in range(Nh)]
        parts += [np.zeros(self.n_s)]      # slacks start at 0 = "assume feasible"
        return np.concatenate(parts)

    # ── Solve the receding-horizon problem, return the first control ──────────
    def solve(self, s0, goal_xy, obstacles, u_prev, static_pts=None, occ_pts=None,
              occ_segs=None):
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

        p = self._pack_params(s0, goal_xy, v_des_eff, u_prev, obstacles, static_pts,
                              occ_pts, occ_segs)

        # ── Warm start: shift previous solution, else roll out from s0 ─────────
        if self._Xopt is None:
            X0 = np.tile(np.asarray(s0, float), (Nh + 1, 1))
            U0 = np.zeros((Nh, nu))
        else:
            # Shift: drop the executed stage, duplicate the last.
            X0 = np.vstack([self._Xopt[1:], self._Xopt[-1:]])
            U0 = np.vstack([self._Uopt[1:], self._Uopt[-1:]])


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

        # Unpack states/controls from the flat solution.
        Xopt = w_opt[:self.n_x].reshape(Nh + 1, nx)
        Uopt = w_opt[self.n_x:self.n_x + self.n_u].reshape(Nh, nu)

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

        max_sst = 0.0
        min_pred_stat = np.inf
        if self._spos_used is not None:
            sp = self._spos_used[self._spos_used[:, 0] < 1e5]     # drop far-parked pads
            if len(sp):
                dxy = (Xopt[1:, None, :2] - sp[None, :, :2])       # (Nh, K, 2)
                d   = np.hypot(dxy[..., 0], dxy[..., 1])
                # Clearance/violation are to the circle SURFACE: keep-out = d_safe_hard + r.
                r_keep = self.d_safe_hard + sp[None, :, 2]       # (1, K)
                min_pred_stat = float((d - sp[None, :, 2]).min())  # nearest surface clearance
                max_sst = float(np.max(np.maximum(0.0, r_keep ** 2 - d ** 2)))

        info = {"solve_ms": dt_solve * 1e3, "success": bool(ok and finite),
                "fallback": used_fallback,
                "iter": stats.get("iter_count", -1),
                "status": stats.get("return_status", "?"),
                "max_sst": max_sst, "min_pred_stat": min_pred_stat}
        return np.array([a_cmd, delta_cmd]), info


# ── Main control loop (MPC) ───────────────────────────────────────────────────

def run(unity_exec_path=None, port=5004, run_sysid=False, noise_std=0.0,
        detect_range=float("inf"),
        d_infl=D_INFL, d_safe=D_SAFE, horizon=N_MPC,
        lidar_costmap=False, lidar_topic="/point_cloud", save_traj=None,
        occlusion_aware=False, show_occlusion_plot=True,
        dynamic_obstacles=False):
    # The dynamic-obstacle ORACLE is gone from the observation vector, so the P_opos /
    # P_ovel slots are fed from a SENSED estimate instead: the same LiDAR cluster tracker
    # MPPI uses (dynamic_clusters.DynamicClusterTracker), whose confirmed movers give a
    # centroid and a Kalman velocity per track. That is the port the old SystemExit here
    # asked for. Without --dynamic-obstacles the slots are simply empty (k_obs = 0) and
    # the MPC plans against static + occlusion keep-outs only.
    print(f"[MPC] Connecting to Unity on port {port} ...")
    print(f"[MPC] Horizon N={horizon} ({horizon * DT:.1f} s)  "
          f"D_INFL={d_infl:.1f} m  D_SAFE={d_safe:.1f} m")
    if detect_range < float("inf"):
        print("[MPC] --detect-range no longer applies (it masked the removed oracle "
              "obstacle slots; LiDAR range is set by scan.max_range) — IGNORED.")
    if noise_std > 0.0:
        print("[MPC] --noise-std no longer applies (it corrupted the removed oracle "
              "obstacle slots) — IGNORED.")

    costmap  = None
    k_static = 0
    k_occ    = 0
    dyn_tracker    = None
    dyn_last_stamp = None
    k_obs_eff      = 0          # obstacle slots actually built — 0 unless clusters are on
    if lidar_costmap:
        from obstacle_circles import ObstacleCircles
        cm = ObstacleCircles(max_age=1.5)
        if cm.start(topic=lidar_topic):
            costmap  = cm
            k_static = K_STATIC
            print(f"[MPC] LiDAR map     : ON  (topic={lidar_topic}) — static keep-out "
                  f"K_STATIC={K_STATIC}, D_SAFE_HARD={D_SAFE_HARD:.1f} m")
            if occlusion_aware:
                k_occ = K_OCC
                # Declare the scan geometry so range-jump boundary SEGMENTS (capsules)
                # are available.
                cm.configure_scan(SCAN_FOV_H, SCAN_FOV_V, SCAN_RES_H, SCAN_RES_V,
                                  SCAN_MAX_RANGE)
                print(f"[MPC] Occlusion-aware: ON  — forward reachable sets K_OCC={K_OCC}, "
                      f"v_target={V_TARGET:.1f} m/s, D_SAFE_HARD={D_SAFE_HARD:.1f} m "
                      f"(expands +{V_TARGET*horizon*DT:.1f} m over the horizon); "
                      f"{'capsules' if OCC_USE_CAPSULES else 'circles'} from range-jump corners "
                      f"({SCAN_FOV_H:g}°x{SCAN_FOV_V:g}° @ {SCAN_RES_H:g}°)")
            if dynamic_obstacles:
                # Sensed movers replace the removed oracle. Same tracker and same config
                # section MPPI uses, so a taxiing aircraft is modelled identically by both
                # planners; only the consumer differs (NLP slots here, rollout cost there).
                from dynamic_clusters import DynamicClusterTracker
                dyn_tracker = DynamicClusterTracker(
                    assoc_radius=tc.DYN_ASSOC, q_accel=tc.DYN_Q_ACCEL,
                    r_frac=tc.DYN_R_FRAC, r_min=tc.DYN_R_MIN,
                    sigma_v0=tc.DYN_SIGMA_V0, extent_alpha=tc.DYN_EXT_ALPHA,
                    ttl=tc.DYN_TTL, min_hits=tc.DYN_MIN_HITS, v_min=tc.DYN_V_MIN,
                    min_dyn_hits=tc.DYN_MIN_DYN_HITS,
                    require_motion=tc.DYN_REQUIRE_MOTION,
                    dyn_window=tc.DYN_WINDOW, extent_frac=tc.DYN_EXTENT_FRAC,
                    resegment_ratio=tc.DYN_RESEG_RATIO)
                k_obs_eff = K_OBS
                print(f"[MPC] Dynamic obst. : ON  — sensed LiDAR clusters, K_OBS={K_OBS}, "
                      f"D_SAFE={d_safe:.1f} m")
        else:
            print("[MPC] LiDAR map     : requested but unavailable (no rclpy) — "
                  "running WITHOUT static keep-out")
    elif occlusion_aware:
        print("[MPC] Occlusion-aware: requested but needs --lidar-costmap — DISABLED")
    if dynamic_obstacles and dyn_tracker is None:
        print("[MPC] Dynamic obst. : requested but needs --lidar-costmap — DISABLED")

    mpc = TaxiMPC(N=horizon, k_obs=k_obs_eff, d_infl=d_infl, d_safe=d_safe,
                  k_static=k_static, k_occ=k_occ,
                  occ_d_target=(V_TARGET * OCC_D_TARGET_HORIZON
                                if OCC_SINGLE_CIRCLE else None))

    env = UnityEnvironment(
        file_name=unity_exec_path,
        base_port=port,
        seed=42,
        no_graphics=unity_exec_path is not None,
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

    unity_stopped = False
    min_h     = np.inf
    min_dist  = np.inf
    collided  = False
    reached   = False
    traj      = []
    obs_track = []
    # Occlusion-boundary segments actually constrained by the MPC, keyed on TRACK ID (a
    # mouth that sweeps as the ego moves would otherwise paint a whole line across many
    # steps even when each step is corner-only).
    seg_track = {}
    corner_tracker = OcclusionCornerTracker(
        assoc_radius=OCC_TRACK_ASSOC, alpha=OCC_TRACK_ALPHA,
        ttl=OCC_TRACK_TTL, min_hits=OCC_TRACK_HITS)
    # Union of every OCC cell seen across the run. The costmap is a rolling, DECAYING,
    # ego-centered grid (occ_ttl≈12s), so a single end-of-run snapshot only shows walls
    # still in view at the goal — walls the ego passed earlier have decayed/rolled out.
    # Accumulate (dedup by cell) to show the real extent.
    occ_seen = {}

    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    ep_steps     = 0
    delta_actual = 0.0
    accel_actual = 0.0
    u_prev       = np.zeros(2)
    episode_done = False
    solve_ms_acc = 0.0
    solve_fail   = 0

    while not episode_done:
        if len(terminal_steps) > 0:
            collided = min_h < 0.0
            reached  = not terminal_steps.interrupted[0] and not collided
            episode_done = True
            break

        if len(decision_steps) == 0:
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            continue

        obs = decision_steps.obs[0][0]
        ep_steps += 1

        # No inject_sensor_noise: it corrupted the ORACLE obstacle slots, which are gone,
        # so it was deleted from taxi_controller_mppi along with them (--noise-std is
        # reported as inert at startup, mirroring MPPI). Ego and goal were always clean.
        # obs_to_state returns (s, goal_xy) — the obstacle slots it used to unpack were
        # removed with the oracle. Movers now come from the cluster tracker below.
        s, goal_xy = obs_to_state(obs, delta_actual, accel_actual)
        obstacles = []

        # ── Refresh the LiDAR static-obstacle costmap and extract OCC points ──
        static_pts = None
        occ_pts    = None
        occ_segs   = None
        n_static   = 0
        n_occ      = 0
        n_dyn      = 0
        if costmap is not None:
            ego_fwd = np.array([np.cos(s[2]), np.sin(s[2])])
            costmap.update(ego_fwd)
            if costmap.ready:
                static_pts = costmap.circles()               # (M,3) world [a0, a1, r]
                if static_pts is not None and len(static_pts):
                    n_static = len(static_pts)
                    if save_traj is not None:
                        # Dedup by rounded world centre so the union stays bounded.
                        for wx, wy, _r in static_pts:
                            occ_seen[(round(float(wx), 1), round(float(wy), 1))] = None
                if k_occ:
                    # Range-jump boundary SEGMENTS (this scan) seed the expanding
                    # keep-outs. They need configure_scan().
                    occ_segs = costmap.occlusion_segments(
                        ego_fwd=ego_fwd,
                        fwd_half_angle_deg=OCC_FWD_HALF_ANGLE)
                    # Associate + smooth against previous scans so the same physical
                    # corner keeps one stable centre instead of a fresh jittered one.
                    occ_segs = corner_tracker.update(occ_segs, time.monotonic())
                    if occ_segs is not None and not OCC_USE_CAPSULES:
                        # Collapse each boundary line onto its CORNER, so the keep-out
                        # is an expanding CIRCLE about that point. Done here, at the
                        # source, so the constraints, the recorded seg_track and the
                        # episode plot all show the same geometry.
                        occ_segs = occ_segs.copy()
                        occ_segs[:, 1, :] = occ_segs[:, 0, :]
                    n_occ    = (len(occ_segs) if occ_segs is not None else 0)

                # ── Sensed movers → the P_opos / P_ovel slots ──────────────
                # Correction only on a FRESH scan stamp (clusters() is memoised on it, so
                # re-fusing the same centroid every control step would collapse the filter
                # covariance and drag the velocity to zero); pure prediction otherwise.
                # This mirrors taxi_controller_mppi.run() — see the comment there.
                if dyn_tracker is not None:
                    _now = time.monotonic()
                    if costmap.stamp != dyn_last_stamp:
                        dyn_last_stamp = costmap.stamp
                        dyn_tracker.update(
                            costmap.clusters(cell=tc.DYN_CELL,
                                             min_points=tc.DYN_MIN_POINTS,
                                             max_radius=tc.DYN_MAX_RADIUS), _now)
                    else:
                        dyn_tracker.predict_to(_now)
                    dyn_now = dyn_tracker.dynamic(_now)
                    if dyn_now is not None and len(dyn_now):
                        vels  = dyn_tracker.velocities()
                        order = np.argsort(np.hypot(dyn_now[:, 0] - s[0],
                                                    dyn_now[:, 1] - s[1]))
                        for j in order[:k_obs_eff]:
                            c0, c1 = float(dyn_now[j, 0]), float(dyn_now[j, 1])
                            if vels is None:
                                vx, vy = 0.0, 0.0
                            else:
                                vx, vy = float(vels[j][0]), float(vels[j][1])
                            # _pack_params wants EGO-RELATIVE positions; it adds s0 back.
                            obstacles.append(((c0 - s[0], c1 - s[1]), (vx, vy)))
                    n_dyn = len(obstacles)

        # ── MPC solve ─────────────────────────────────────────────────────
        # MPC state is [x, y, theta, v, accel]; delta_actual is not a state
        # (the steering-rate limit is a constraint), so it isn't passed.
        s_mpc = np.array([s[0], s[1], s[2], s[3], accel_actual])
        u_cmd, info = mpc.solve(s_mpc, goal_xy, obstacles, u_prev, static_pts,
                                occ_pts, occ_segs)
        solve_ms_acc += info["solve_ms"]
        solve_fail   += (0 if info["success"] else 1)

        a_cmd     = float(np.clip(u_cmd[0], A_MIN, A_MAX))
        delta_cmd = float(np.clip(u_cmd[1], -DELTA_LIM, DELTA_LIM))
        u_prev    = np.array([a_cmd, delta_cmd])

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

        # Barrier value h: clearance to the nearest hazard, negative ⇒ inside a keep-out
        # (h < 0 at any step is what marks the episode a collision). This used to read
        # obs[17], a slot of the removed dynamic-obstacle ORACLE — the observation vector
        # is 7 long now, so that indexed past the end. Recomputed from what the ego
        # actually SENSES: the sensed movers against d_safe and the LiDAR circles against
        # d_safe_hard (near_st is centre-to-centre, so subtract the circle radius too).
        h_obs = near - d_safe
        h_st  = np.inf
        if static_pts is not None and len(static_pts):
            h_st = float(np.min(np.hypot(static_pts[:, 0] - s[0],
                                         static_pts[:, 1] - s[1]) - static_pts[:, 2])
                         ) - D_SAFE_HARD
        h_val = float(min(h_obs, h_st))
        dist  = float(min((np.hypot(rel[0], rel[1]) for rel, _ in obstacles),
                          default=np.inf))
        min_h    = min(min_h, h_val)
        min_dist = min(min_dist, dist)

        traj.append((ep_steps * DT, s[0], s[1], s[2], s[3], a_cmd, delta_cmd))
        for rel, _ov in obstacles:
            obs_track.append((ep_steps * DT, s[0] + rel[0], s[1] + rel[1]))

        if k_occ and occ_segs is not None and len(occ_segs):

            tids = corner_tracker.ids()
            ego_xy = np.array([s_mpc[0], s_mpc[1]])
            for ti, seg_i in enumerate(occ_segs):
                if ti >= len(tids):
                    break
                d = point_segment_distance_np(ego_xy[None, :], seg_i[0], seg_i[1])[0]
                if d <= OCC_QUERY_R:
                    seg_track.setdefault(tids[ti],
                                         (seg_i[0][0], seg_i[0][1], seg_i[1][0],
                                          seg_i[1][1], s[0], s[1], ep_steps * DT))

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
        print("\n[MPC] Unity stopped early (Editor left Play mode, or the app closed).")
    try:
        env.close()
    except Exception:
        pass

    verdict = "collision" if collided else ("reached" if reached else "timeout")

    if save_traj is not None and traj:
        # Overlay the UNION of static occluders seen across the whole run (not the
        # decaying end-of-run snapshot) so the plot shows every wall the ego actually
        # reacted to, including ones it has since driven past.
        occ_pts = np.array(list(occ_seen.keys())) if occ_seen else None
        if seg_track:
            _vals = np.array(list(seg_track.values()), dtype=float)
            occ_segments = _vals[:, :4].reshape(-1, 2, 2)
            occ_ego = _vals[:, 4:]
        else:
            occ_segments = occ_ego = None
        _save_trajectory(save_traj, verdict, goal_xy, np.asarray(traj),
                         np.asarray(obs_track) if obs_track else None, occ_pts,
                         occ_segments=occ_segments, occ_ego=occ_ego,
                         # OCC_HORIZON (shared with MPPI), not the planning horizon,
                         # so the drawn radius always equals the enforced one.
                         capsule_horizon=(OCC_D_TARGET_HORIZON if OCC_SINGLE_CIRCLE
                                          else OCC_HORIZON),
                         show_occlusion=show_occlusion_plot,
                         single_radius=OCC_SINGLE_CIRCLE)

    mean_ms = solve_ms_acc / max(ep_steps, 1)
    print(f"\n[MPC] steps={ep_steps:4d}  min_dist={min_dist:5.2f} m (target >= {d_safe:.1f} m)  "
          f"min_h={min_h:8.2f} (target >= 0)  solve≈{mean_ms:4.1f}ms  fails={solve_fail}  "
          f"→ {verdict.upper()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MPC (CasADi/IPOPT) taxiing controller.")
    p.add_argument("--config",         default=None,
                   help="Path to a tuning YAML (default: config.yaml next to this script). "
                        "Applied at IMPORT time by taxi_config, not here — this entry exists "
                        "so the flag appears in --help. $TAXI_CONFIG does the same.")
    p.add_argument("--exec",           default=None)
    p.add_argument("--port",           default=5004, type=int)
    p.add_argument("--sysid",          default=False, type=lambda x: x.lower() == "true")
    p.add_argument("--noise-std",      default=0.0, type=float,
                   help="Std-dev of Gaussian noise injected into obstacle obs [m]. 0=off.")
    p.add_argument("--detect-range",   default=float("inf"), type=float,
                   help="Euclidean detection range [m]; obstacles beyond are masked. Default=inf.")
    p.add_argument("--horizon",        default=N_MPC, type=int,
                   help=f"MPC prediction horizon in steps (default {N_MPC}; {N_MPC*DT:.1f} s).")
    p.add_argument("--d-infl",         default=D_INFL, type=float,
                   help="Obstacle soft-influence radius [m]. Must stay >= --d-safe.")
    p.add_argument("--d-safe",         default=D_SAFE, type=float,
                   help="Hard keep-out radius [m] (steep penalty inside). Must stay <= --d-infl.")
    p.add_argument("--lidar-costmap",  action="store_true",
                   help="Down-sample the published PointCloud2 each step and cover the static "
                        "obstacles with circles; the K_STATIC nearest are added to the MPC as "
                        "keep-out constraints ‖p_ego−c‖ ≥ D_SAFE_HARD + r. Needs ROS 2 sourced "
                        "(rclpy) and the ros_tcp_endpoint running. WITHOUT this the MPC only avoids "
                        "the dynamic obstacles in the observation vector.")
    p.add_argument("--lidar-topic",    default="/point_cloud",
                   help="PointCloud2 topic for the LiDAR map. Default: /point_cloud.")
    p.add_argument("--dynamic-obstacles", action="store_true",
                   help="Cluster + track MOVING LiDAR returns and feed the K_OBS nearest "
                        "confirmed movers into the MPC's obstacle slots as a keep-out "
                        "‖p_ego − p_obs(t)‖ ≥ D_SAFE on a constant-velocity prediction. "
                        "Replaces the removed oracle; needs --lidar-costmap.")
    p.add_argument("--no-occlusion-plot", action="store_true",
                   help="omit the occlusion keep-out circles, corner centres and "
                        "ego-at-detection markers from the saved trajectory plots")
    p.add_argument("--occlusion-aware", action="store_true",
                   help="Add occlusion-aware forward-reachable-set keep-outs (Firoozi et al.): "
                        "the K_OCC nearest blind-spot cells behind occluders each seed an "
                        "EXPANDING keep-out circle (radius grows as v_target·t over the horizon), "
                        "so the ego gives occluded corners a wider berth. Requires --lidar-costmap.")
    p.add_argument("--save-traj",      default=None, metavar="DIR",
                   help="Save the run's ego trajectory as CSV + a top-down PNG plot into DIR.")
    args = p.parse_args()

    if args.d_infl < args.d_safe:
        p.error(f"--d-infl ({args.d_infl}) must be >= --d-safe ({args.d_safe}).")

    run(unity_exec_path=args.exec if args.exec != "None" else None,
        port=args.port,
        run_sysid=args.sysid,
        noise_std=args.noise_std,
        detect_range=args.detect_range,
        d_infl=args.d_infl,
        d_safe=args.d_safe,
        horizon=args.horizon,
        lidar_costmap=args.lidar_costmap,
        lidar_topic=args.lidar_topic,
        save_traj=args.save_traj,
        occlusion_aware=args.occlusion_aware,
        show_occlusion_plot=not args.no_occlusion_plot,
        dynamic_obstacles=args.dynamic_obstacles)
