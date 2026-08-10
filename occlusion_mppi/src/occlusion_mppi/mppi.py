"""MPPI planner with an occlusion keep-out term.

The occlusion cost is imported unchanged from cost.py, which is itself a verbatim
copy of the 2D controller's. Everything novel is in how `d` is obtained
(boundary.BoundarySet) and in the plant (dynamics.DoubleIntegrator).
"""

import time

import numpy as np

from .cost import occlusion_stage_cost


class MPPIConfig:
    def __init__(self, **kw):
        self.horizon = 60
        self.samples = 6000
        self.lam = 1.0          # temperature
        self.sigma_a = 1.5      # action noise std [m/s^2]

        # How the K sampled action sequences are collapsed into one command.
        #
        #   "mean" -- textbook MPPI: the softmax-weighted average of all K.
        #   "best" -- the single lowest-cost rollout (argmin), i.e. random
        #             shooting / MPPI with a zero-temperature limit.
        #
        # "best" exists because the weighted mean is an average of trajectories,
        # not itself a trajectory. At a corner the rollouts split into two modes
        # -- around the left of the occluder and around the right -- both cheap,
        # and their mean points straight between them, into the obstacle neither
        # mode chose. The published "best" marker was visibly clearing the corner
        # while the ego, following the mean, drove into the wall.
        #
        # The trade is real: argmin discards the variance reduction that is the
        # whole point of averaging, so consecutive ticks may pick samples from
        # unrelated parts of the distribution and the command jitters. That
        # jitter is bounded by nominal warm-starting and by sigma_a, but it is
        # not zero. Prefer "mean" again once the modes stop being symmetric
        # (a geodesic goal cost does that); until then a coherent wrong plan is
        # worse than an incoherent right one.
        self.selection = "mean"

        self.dt = 0.1

        # Goal / regulation weights
        self.w_goal = 20.0
        self.w_goal_term = 10.0
        self.w_ctrl = 0.05
        self.w_v = 1.2

        # Occlusion keep-out. THESE ARE NOT THE 2D CONTROLLER'S VALUES.
        # config.yaml there uses d_safe_hard=15 m and v_target=10 m/s, sized for a
        # road vehicle. Applied to a quadrotor in a 36 m room that is a keep-out of
        # 15 + 10*8 = 95 m, which covers the entire world and makes every rollout
        # infeasible. Rescaled here to drone/indoor scale; the FUNCTION is identical,
        # only its parameters are not.
        self.d_safe = 1.0       # [m]
        self.v_target = 1.5     # [m/s] assumed speed of a hidden agent
        self.t_grow_max = 3.0   # [s] cap on keep-out growth
        self.w_soft = 50.0
        self.d_infl = 1.0       # [m]

        # Collision: binary "is this rollout pose in an occupied voxel". Independent
        # of the occlusion term, so the two can be switched on and off separately.
        self.w_collision = 1.0e6
        self.use_occlusion = True

        # Obstacle clearance keep-out. A margin around the OCCUPIED voxels, i.e.
        # around real matter -- NOT around the occlusion frontier, which is empty
        # space at the mouth of a shadow and is handled by d_safe/v_target above.
        # The two are separate sets: a fully observed wall has no frontier near
        # its face, and a doorway has a frontier with nothing solid behind it.
        #
        # Distance is measured from the surface of the sphere CIRCUMSCRIBING the
        # nearest voxel, so the voxel's own extent is never mistaken for
        # clearance. Cost ramps linearly from 0 at d_clear to w_clear at the
        # surface, and keeps climbing inside it -- that continuation is what lets
        # a rollout leaving an occupied cell score better than one driving
        # deeper, which the binary w_collision term cannot express.
        self.w_clear = 200.0    # cost/stage for sitting exactly on the surface
        self.d_clear = 1.5      # [m] keep-out measured OUTSIDE the voxel sphere

        # Altitude bounds [m], 3D only. None disables.
        #
        # NOT redundant with the collision term. That term tests membership of the
        # published occupancy cloud, and ROG-Map's virtual ground/ceiling are
        # z-comparisons inside isOccupied() -- they are never written to the
        # occupancy buffer, so boxSearch never emits them and inf_occ contains no
        # floor. With a wall-only scene there is then nothing at all below the
        # wall's bottom edge, and diving under the occluder is a collision-free,
        # occlusion-free way around it. That is exactly what MPPI finds.
        self.z_min = None
        self.z_max = None

        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"unknown MPPI config key: {k}")
            setattr(self, k, v)


class OcclusionMPPI:
    def __init__(self, plant, cfg=None, rng=None):
        self.plant = plant
        self.cfg = cfg or MPPIConfig()
        self.rng = rng or np.random.default_rng(0)
        self.nominal = np.zeros((self.cfg.horizon, plant.dim))

    def plan(self, pos, vel, goal, boundaries, occupancy=None, clearance=None):
        """One control step.

        occupancy : OccupancySet or None. Without it nothing stops a rollout going
        through a wall -- the occlusion boundaries are the shadow mouth, not the
        obstacle surface.

        clearance : OccupancySet used for the w_clear keep-out, or None to reuse
        `occupancy`. They are separate because the two terms want different
        geometry. inside_segment() must see the floor -- flying into the ground
        is a collision. nearest() must NOT: it returns only THE nearest voxel, so
        with the ground 1 m below a drone at cruise_z every query returns ~1 m
        and no wall further than the flight altitude is ever visible to the
        keep-out. Pass a z-banded set here and the term regains its full reach.

        Returns (action, info) where action is the acceleration to apply now and
        info carries the rollout batch for visualisation / debugging.
        """
        c = self.cfg
        d = self.plant.dim
        K, H = c.samples, c.horizon
        t_start = time.perf_counter()

        clear = clearance if clearance is not None else occupancy

        noise = self.rng.normal(0.0, c.sigma_a, size=(K, H, d))
        actions = self.nominal[None, :, :] + noise
        # Sample only in the plane the plant can actually move in. The plant
        # ignores a_z anyway, but leaving it sampled would charge w_ctrl for
        # vertical effort that never happens and would return a nonzero a_z for
        # the node to integrate into the setpoint.
        if getattr(self.plant, "lock_z", False):
            actions[:, :, 2] = 0.0

        s0 = self.plant.initial_state(pos, vel, K)
        traj = self.plant.rollout(s0, actions)          # (K, H, 2d)

        pos_t = traj[:, :, :d]
        vel_t = traj[:, :, d:]
        goal = np.asarray(goal, dtype=float)[:d]

        # Surface distance for every (rollout, stage) pose, in ONE tree query
        # rather than H queries of K points inside the loop. Same work for the
        # tree, a thirtieth of the per-call overhead, and it parallelises -- the
        # per-stage version pushed solve time from ~3 to ~17 us/rollout-step and
        # blew through the control period.
        d_surf = None
        if clear is not None and c.w_clear and len(clear):
            # Half the voxel's space diagonal: the radius of the sphere that
            # CIRCUMSCRIBES it. Inscribing would claim clearance in the corners
            # the voxel actually occupies; circumscribing errs safe.
            r_voxel = 0.5 * np.sqrt(2 if clear.planar else 3) * clear.resolution
            d_ctr, _ = clear.nearest(pos_t.reshape(-1, d))
            d_surf = d_ctr.reshape(K, H) - r_voxel

        cost = np.zeros(K)
        collided = np.zeros(K, dtype=bool)
        # Breached the obstacle keep-out (as opposed to `breached`, which is the
        # occlusion one). Reported separately so a run can be read as "too close
        # to a wall" vs "too close to a shadow" without guessing.
        too_close = np.zeros(K, dtype=bool)
        # A rollout is infeasible if ANY stage breached a hard keep-out. Tracked
        # separately from `cost`: the goal term alone sums to hundreds over the
        # horizon, so comparing the total against w_hard reads 1.00 always and
        # says nothing.
        breached = np.zeros(K, dtype=bool)
        for k in range(H):
            t_k = (k + 1) * c.dt
            p = pos_t[:, k, :]
            v = vel_t[:, k, :]
            speed = np.linalg.norm(v, axis=1)

            cost += c.w_goal * np.linalg.norm(p - goal, axis=1) * c.dt
            cost += c.w_ctrl * np.sum(actions[:, k, :] ** 2, axis=1) * c.dt
            # Velocity damping: near the goal the position gradient flattens and
            # the weighted mean degenerates to averaged noise; this keeps the
            # low-cost rollouts the slow ones so the ego settles instead of
            # dithering around the goal.
            #cost += c.w_v * np.sum(v ** 2, axis=1) * c.dt

            # Collision: charged once per stage whose SEGMENT crosses an occupied
            # cell. Swept rather than point-sampled, because v*dt is the same order
            # as the voxel pitch and a point test steps clean over a thin wall.
            # Binary on purpose -- penetration depth is not a useful gradient, and
            # the sampled rollouts supply the gradient by simply missing the wall.
            if occupancy is not None and c.w_collision:
                p_prev = pos_t[:, k - 1, :] if k else np.broadcast_to(
                    np.asarray(pos, dtype=float)[:d], p.shape)
                hit = occupancy.inside_segment(p_prev, p)
                cost += c.w_collision * hit
                collided |= hit

            # if d_surf is not None:
            #     margin = np.maximum(0.0, c.d_clear - d_surf[:, k])
            #     cost += c.w_clear * (margin / c.d_clear) * c.dt
            #     too_close |= margin > 0.0

            # if d > 2 and (c.z_min is not None or c.z_max is not None):
            #     z = p[:, 2]
            #     out = np.zeros(len(z), dtype=bool)
            #     if c.z_min is not None:
            #         out |= z < c.z_min
            #     if c.z_max is not None:
            #         out |= z > c.z_max
            #     cost += c.w_hard * out

            if c.use_occlusion:
                dist = boundaries.distance(p)
                t_eff = min(t_k, c.t_grow_max) if c.t_grow_max is not None else t_k
                breached |= dist < c.d_safe + c.v_target * t_eff
                cost += occlusion_stage_cost(
                    d=dist, v=speed, t_k=t_k,
                    v_target=c.v_target, d_safe=c.d_safe, w_obs=c.w_collision,
                    t_grow_max=c.t_grow_max, w_soft=c.w_soft, d_infl=c.d_infl,
                )


        cost += c.w_goal_term * np.linalg.norm(pos_t[:, -1, :] - goal, axis=1)
    
        # MPPI weighting. Subtracting the min before exp is what keeps this from
        # underflowing to all-zero when every rollout is infeasible (cost ~1e6).
        beta = cost.min()
        w = np.exp(-(cost - beta) / c.lam)
        w_sum = w.sum()

        w = w / w_sum

        # Weights are computed either way: `ess` is the diagnostic that says how
        # multimodal the batch was, and it is exactly as informative under "best"
        # as under "mean" -- more so, since it is the number that justifies not
        # trusting the mean in the first place.
        k_best = int(np.argmin(cost))
        if c.selection == "best":
            self.nominal = actions[k_best].copy()
        elif c.selection == "mean":
            self.nominal = np.einsum("k,khd->hd", w, actions)
        else:
            raise ValueError("selection must be 'mean' or 'best', got %r"
                             % (c.selection,))
        action = self.nominal[0].copy()

        # Shift the nominal for the next step (receding horizon).
        self.nominal = np.roll(self.nominal, -1, axis=0)
        self.nominal[-1] = 0.0

        solve_s = time.perf_counter() - t_start
        info = {
            "traj": traj, "cost": cost, "weights": w,
            "solve_s": solve_s,
            "solve_hz": 1.0 / solve_s if solve_s > 0 else float("inf"),
            # Cost of one (rollout, stage) pair -- the unit that scales with K*H.
            "us_per_rollout_step": solve_s * 1e6 / (K * H),
            # Under selection="best" this marker IS the executed plan, so the
            # RViz curve and the ego's behaviour finally describe the same thing.
            "best": traj[k_best],
            "selection": c.selection,
            "frac_infeasible": float(np.mean(breached | collided)),
            "frac_collide": float(np.mean(collided)),
            "frac_too_close": float(np.mean(too_close)),
            "ess": float(1.0 / np.sum(w ** 2)),   # effective sample size, out of K
        }
        return action, info
