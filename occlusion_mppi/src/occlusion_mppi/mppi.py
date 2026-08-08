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
        self.w_hard = 50.0
        self.w_soft = 50.0
        self.d_infl = 1.0       # [m]

        # Collision: binary "is this rollout pose in an occupied voxel". Independent
        # of the occlusion term, so the two can be switched on and off separately.
        self.w_collision = 1.0e6
        self.use_occlusion = True

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

    def plan(self, pos, vel, goal, boundaries, occupancy=None):
        """One control step.

        occupancy : OccupancySet or None. Without it nothing stops a rollout going
        through a wall -- the occlusion boundaries are the shadow mouth, not the
        obstacle surface.

        Returns (action, info) where action is the acceleration to apply now and
        info carries the rollout batch for visualisation / debugging.
        """
        c = self.cfg
        d = self.plant.dim
        K, H = c.samples, c.horizon
        t_start = time.perf_counter()

        noise = self.rng.normal(0.0, c.sigma_a, size=(K, H, d))
        actions = self.nominal[None, :, :] + noise

        s0 = self.plant.initial_state(pos, vel, K)
        traj = self.plant.rollout(s0, actions)          # (K, H, 2d)

        pos_t = traj[:, :, :d]
        vel_t = traj[:, :, d:]
        goal = np.asarray(goal, dtype=float)[:d]

        cost = np.zeros(K)
        collided = np.zeros(K, dtype=bool)
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
            cost += c.w_v * np.sum(v ** 2, axis=1) * c.dt

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


            #TODO add constrain on the actions proposed and removed the cost here, it needs this handled only the ground is not correclty occupied
            # Altitude bounds. Binary and charged at w_hard for the same reason
            # collision is: this is a constraint, and depth of violation is not a
            # useful gradient -- the sampled rollouts supply the gradient by
            # simply staying inside.
            if d > 2 and (c.z_min is not None or c.z_max is not None):
                z = p[:, 2]
                out = np.zeros(len(z), dtype=bool)
                if c.z_min is not None:
                    out |= z < c.z_min
                if c.z_max is not None:
                    out |= z > c.z_max
                cost += c.w_hard * out

            if c.use_occlusion:
                dist = boundaries.distance(p)
                cost += occlusion_stage_cost(
                    d=dist, v=speed, t_k=t_k,
                    v_target=c.v_target, d_safe=c.d_safe, w_obs=c.w_hard,
                    t_grow_max=c.t_grow_max, w_soft=c.w_soft, d_infl=c.d_infl,
                )


        cost += c.w_goal_term * np.linalg.norm(pos_t[:, -1, :] - goal, axis=1)
    
        # MPPI weighting. Subtracting the min before exp is what keeps this from
        # underflowing to all-zero when every rollout is infeasible (cost ~1e6).
        beta = cost.min()
        w = np.exp(-(cost - beta) / c.lam)
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            w = np.ones(K) / K
        else:
            w = w / w_sum

        self.nominal = np.einsum("k,khd->hd", w, actions)
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
            "best": traj[int(np.argmin(cost))],
            "frac_infeasible": float(np.mean(cost >= c.w_hard * c.dt)),
            "frac_collide": float(np.mean(collided)),
        }
        return action, info
