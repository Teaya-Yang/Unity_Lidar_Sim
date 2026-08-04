"""Rollout dynamics for the MARSIM quadrotor.

DELIBERATELY SIMPLER THAN THE TAXI
----------------------------------
The 2D controller rolled out a bicycle model (`_rollout_step` in
taxi_controller_mppi.py) because a car cannot move sideways: heading and
velocity are coupled through the steering angle.

MARSIM's `perfect_drone` consumes quadrotor_msgs/PositionCommand and tracks the
commanded velocity essentially perfectly, so the plant here is a
velocity-commanded double integrator with no heading constraint. The action is
an acceleration; the state is (pos, vel).

That is not a simplification for convenience -- it is what the plant actually
is. Rolling out a bicycle model against a holonomic plant would make every
rollout wrong in the same direction.
"""

import numpy as np


class DoubleIntegrator:
    """Batched double-integrator rollout.

    State  : (K, 2*dim)  -> [pos, vel]
    Action : (K, dim)    -> acceleration command [m/s^2]
    """

    def __init__(self, dim=2, dt=0.1, v_max=3.0, a_max=4.0):
        self.dim = dim
        self.dt = dt
        self.v_max = v_max
        self.a_max = a_max

    def initial_state(self, pos, vel, K):
        s = np.zeros((K, 2 * self.dim))
        s[:, :self.dim] = np.asarray(pos, dtype=float)[:self.dim]
        s[:, self.dim:] = np.asarray(vel, dtype=float)[:self.dim]
        return s

    def step(self, state, action):
        d = self.dim
        a = np.clip(action, -self.a_max, self.a_max)

        vel = state[:, d:] + a * self.dt
        # Clip SPEED, not per-axis velocity: per-axis clipping would silently bias
        # diagonal motion to be faster than axis-aligned motion by sqrt(2).
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        over = speed > self.v_max
        vel = np.where(over, vel * (self.v_max / np.maximum(speed, 1e-9)), vel)

        pos = state[:, :d] + vel * self.dt

        out = np.empty_like(state)
        out[:, :d] = pos
        out[:, d:] = vel
        return out

    def rollout(self, s0, actions):
        """actions : (K, H, dim). Returns (K, H, 2*dim) states, one per stage."""
        K, H, _ = actions.shape
        traj = np.empty((K, H, 2 * self.dim))
        st = s0.copy()
        for k in range(H):
            st = self.step(st, actions[:, k, :])
            traj[:, k, :] = st
        return traj
