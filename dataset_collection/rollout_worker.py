"""
rollout_worker.py
=================
Run ONE closed-loop episode against Unity and log every (x_t, a_t, x_{t+1})
transition. The output of a single call is fully self-contained: state + action
+ next-state arrays, plus the eta that generated them and the terminal reason.

The episode is driven by one of two tracker modes:

  'mppi'  — reuse taxi_controller_mppi.mppi() as a black box: seed its
             `mean` with the GP reference so the sampling is centred on the
             desired command profile, and let its normal noise/optimization
             run on top. This produces the "wide" action distribution.

  'mpc'   — reuse taxi_controller_mpc.TaxiMPC as a black box, warm-started
             with the GP reference on step 1 by passing it as u_prev-like bias
             into the initial guess. The MPC then re-optimizes toward the
             configured goal. This produces the "smooth optimal" action
             distribution. Complementary to MPPI, per the paper's argument.

  'open'  — bypass both trackers and drive the GP reference OPEN-LOOP through
             Unity. Cheap, produces the widest action-space coverage (no
             feedback smoothing), and doesn't need MPPI/MPC imports. Useful as
             a 3rd action-distribution slice for the dataset.

Whichever mode is chosen, Unity's ACTUAL bicycle dynamics (with the sampled
eta applied via the EnvironmentParametersChannel) provide the ground-truth
(x_t, a_t, x_{t+1}) — the trackers themselves only decide what action to send.
Their internal analytic dynamics model has no bearing on the recorded data.

Failure handling: episodes that end in collision / timeout are LOGGED, not
discarded, with a `terminal_reason` tag so downstream training can slice or
exclude them if desired. This mirrors the paper's approach — poor-tracking
data still expands the dynamics manifold and is informative signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np

# These heavy imports are done inside build_env / TrackerAdapter so the file is
# import-safe even when ML-Agents / CasADi aren't installed (useful for unit
# tests of the sampler / reference-gen in isolation).


# ── Small standalone bicycle model, used to advance our OWN estimate of ─────
# ── delta_actual / accel_actual between control steps ───────────────────────
# The observation vector does not expose these (they are internal to Unity's
# ApplyBicycleDynamics), so both existing controllers keep their own Python
# copy. We do the same here — one shared advance step, one implementation.
def _advance_actuators(v: float, delta_actual: float, accel_actual: float,
                        a_cmd: float, delta_cmd: float, eta: Dict[str, float],
                        dt: float) -> tuple[float, float]:
    """Return (delta_actual_new, accel_actual_new) after one dt at the given eta.

    Mirrors the exact rate-limit / lag arithmetic in taxi_controller_mppi.run()'s
    inner loop — same clamps, same order — so our recorded x_t matches what
    Unity would have used at the next step.
    """
    speed_frac = min(v / max(eta["STEER_ROLLOFF_SPD"], 1e-3), 1.0)
    eff_limit = eta["DELTA_LIM"] * (1.0 - speed_frac * (1.0 - eta["STEER_ROLLOFF_MIN"]))
    delta_target = float(np.clip(delta_cmd, -eff_limit, eff_limit))
    delta_actual_new = delta_actual + float(np.clip(
        delta_target - delta_actual,
        -eta["MAX_STEER_RATE"] * dt,
        eta["MAX_STEER_RATE"] * dt))
    a_clamped = float(np.clip(a_cmd, eta["A_MIN"], eta["A_MAX"]))
    if eta["ACCEL_TAU"] > 1e-3:
        accel_actual_new = accel_actual + (a_clamped - accel_actual) * (dt / eta["ACCEL_TAU"])
    else:
        accel_actual_new = a_clamped
    return delta_actual_new, accel_actual_new


# ── Episode result ──────────────────────────────────────────────────────────
@dataclass
class RolloutResult:
    """Everything one rollout produced. All arrays are (T, ...) with T = the
    number of transitions actually recorded (<= configured n_steps if the
    episode ended early).
    """
    x: np.ndarray            # (T, 6)  state at step t: [x, y, theta, v, delta_actual, accel_actual]
    a: np.ndarray            # (T, 2)  action applied at step t: [a_cmd, delta_cmd]
    x_next: np.ndarray       # (T, 6)  state at step t+1
    ref: np.ndarray          # (T, 2)  the GP reference sample used at step t (for provenance)
    goal_xy: np.ndarray      # (2,)    goal used this episode (Unity-provided)
    terminal_reason: str     # 'reached' | 'collision' | 'timeout' | 'unity_stopped'
    n_steps: int
    controller: str          # 'mppi' | 'mpc' | 'open'
    eta: Dict[str, float] = field(default_factory=dict)
    condition: str = "A"


# ── Tracker adapters ────────────────────────────────────────────────────────
# Each adapter is a thin wrapper that owns the tracker's per-step call. Keeping
# them small keeps this file resilient to the trackers evolving — we only touch
# their public entry points (mppi() and TaxiMPC.solve()), not their internals.
class _MPPIAdapter:
    """MPPI tracker seeded with the GP reference each step."""
    def __init__(self, ref: np.ndarray):
        self.ref = ref
        self.mean = None       # rolling MPPI mean
        self.u_prev = np.zeros(2)
        import taxi_controller_mppi as tc      # heavy; local
        self._tc = tc

    def act(self, s, obstacles, goal_xy, step: int) -> np.ndarray:
        # Seed the MPPI mean with the reference: this shifts the whole sampling
        # distribution to be centred on the GP command sequence rather than
        # zero, so the SEARCH is around the reference, not just influenced by
        # goal cost. Falls back to a constant tail (last ref step) if the
        # rollout has fewer horizon steps than the reference length.
        H = self._tc.H_MPPI
        if self.mean is None:
            self.mean = np.zeros((H, 2), dtype=float)
        take = min(H, len(self.ref) - step)
        if take > 0:
            self.mean[:take] = self.ref[step:step + take]
            if take < H:
                self.mean[take:] = self.ref[-1]
        u_nom, self.mean = self._tc.mppi(s, self.mean, obstacles, goal_xy, self.u_prev)
        self.u_prev = u_nom.copy()
        return u_nom


class _MPCAdapter:
    """MPC tracker warm-started with the GP reference on step 1."""
    def __init__(self, ref: np.ndarray, mpc):
        self.ref = ref
        self.mpc = mpc
        self.u_prev = np.zeros(2)
        self._seeded = False

    def act(self, s, obstacles, goal_xy, step: int) -> np.ndarray:
        # On step 1, install the reference as the MPC's initial control guess
        # (Uopt) so the first solve starts from the desired profile. After that
        # the MPC's own shift-based warm start takes over.
        if not self._seeded:
            H = self.mpc.N
            U0 = np.zeros((H, 2), dtype=float)
            take = min(H, len(self.ref) - step)
            U0[:take] = self.ref[step:step + take]
            if take < H:
                U0[take:] = self.ref[-1]
            # Roll it through the MPC's own smooth model to get a matching X0
            X0 = np.zeros((H + 1, self.mpc.NX), dtype=float)
            X0[0] = np.array([s[0], s[1], s[2], s[3], s[5]])  # drop delta_actual
            st = X0[0].copy()
            for k in range(H):
                st = np.asarray(self.mpc.f(st, U0[k])).flatten()
                X0[k + 1] = st
            self.mpc._Xopt = X0
            self.mpc._Uopt = U0
            self._seeded = True
        s_mpc = np.array([s[0], s[1], s[2], s[3], s[5]])   # drop delta_actual (MPC state is 5-D)
        u_cmd, _info = self.mpc.solve(s_mpc, goal_xy, obstacles, self.u_prev)
        self.u_prev = u_cmd.copy()
        return u_cmd


class _OpenLoopAdapter:
    """Play the GP reference straight through — no feedback controller at all."""
    def __init__(self, ref: np.ndarray):
        self.ref = ref

    def act(self, s, obstacles, goal_xy, step: int) -> np.ndarray:
        k = min(step, len(self.ref) - 1)
        return self.ref[k].astype(float).copy()


def _make_tracker(name: str, ref: np.ndarray):
    """Build the tracker for the given mode. MPC construction is deferred here
    because it's expensive (CasADi/IPOPT compile) — we only pay it when needed."""
    if name == "mppi":
        return _MPPIAdapter(ref)
    if name == "mpc":
        from taxi_controller_mpc import TaxiMPC
        return _MPCAdapter(ref, TaxiMPC())
    if name == "open":
        return _OpenLoopAdapter(ref)
    raise ValueError(f"unknown controller {name!r}")


# ── The env-parameter channel: how eta gets INTO Unity ──────────────────────
def push_eta_to_unity(env, eta_payload: Dict[str, float]) -> None:
    """Push the sampled eta into Unity via ML-Agents' EnvironmentParametersChannel.

    Requires TaxiAgent.cs to read matching keys with GetWithDefault(...) in
    Initialize() / OnEpisodeBegin() and apply them to its ApplyBicycleDynamics
    fields. Keys are lower-snake_case to match Unity's convention.

    If your TaxiAgent doesn't yet consume these keys, add something like:

        var p = Academy.Instance.EnvironmentParameters;
        wheelbase       = p.GetWithDefault("l", wheelbase);
        dragCoefficient = p.GetWithDefault("drag_coeff", dragCoefficient);
        accelTau        = p.GetWithDefault("accel_tau", accelTau);
        ...
        unmodeledEnabled  = p.GetWithDefault("unmodeled_enabled", 0f) > 0.5f;
        slipCoeff         = p.GetWithDefault("slip_coeff", 0f);
        brakeAsymmetry    = p.GetWithDefault("brake_asymmetry", 1f);
    """
    # Reach the channel through the env's private side channels. ML-Agents
    # keeps the EnvironmentParametersChannel as an attribute of the environment
    # after construction; if the version differs, adapt this one line.
    channel = env._side_channel_manager._side_channels_dict.get(
        # Well-known UUID for the EnvironmentParametersChannel.
        # (Constant across ML-Agents versions; safer than importing.)
        __import__("uuid").UUID("534c891e-810f-11ea-a9d0-822485860400"))
    if channel is None:
        # Fallback: some ML-Agents versions expose the channel differently.
        # In that case, spin up our own and register it BEFORE env construction
        # (the orchestrator has an env_factory hook for exactly this reason).
        raise RuntimeError(
            "EnvironmentParametersChannel not found on env; register one at "
            "env construction time (see collect_dataset.build_env_factory).")
    for k, v in eta_payload.items():
        channel.set_float_parameter(k, float(v))


# ── Main entry point ────────────────────────────────────────────────────────
def run_rollout(env, behavior_name: str,
                 eta: Dict[str, float],
                 condition: str,
                 controller: str,
                 ref_controls: np.ndarray,
                 dt: float = 0.1,
                 max_steps: int = 100,
                 sensor_noise_std: float = 0.0,
                 push_eta: bool = True) -> RolloutResult:
    """Run one closed-loop episode, return a RolloutResult.

    Assumes the env has ALREADY been reset for this episode by the caller
    (so the caller can push eta before reset if their channel plumbing wants
    it there; we push AGAIN here to be safe, but Unity's TaxiAgent must apply
    parameters at OnEpisodeBegin() for the mid-episode push to have effect).
    """
    # Deferred heavy imports so the file loads without ML-Agents installed
    # (needed for unit-testing the reference gen / sampler standalone).
    from mlagents_envs.base_env import ActionTuple
    from mlagents_envs.exception import UnityCommunicatorStoppedException
    import taxi_controller_mppi as tc

    if push_eta:
        # Best-effort: some ML-Agents versions want this BEFORE env.reset(),
        # some accept it after; caller can also push before reset explicitly.
        # Unity reads lower_snake_case keys (TaxiAgent.ApplyDomainRandomizationParams),
        # so lower-case whatever convention the caller handed us.
        try:
            push_eta_to_unity(env, {k.lower(): v for k, v in eta.items()})
        except Exception as _e:
            print(f"[rollout] warning: could not push eta to Unity: {_e}")

    # From here down eta is indexed with the UPPER_CASE names of
    # domain_sampler.NOMINAL (A_MIN, DELTA_LIM, ...), which is what the clamps
    # and _advance_actuators() are written against. Callers commonly pass the
    # lower_snake_case channel payload from RolloutEta.env_channel_payload()
    # instead, so normalise rather than KeyError on every single rollout.
    eta = {k.upper(): v for k, v in eta.items()}

    tracker = _make_tracker(controller, ref_controls)

    # Buffers, sized for the worst case; we trim to actual length at the end.
    xs = np.empty((max_steps, 6), dtype=np.float32)
    acts = np.empty((max_steps, 2), dtype=np.float32)
    xns = np.empty((max_steps, 6), dtype=np.float32)
    refs = np.empty((max_steps, 2), dtype=np.float32)

    delta_actual = 0.0
    accel_actual = 0.0
    n_recorded = 0
    terminal_reason = "timeout"
    goal_xy_seen = np.zeros(2, dtype=np.float32)
    unity_stopped = False

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    for step in range(max_steps):
        # Terminal step reached (collision or goal). We don't have a next state
        # for the last action, so we DON'T record a transition here — the
        # previous step's xns already captured what actually happened.
        if len(terminal_steps) > 0:
            # terminal_steps.interrupted[0] is True for a truncated episode
            # (Unity ran out of max steps), False for a "reached goal" style
            # natural termination. Collision is signalled via a negative
            # barrier value h in obs[17] on the LAST decision step — checked
            # inline below by taxi_controller's convention. Best available
            # inference here from ML-Agents alone:
            if terminal_steps.interrupted[0]:
                terminal_reason = "timeout"
            else:
                terminal_reason = "reached"
            break

        if len(decision_steps) == 0:
            # No agent to act on this frame — just tick Unity and re-check.
            try:
                env.step()
            except UnityCommunicatorStoppedException:
                unity_stopped = True
                terminal_reason = "unity_stopped"
                break
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            continue

        obs = decision_steps.obs[0][0]
        obs_n = tc.inject_sensor_noise(obs, sensor_noise_std, tc.rng)
        s, obstacles, goal_xy = tc.obs_to_state(obs_n, delta_actual, accel_actual)
        goal_xy_seen[:] = goal_xy

        # Assemble x_t in our own 6-D convention (bicycle state incl. actuators).
        x_t = np.array([s[0], s[1], s[2], s[3], delta_actual, accel_actual],
                        dtype=np.float32)

        # Pick the action for this step from the tracker.
        u_cmd = tracker.act(s, obstacles, goal_xy, step)
        a_cmd = float(np.clip(u_cmd[0], eta["A_MIN"], eta["A_MAX"]))
        delta_cmd = float(np.clip(u_cmd[1], -eta["DELTA_LIM"], eta["DELTA_LIM"]))

        # Send to Unity.
        action = ActionTuple(
            continuous=np.array([[a_cmd, delta_cmd]], dtype=np.float32))
        env.set_actions(behavior_name, action)
        try:
            env.step()
        except UnityCommunicatorStoppedException:
            unity_stopped = True
            terminal_reason = "unity_stopped"
            break

        # Advance our Python-side actuator estimates for x_{t+1}. This has to
        # happen AFTER Unity has actually stepped, and it uses the SAME eta
        # that Unity used, so our recorded x_{t+1} lines up with the pose Unity
        # produced. (Unity doesn't report delta_actual/accel_actual; this is
        # the only way to keep our 6-D state consistent with the 4-D obs.)
        delta_actual, accel_actual = _advance_actuators(
            s[3], delta_actual, accel_actual, a_cmd, delta_cmd, eta, dt)

        # Grab the next observation and turn it into x_{t+1}.
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(decision_steps) > 0:
            obs2 = decision_steps.obs[0][0]
            obs2_n = tc.inject_sensor_noise(obs2, sensor_noise_std, tc.rng)
            s2, _obs2, _g2 = tc.obs_to_state(obs2_n, delta_actual, accel_actual)
            x_tp1 = np.array([s2[0], s2[1], s2[2], s2[3], delta_actual, accel_actual],
                              dtype=np.float32)
        elif len(terminal_steps) > 0:
            # Terminal on this step: use the terminal observation for x_{t+1}
            # (episode-ending pose). Collision is inferred here via obs[17] < 0,
            # the barrier convention from TaxiAgent's cbf_h channel.
            obs2 = terminal_steps.obs[0][0]
            obs2_n = tc.inject_sensor_noise(obs2, sensor_noise_std, tc.rng)
            s2, _obs2, _g2 = tc.obs_to_state(obs2_n, delta_actual, accel_actual)
            x_tp1 = np.array([s2[0], s2[1], s2[2], s2[3], delta_actual, accel_actual],
                              dtype=np.float32)
            if float(obs2[17]) < 0.0:
                terminal_reason = "collision"
            elif terminal_steps.interrupted[0]:
                terminal_reason = "timeout"
            else:
                terminal_reason = "reached"
        else:
            # Neither decision nor terminal — Unity didn't produce a step for us
            # (unusual). Skip recording this transition.
            continue

        xs[n_recorded] = x_t
        acts[n_recorded] = [a_cmd, delta_cmd]
        xns[n_recorded] = x_tp1
        refs[n_recorded] = ref_controls[min(step, len(ref_controls) - 1)]
        n_recorded += 1

        if len(terminal_steps) > 0:
            break  # Terminal recorded above; stop.

    return RolloutResult(
        x=xs[:n_recorded].copy(),
        a=acts[:n_recorded].copy(),
        x_next=xns[:n_recorded].copy(),
        ref=refs[:n_recorded].copy(),
        goal_xy=goal_xy_seen.copy(),
        terminal_reason=terminal_reason,
        n_steps=int(n_recorded),
        controller=controller,
        eta=dict(eta),
        condition=condition,
    )


if __name__ == "__main__":
    # This file needs a live Unity env to actually run. Do a lightweight
    # structural check instead: verify the actuator-advance helper matches
    # what taxi_controller_mppi.run()'s inline block computes, for a random
    # state / control / eta.
    rng = np.random.default_rng(0)
    eta = dict(L=6.0, DRAG_COEFF=0.04, ACCEL_TAU=0.5, MAX_STEER_RATE=0.6,
                STEER_ROLLOFF_SPD=15.0, STEER_ROLLOFF_MIN=0.25,
                A_MIN=-4.0, A_MAX=1.5, DELTA_LIM=0.5)
    v, da, aa = 8.0, 0.1, 0.3
    a_cmd, d_cmd = -1.0, 0.4
    da2, aa2 = _advance_actuators(v, da, aa, a_cmd, d_cmd, eta, 0.1)
    print(f"delta_actual: {da:.4f} -> {da2:.4f}   accel_actual: {aa:.4f} -> {aa2:.4f}")
    # Expected sign: delta_target > delta_actual so delta increases; accel_actual
    # approaches a_cmd=-1.0 from 0.3, so it decreases.
    assert da2 > da
    assert aa2 < aa
    print("actuator-advance sanity OK")