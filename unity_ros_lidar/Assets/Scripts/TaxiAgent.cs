using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// ML-Agents Agent for aircraft taxiing.
/// External Python process sends actions [a, delta] each decision step.
///
/// No taxiway/map/Frenet path-following: the ego always navigates by a live straight-line
/// bearing to a goal marker, in free space. Obstacle avoidance (CBF/MPPI for dynamic agents,
/// the LiDAR static costmap for buildings/walls) is frame-independent and unaffected.
///
/// OBSERVATION VECTOR (7 floats — must match Python OBS_SIZE=7), always WORLD frame:
///   [0]  x_ego    — Unity Z position [m]
///   [1]  y_ego    — Unity X position [m]
///   [2]  theta    — world heading [rad]
///   [3]  v
///   [4]  goal_dx  — true Euclidean distance to the goal [m] (999 if no goal marker)
///   [5]  goal_z   — goal world Z position [m] (same axis as [0]); = x_ego if no goal marker
///   [6]  goal_x   — goal world X position [m] (same axis as [1]); = y_ego if no goal marker
///
/// NO DYNAMIC-OBSTACLE SLOTS. Until now this vector carried the nearest K_OBS agents'
/// exact positions and velocities, plus a cbf_h barrier value derived from the nearest
/// one — an ORACLE, straight out of the scenario manager, with no sensing involved. The
/// controller now perceives other aircraft only through the LiDAR point cloud, the same
/// way it perceives buildings, so what it plans against is what it can actually see.
/// Re-adding a slot here re-introduces ground truth the sensor model cannot justify.
///
/// COORDINATE MAPPING:
///   Python X (forward) = Unity +Z
///   Python Y (lateral)  = Unity +X
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class TaxiAgent : Unity.MLAgents.Agent
{
    // ── Aircraft parameters ────────────────────────────────────────────────────

    [Header("Aircraft parameters — must match Python DT, L, and dynamics constants")]
    public float wheelbase    = 6.0f;   // nose-to-main-gear distance [m]
    public float maxAccel     = 1.5f;   // max thrust accel [m/s²]
    public float maxBrake     = 4.0f;   // max brake decel [m/s²]
    public float maxSteer     = 0.5f;   // nose-wheel max angle at zero speed [rad]
    public float desiredSpeed = 8.0f;   // target taxi speed [m/s]

    [Header("Realistic kinematic extensions")]
    [Tooltip("Aerodynamic + rolling drag coefficient [1/s]. v_dot -= dragCoeff * v.")]
    public float dragCoeff = 0.04f;     // at 8 m/s gives 0.32 m/s² passive deceleration
    [Tooltip("First-order time constant for thrust/brake lag [s]. 0 = instant.")]
    public float accelTau  = 0.5f;      // ~0.5s engine/brake response
    [Tooltip("Max nose-wheel steering rate [rad/s].")]
    public float maxSteerRate = 0.6f;   // ~34 deg/s
    [Tooltip("Speed above which nose-wheel authority rolls off [m/s]. " +
             "At rolloffSpeed, max steer = maxSteer * steerRolloffMin.")]
    public float steerRolloffSpeed = 15f;
    [Tooltip("Minimum steering authority fraction at high speed (0-1).")]
    public float steerRolloffMin   = 0.25f;

    // ── Unmodeled effect (dataset Condition C) ────────────────────────────────
    // These have NO analogue in the Python analytic bicycle model, by construction:
    // Condition C exists to probe structural model mismatch, not parameter drift.
    // All default to no-ops, so an A/B rollout (which never pushes
    // unmodeled_enabled=1) behaves exactly like the analytic model.

    [Header("Unmodeled effect (dataset Condition C only)")]
    [Tooltip("Master gate. Set from the 'unmodeled_enabled' environment parameter.")]
    public bool  unmodeledEnabled  = false;
    [Tooltip("Tyre-slip understeer: yaw rate loses slipCoeff * v² * delta [rad/s].")]
    public float slipCoeff         = 0.0f;
    [Tooltip("Multiplier on braking commands (a_cmd < 0) — brakes bite harder than thrust.")]
    public float brakeAsymmetry    = 1.0f;
    [Tooltip("Amplitude of a smooth spatial friction field modulating effective drag.")]
    public float frictionNoiseAmp  = 0.0f;
    [Tooltip("Pure transport delay on commands, in control steps. The command " +
             "issued now reaches the actuator this many steps later — a time-shift " +
             "that NO static parameter set (drag/lag/wheelbase) can reproduce, " +
             "unlike slip/brake/friction. This is the non-absorbable effect.")]
    public int   actuationDelaySteps = 0;

    // ── Scene references ───────────────────────────────────────────────────────

    [Header("Performance")]
    [Tooltip("Time scale for headless/fast runs. 1 = real-time, 20 = 20x speed.")]
    public float simulationTimeScale = 1f;

    [Header("Scene references")]
    public Transform goalMarker;
    public float taxiwayHalfWidth = 10f;
    public float dSafe            = 6.0f;

    [Header("Follow-vehicle arrival (Task 5)")]
    [Tooltip("Extra slack beyond dSafe within which the ego counts as 'arrived' when parked " +
             "behind a lead that has itself reached the shared goal. Prevents a deadlock where " +
             "the lead occupies the goal and the ego can never close the final dSafe metres.")]
    public float followArriveMargin = 3.0f;
    [Tooltip("Ego must be slower than this [m/s] to be considered settled behind the parked lead.")]
    public float followArriveSpeed  = 0.5f;

    [Header("Spawn randomisation (set ranges to 0 to disable)")]
    public float spawnLateralRange = 0.0f;
    public float spawnHeadingRange = 0.0f;

    // ── Multi-agent scenario (new path) ───────────────────────────────────────

    [Header("Multi-agent scenario (assign ScenarioManager for multi-obstacle support)")]
    [Tooltip("Assign to enable multi-obstacle observations and difficulty curriculum. " +
             "Leave null to fall back to the single-agent incursionController path.")]
    public TaxiScenarioManager scenarioManager;

    // ── Legacy single-agent fields (kept for backward compatibility) ───────────

    [Header("Legacy single-agent (used only when scenarioManager is null)")]
    public IncursionAgentController incursionController;
    [Tooltip("Transform of the crossing obstacle. Used for observation when scenarioManager is null.")]
    public Transform incursionAgent;
    [Tooltip("Empty GameObject where the incursion path crosses the taxiway centreline.")]
    public Transform conflictPoint;
    public float defaultIncursionDt   = 0.0f;

    public float maxEpisodeSeconds = 60f;

    // ── Constants ─────────────────────────────────────────────────────────────


    // ── Private state ──────────────────────────────────────────────────────────

    Rigidbody _rb;
    float     _speed;
    bool      _collided;
    float     _episodeTime;
    Vector3   _spawnPos;
    int       _episodeIndex;

    // Realistic kinematic state
    float   _deltaActual;  // current nose-wheel angle [rad] (rate-limited)
    float   _accelActual;  // current acceleration [m/s²] (lag-filtered)

    // Condition-C actuation delay: FIFO of (a_cmd, delta_cmd) awaiting arrival.
    // Cleared each episode in ApplyDomainRandomizationParams so a delay change
    // between rollouts never carries stale commands across the boundary.
    readonly Queue<Vector2> _cmdDelayBuffer = new Queue<Vector2>();


    // Current scenario type (from the side channel) — used for scenario-specific goal/
    // arrival logic (e.g. the follow-vehicle arrival test).
    int _scenarioType;

    // True once the goal has been reached this episode: freezes the aircraft (zero speed, no
    // further dynamics) instead of letting it keep driving/overshoot for the frame(s) between
    // detecting arrival and the episode actually resetting. Reset each episode.
    bool _stoppedAtGoal;

    // ── Unity / ML-Agents lifecycle ────────────────────────────────────────────

    public override void Initialize()
    {
        Time.timeScale = simulationTimeScale;
        _rb = GetComponent<Rigidbody>();
        _rb.isKinematic = true;
        _rb.constraints = RigidbodyConstraints.FreezeRotationX
                        | RigidbodyConstraints.FreezeRotationZ
                        | RigidbodyConstraints.FreezePositionY;
        _spawnPos = transform.position;
    }

    public override void OnEpisodeBegin()
    {
        _collided     = false;
        _speed        = desiredSpeed;
        _deltaActual  = 0f;
        _accelActual  = 0f;
        _episodeTime  = 0f;
        _stoppedAtGoal = false;

        var ep = Academy.Instance.EnvironmentParameters;

        // Per-rollout dynamics randomisation. Read HERE (not in Initialize) so a
        // fresh eta is picked up every episode — collect_dataset.py pushes one
        // eta per rollout immediately BEFORE env.reset().
        ApplyDomainRandomizationParams(ep);

        // ── Spawn airplane ─────────────────────────────────────────────────────
        float latOff = ep.GetWithDefault("spawn_lateral", float.NaN);
        if (float.IsNaN(latOff))
            latOff = Random.Range(-spawnLateralRange, spawnLateralRange);
        float hdgOff = Random.Range(-spawnHeadingRange, spawnHeadingRange);

        transform.position    = _spawnPos + new Vector3(latOff, 0f, 0f);
        transform.eulerAngles = new Vector3(0f, hdgOff * Mathf.Rad2Deg, 0f);

        // ── Multi-agent path (TaxiScenarioManager) ────────────────────────────────
        if (scenarioManager != null)
        {
            float difficulty     = ep.GetWithDefault("difficulty",      scenarioManager.defaultDifficulty);
            float incursionDt    = ep.GetWithDefault("incursion_dt",    scenarioManager.defaultIncursionDt);
            float ambulanceSpeed = ep.GetWithDefault("ambulance_speed", scenarioManager.defaultAmbulanceSpeed);

            float conflictZOffset = ep.GetWithDefault("conflict_z_offset", float.NaN);
            float crossDirSign    = ep.GetWithDefault("cross_dir_sign",    0f);
            int   scenarioType    = (int)ep.GetWithDefault("scenario_type",  0f);
            _scenarioType         = scenarioType;
            float headOnProb      = ep.GetWithDefault("head_on_prob",       0f);
            float episodeSpeed    = ep.GetWithDefault("desired_speed",      -1f);
            if (episodeSpeed > 0f) _speed = episodeSpeed;

            // Deterministic map/path selection: when Python supplies a per-episode
            // seed, re-seed Unity's RNG so the ego path (and obstacle assignment)
            // are reproducible across runs — required for a fair CBF vs no-CBF
            // comparison on identical geometry.
            float episodeSeed = ep.GetWithDefault("episode_seed", -1f);
            if (episodeSeed >= 0f) Random.InitState(Mathf.RoundToInt(episodeSeed));

            // Fix B: allow Python to shrink the reachable arc each episode so
            // agents spawn close to the conflict point and arrive during the episode.
            float reachSecs = ep.GetWithDefault("episode_reach_seconds", -1f);
            if (reachSecs > 0f) scenarioManager.episodeReachSeconds = reachSecs;

            // Bound the ego goal so long (runway-length) paths don't run the episode to timeout.
            float maxGoal = ep.GetWithDefault("max_goal_dist", -1f);
            if (maxGoal > 0f) scenarioManager.maxEgoGoalDist = maxGoal;

            // Converging scenario: let Python set the ring radius (long-range spawn distance).
            float convergeDist = ep.GetWithDefault("converge_spawn_dist", -1f);
            if (convergeDist > 0f) scenarioManager.convergeSpawnDist = convergeDist;

            // Head-on scenario: let Python set how far ahead the oncoming agents spawn.
            float headOnGap = ep.GetWithDefault("head_on_gap", -1f);
            if (headOnGap > 0f) scenarioManager.headOnApproachGap = headOnGap;

            scenarioManager.ResetEpisode(
                difficulty,
                episodeSpeed > 0f ? episodeSpeed : desiredSpeed,
                transform, incursionDt, ambulanceSpeed,
                conflictZOffset, crossDirSign, scenarioType, headOnProb);

            // Two-point sandbox: place the plane at the start marker with NO rotation applied
            // (identity heading, facing world +Z). It no longer auto-faces the goal/lane tangent.
            if (scenarioManager.EgoHasSpawn)
                transform.SetPositionAndRotation(
                    scenarioManager.EgoSpawnPos,
                    Quaternion.identity);
        }
        // ── Legacy single-agent path ──────────────────────────────────────────
        else if (incursionController != null && conflictPoint != null)
        {
            float incursionDt    = ep.GetWithDefault("incursion_dt",    defaultIncursionDt);
            float ambulanceSpeed = ep.GetWithDefault("ambulance_speed", -1f);

            float egoTtc = Mathf.Max(0f,
                (conflictPoint.position.z - transform.position.z) / desiredSpeed);
            float speed  = ambulanceSpeed > 0f ? ambulanceSpeed : incursionController.crossSpeed;
            Vector3 dir  = incursionController.CrossDirectionNormalized;
            Vector3 start = conflictPoint.position - dir * speed * (egoTtc + incursionDt);
            incursionController.ResetCrossing(start, speed);
        }

        _episodeIndex++;
    }

    // ── Observations (7 floats) ────────────────────────────────────────────────
    // Layout: [ego(4)] [goal_dist(1)] [goal_pos(2)]
    // See class doc-comment for full slot descriptions.

    public override void CollectObservations(VectorSensor sensor)
    {
        // ── Ego state [0..3] ───────────────────────────────────────────────────
        // No TaxiwayNetwork/Frenet path-following: the ego always navigates by a live
        // straight-line bearing to the goal marker, in free space. d is always 0 by
        // construction (there is no lane), so only progress-to-goal, control effort, and
        // obstacle avoidance (CBF/MPPI + LiDAR static cost, both frame-independent) drive
        // the plan.
        // Everything is reported in the WORLD frame: ego world position (z, x) and world heading,
        // plus the goal's world position (z, x). Python computes the true Euclidean distance to
        // the goal at every rollout point directly from these — no projection / bearing frame.
        float thWorld = transform.eulerAngles.y * Mathf.Deg2Rad;
        float ego0 = transform.position.z;   // world Z → x_fwd (ROS convention used downstream)
        float ego1 = transform.position.x;   // world X → y_lat
        float ego2 = Mathf.Atan2(Mathf.Sin(thWorld), Mathf.Cos(thWorld));

        float goal_val, goalZ, goalX;
        Transform goalT = ResolveGoalMarker();
        if (goalT != null)
        {
            Vector3 toGoal = goalT.position - transform.position;
            toGoal.y = 0f;
            goal_val = toGoal.magnitude;      // true Euclidean distance to goal [m]
            goalZ    = goalT.position.z;      // goal world position (same axes as ego0/ego1)
            goalX    = goalT.position.x;
        }
        else
        {
            // Legacy fallback (no goal marker): no goal to measure against.
            goal_val = 999f;
            goalZ    = ego0;                  // degenerate: goal at the ego ⇒ zero goal cost
            goalX    = ego1;
        }

        sensor.AddObservation(ego0);
        sensor.AddObservation(ego1);
        sensor.AddObservation(ego2);
        sensor.AddObservation(_speed);

        // ── Goal distance and goal world position [4..6] ──────────────────────
        // The scenario manager's agents are deliberately NOT reported here — see the
        // class doc-comment. They reach the controller only as LiDAR returns.
        sensor.AddObservation(goal_val);
        sensor.AddObservation(goalZ);      // [5] goal world Z (x_fwd axis)
        sensor.AddObservation(goalX);      // [6] goal world X (y_lat axis)
    }

    // ── Actions received from Python ───────────────────────────────────────────

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Already reached the goal this episode: stay parked, ignore further commands, until
        // OnEpisodeBegin resets us. Prevents the aircraft continuing to drive/overshoot past the
        // goal (or into something beyond it) for the frame(s) between arrival and episode reset.
        if (_stoppedAtGoal)
        {
            _speed = 0f;
            return;
        }

        float a_cmd     = actions.ContinuousActions[0];
        float delta_cmd = actions.ContinuousActions[1];

        ApplyBicycleDynamics(a_cmd, delta_cmd);
        AddReward(ComputeReward());

        _episodeTime += Time.fixedDeltaTime;

        // Goal distance: straight-line distance to the goal marker (no map/lane involved).
        Transform goalTAct = ResolveGoalMarker();
        float goal_dx = goalTAct != null
            ? Vector3.Distance(
                  new Vector3(transform.position.x, 0f, transform.position.z),
                  new Vector3(goalTAct.position.x,   0f, goalTAct.position.z))
            : 999f;

        // Off-road: never in free-field mode (no lane to leave) — see LaneLateralError.
        float lateralErr = LaneLateralError();

        bool reached = goal_dx < 3.0f;

        // Follow-vehicle: the lead parks ON the shared goal, so the ego settles ~dSafe behind it
        // and goal_dx never reaches the 2 m threshold — a false timeout. Count arrival when the
        // ego has stopped a safe gap behind a lead that has itself reached the goal.
        if (!reached && _scenarioType == (int)ScenarioType.FollowVehicle && scenarioManager != null)
        {
            var agents = scenarioManager.ActiveAgents;
            if (agents.Count > 0 && agents[0] != null && agents[0].ReachedPathEnd
                && goal_dx < dSafe + followArriveMargin
                && _speed < followArriveSpeed)
                reached = true;
        }

        bool timeout = _episodeTime >= maxEpisodeSeconds;
        bool offRoad = Mathf.Abs(lateralErr) > taxiwayHalfWidth + 2f;

        if (reached)   AddReward( 10f);
        if (_collided) AddReward(-20f);

        if (reached && !_stoppedAtGoal)
        {
            // Stop right here and STAY stopped — deliberately NOT ending the episode, so
            // ML-Agents doesn't immediately reset/respawn the plane back at the start marker for
            // the next episode (which looked like "it never actually stopped"). It parks here
            // until collision/timeout/off-road ends the episode for real.
            _speed         = 0f;
            _accelActual   = 0f;
            _deltaActual   = 0f;
            _stoppedAtGoal = true;
        }

        if (_collided || timeout || offRoad || reached)
            EndEpisode();
    }

    // ── Domain randomisation (dataset collection) ─────────────────────────────
    //
    // Receives the per-rollout eta pushed by dataset_collection/collect_dataset.py
    // over ML-Agents' EnvironmentParametersChannel. Keys are the lower-snake_case
    // names written by domain_sampler.RolloutEta.env_channel_payload():
    //   l, drag_coeff, accel_tau, max_steer_rate, steer_rolloff_spd,
    //   steer_rolloff_min, a_min, a_max, delta_lim,
    //   unmodeled_enabled, slip_coeff, brake_asymmetry, friction_noise_amp
    // Every lookup falls back to the current Inspector value, so a normal
    // (non-dataset) Play-mode run is completely unaffected.

    void ApplyDomainRandomizationParams(EnvironmentParameters ep)
    {
        wheelbase         = ep.GetWithDefault("l",                 wheelbase);
        dragCoeff         = ep.GetWithDefault("drag_coeff",        dragCoeff);
        accelTau          = ep.GetWithDefault("accel_tau",         accelTau);
        maxSteerRate      = ep.GetWithDefault("max_steer_rate",    maxSteerRate);
        steerRolloffSpeed = ep.GetWithDefault("steer_rolloff_spd", steerRolloffSpeed);
        steerRolloffMin   = ep.GetWithDefault("steer_rolloff_min", steerRolloffMin);
        maxAccel          = ep.GetWithDefault("a_max",             maxAccel);
        maxSteer          = ep.GetWithDefault("delta_lim",         maxSteer);
        // Python's A_MIN is a signed lower bound (≈ -4); maxBrake is its magnitude.
        maxBrake          = Mathf.Abs(ep.GetWithDefault("a_min",  -maxBrake));

        unmodeledEnabled   = ep.GetWithDefault("unmodeled_enabled", 0f) > 0.5f;
        slipCoeff          = ep.GetWithDefault("slip_coeff",        0f);
        brakeAsymmetry     = ep.GetWithDefault("brake_asymmetry",   1f);
        frictionNoiseAmp   = ep.GetWithDefault("friction_noise_amp", 0f);
        actuationDelaySteps = Mathf.RoundToInt(ep.GetWithDefault("actuation_delay_steps", 0f));
        _cmdDelayBuffer.Clear();   // fresh pipeline each rollout
    }

    // ── Bicycle kinematic model (with realistic extensions) ──────────────────────
    //
    // Four additions over the simple model:
    //   1. Steering rate limiter  — nose wheel moves at most maxSteerRate rad/s
    //   2. Speed-dependent limit  — authority rolls off linearly above steerRolloffSpeed
    //   3. Acceleration lag       — first-order filter with time constant accelTau
    //   4. Aerodynamic drag       — passive deceleration proportional to speed

    void ApplyBicycleDynamics(float a_cmd, float delta_cmd)
    {
        float dt = Time.fixedDeltaTime;

        // 0 — Condition-C actuation delay (applied BEFORE the actuator dynamics,
        // as a physical transport/computation delay would be). The command issued
        // this step is buffered; the command that actually drives the actuators is
        // the one issued `actuationDelaySteps` steps ago. While the pipeline fills
        // at episode start, the actuator coasts (neutral command). A pure delay is
        // a time-shift of the whole command sequence, which no static (drag/lag/
        // wheelbase) parameterisation can mimic — that is what makes condition C a
        // genuine structural mismatch rather than an absorbable parameter drift.
        if (unmodeledEnabled && actuationDelaySteps > 0)
        {
            _cmdDelayBuffer.Enqueue(new Vector2(a_cmd, delta_cmd));
            if (_cmdDelayBuffer.Count > actuationDelaySteps)
            {
                Vector2 arrived = _cmdDelayBuffer.Dequeue();
                a_cmd     = arrived.x;
                delta_cmd = arrived.y;
            }
            else
            {
                a_cmd     = 0f;   // no command has arrived yet — coast
                delta_cmd = 0f;
            }
        }

        // 1 + 2 — rate-limited, speed-dependent nose-wheel steering
        float speedFraction  = Mathf.Clamp01(_speed / Mathf.Max(1f, steerRolloffSpeed));
        float effectiveLimit = maxSteer * Mathf.Lerp(1f, steerRolloffMin, speedFraction);
        float deltaTarget    = Mathf.Clamp(delta_cmd, -effectiveLimit, effectiveLimit);
        float maxDelta       = maxSteerRate * dt;
        _deltaActual = Mathf.MoveTowards(_deltaActual, deltaTarget, maxDelta);

        // 3 — first-order acceleration lag  (τ·ȧ + a = a_cmd)
        // Condition C: asymmetric brake response — real brakes bite harder than
        // thrust pushes, so scale negative commands only. Applied BEFORE the clamp
        // so the actuator box itself is still respected.
        if (unmodeledEnabled && a_cmd < 0f)
            a_cmd *= brakeAsymmetry;

        float a_clamped  = Mathf.Clamp(a_cmd, -maxBrake, maxAccel);
        if (accelTau > 1e-3f)
            _accelActual += (a_clamped - _accelActual) * (dt / accelTau);
        else
            _accelActual  = a_clamped;

        // 4 — aerodynamic + rolling drag (acts opposite to motion)
        // Condition C: a smooth pseudo-random spatial friction field, so effective
        // drag varies across the taxiway in a way no single dragCoeff scalar can
        // represent.
        float dragScale = 1f;
        if (unmodeledEnabled && frictionNoiseAmp > 0f)
        {
            Vector3 p = transform.position;
            dragScale = 1f + frictionNoiseAmp
                      * Mathf.Sin(p.x * 0.05f) * Mathf.Cos(p.z * 0.07f);
        }
        float drag    = dragCoeff * dragScale * _speed;
        float v_dot   = _accelActual - drag;
        _speed        = Mathf.Max(0f, _speed + v_dot * dt);

        // Bicycle yaw rate: dθ/dt = (v/L) * tan(δ)
        float dTheta = (_speed / wheelbase) * Mathf.Tan(_deltaActual) * dt;

        // Condition C: tyre slip. Lateral force ~ slipCoeff * v² * δ opposes the
        // turn, so the realised yaw rate understeers relative to the kinematic
        // prediction — growing with v². The Python model's dθ = (v/L)·tan(δ)·dt has
        // no v² term at any parameter setting, which is exactly the structural
        // mismatch this condition probes.
        if (unmodeledEnabled && slipCoeff > 0f)
            dTheta -= slipCoeff * _speed * _speed * _deltaActual * dt / wheelbase;

        transform.Rotate(Vector3.up, dTheta * Mathf.Rad2Deg);
        transform.position += transform.forward * (_speed * dt);
    }

    // ── Reward ────────────────────────────────────────────────────────────────

    float ComputeReward()
    {
        // Lane error: cross-track + heading-error in map mode, global X + global heading otherwise.
        float lateralErr = LaneLateralError();
        float theta      = LaneHeadingError();

        float r = -0.01f * lateralErr * lateralErr
                  -0.04f * theta * theta
                  -0.01f * Mathf.Pow(_speed - desiredSpeed, 2f)
                  + 0.02f * _speed;

        if (Mathf.Abs(lateralErr) > taxiwayHalfWidth) r -= 0.5f;
        return r;
    }

    // Signed heading error [rad] from a world-space travel tangent to the plane's forward,
    // matching Python's Frenet convention: theta_e = atan2(tan_x, tan_z) − plane_heading.
    // Used in two-point mode when the travel direction opposes the taxiway's stored tangent.
    float HeadingErrorTo(Vector3 tan)
    {
        float e = Mathf.Atan2(tan.x, tan.z) - Mathf.Atan2(transform.forward.x, transform.forward.z);
        return Mathf.Atan2(Mathf.Sin(e), Mathf.Cos(e));
    }

    // Goal marker to drive to: prefers the scenario manager's goal marker (freeGoalMode /
    // two-point workflows) so it doesn't need to be duplicated on this component; falls back to
    // the legacy goalMarker field.
    Transform ResolveGoalMarker()
    {
        if (scenarioManager != null && scenarioManager.egoGoalMarker != null)
            return scenarioManager.egoGoalMarker;
        return goalMarker;
    }

    // No taxiway/lane: always 0 (never off-road, no cross-track cost) when there's a goal to
    // navigate to; global X only in the legacy no-goal-marker fallback.
    float LaneLateralError()
    {
        if (ResolveGoalMarker() != null) return 0f;
        return transform.position.x;
    }

    // Heading error vs the live bearing to the goal; global heading in the legacy fallback.
    float LaneHeadingError()
    {
        Transform goalT = ResolveGoalMarker();
        if (goalT != null)
        {
            Vector3 toGoal = goalT.position - transform.position;
            toGoal.y = 0f;
            if (toGoal.sqrMagnitude > 1e-6f) return HeadingErrorTo(toGoal.normalized);
            return 0f;
        }
        float th = transform.eulerAngles.y * Mathf.Deg2Rad;
        return Mathf.Atan2(Mathf.Sin(th), Mathf.Cos(th));
    }

    // ── Collision detection ────────────────────────────────────────────────────

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject == gameObject) return;
        if (other.gameObject.name == "Plane" || other.gameObject.isStatic) return;
        _collided = true;
        Debug.LogWarning($"[TaxiAgent] COLLISION with '{other.gameObject.name}' at t={_episodeTime:F2}s", this);
    }

    // ── Heuristic (keyboard testing without Python) ────────────────────────────

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        ca[0] = Input.GetAxis("Vertical")   * maxAccel;
        ca[1] = Input.GetAxis("Horizontal") * maxSteer;
    }
}
