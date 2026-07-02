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
/// OBSERVATION VECTOR (20 floats — must match Python OBS_SIZE=20):
///
///   WITHOUT network (global frame, backward-compat fallback):
///   [0]  x_ego    — Unity Z position [m]
///   [1]  y_ego    — Unity X position [m]
///   [2]  theta    — heading [rad]
///   [3]  v
///   [4..15]        3 × obstacle (dx_global, dy_global, vx, vy)
///   [16] goal_dx  — Unity Z distance to goal
///   [17] cbf_h
///   [18] 0        (tangent_x stub)
///   [19] 1        (tangent_z stub — points forward)
///
///   WITH network (Frenet frame):
///   [0]  s         — arc-length along ego path [m]
///   [1]  d         — signed cross-track error [m]  (+ = left)
///   [2]  theta_e   — heading error vs path tangent [rad]
///   [3]  v
///   [4..15]        3 × obstacle (dx_global, dy_global, vx, vy)  ← still global Δ
///   [16] goal_ds  — remaining arc-length to path end [m]
///   [17] cbf_h
///   [18] tangent_x — world X component of path tangent (for CBF rotation in Python)
///   [19] tangent_z — world Z component of path tangent
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

    // ── Scene references ───────────────────────────────────────────────────────

    [Header("Performance")]
    [Tooltip("Time scale for headless/fast runs. 1 = real-time, 20 = 20x speed.")]
    public float simulationTimeScale = 1f;

    [Header("Scene references")]
    public Transform goalMarker;
    public float taxiwayHalfWidth = 10f;
    public float dSafe            = 6.0f;

    [Header("Spawn randomisation (set ranges to 0 to disable)")]
    public float spawnLateralRange = 0.0f;
    public float spawnHeadingRange = 0.0f;

    // ── Multi-agent scenario (new path) ───────────────────────────────────────

    [Header("Multi-agent scenario (assign ScenarioManager for multi-obstacle support)")]
    [Tooltip("Assign to enable multi-obstacle observations and difficulty curriculum. " +
             "Leave null to fall back to the single-agent incursionController path.")]
    public TaxiScenarioManager scenarioManager;

    [Header("Map network (optional — enables Frenet frame observations)")]
    [Tooltip("Assign the TaxiwayNetwork in the scene to switch observations to Frenet frame. " +
             "BehaviorParameters must be set to 20 continuous observations.")]
    public TaxiwayNetwork network;

    [Header("Task 7 — Blind Corner (occlusion)")]
    [Tooltip("Layers that block line-of-sight to obstacles (e.g. Terminal buildings). In the " +
             "BlindCorner scenario an obstacle is masked from the observation while a linecast " +
             "from the ego to it hits one of these layers, so it 'pops out' late. Leave as " +
             "Nothing to disable occlusion (obstacles always visible).")]
    public LayerMask occlusionMask = 0;
    [Tooltip("Eye/sensor height used for the line-of-sight check [m].")]
    public float occlusionEyeHeight = 1.5f;
    [Tooltip("Reveal-range floor [m]: obstacles closer than this are ALWAYS visible, even " +
             "if a building blocks the linecast. Guarantees a bounded reaction window so the " +
             "crosser is 'revealed late' (≈dist/egoSpeed seconds out) rather than never — " +
             "otherwise a permanent occluder hides the threat until contact and the recorded " +
             "expert has no evasive action. At 8 m/s, 18 m ≈ 2.25 s of warning.")]
    public float occlusionRevealDist = 18f;

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

    const int K_OBS = 3;   // nearest obstacles packed into the observation vector

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

    // Legacy single-agent velocity estimation
    Vector3 _obsPosPrev;
    Vector3 _obsVel;

    // Current scenario type (from the side channel) — needed in CollectObservations
    // for the BlindCorner occlusion mask.
    int _scenarioType;

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

        var ep = Academy.Instance.EnvironmentParameters;

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

            scenarioManager.ResetEpisode(
                difficulty,
                episodeSpeed > 0f ? episodeSpeed : desiredSpeed,
                transform, incursionDt, ambulanceSpeed,
                conflictZOffset, crossDirSign, scenarioType, headOnProb);

            // ── Task 6: Ego Recovery bad-spawn ─────────────────────────────────
            // ResetEpisode has just placed the ego on the path centreline. For the
            // recovery task, inject a lateral offset (along the path normal) and a
            // heading error so the controller has to steer back to the centreline —
            // recording recovery transitions. Applied here (not in the manager) so it
            // overrides the clean placement without touching the crossing scenarios.
            if (scenarioType == (int)ScenarioType.EgoRecovery)
                ApplyRecoveryBadSpawn();
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

        // Legacy velocity estimator seed
        if (incursionAgent != null) { _obsPosPrev = incursionAgent.position; _obsVel = Vector3.zero; }

        _episodeIndex++;
    }

    // ── Task 6: Ego Recovery bad-spawn ─────────────────────────────────────────
    // Shift the ego off the centreline (along the local path normal) and rotate it
    // off the path tangent, within the manager's configured ranges. In map mode the
    // normal/tangent come from the Frenet state; otherwise fall back to world axes.
    void ApplyRecoveryBadSpawn()
    {
        float latRange = scenarioManager != null ? scenarioManager.recoveryLateralRange    : 8f;
        float hdgRange = scenarioManager != null ? scenarioManager.recoveryHeadingRangeDeg : 25f;

        float latOff = Random.Range(-latRange, latRange);
        float hdgOff = Random.Range(-hdgRange, hdgRange);   // degrees

        TaxiwayPath egoPath = scenarioManager != null ? scenarioManager.EgoPath : null;
        if (network != null && egoPath != null)
        {
            PathState ps    = network.GetRelativeState(transform.position, transform.forward, egoPath);
            Vector3   tan   = ps.tangent; tan.y = 0f;
            if (tan.sqrMagnitude < 1e-6f) tan = transform.forward;
            tan.Normalize();
            Vector3   normal = Vector3.Cross(Vector3.up, tan).normalized;  // left of travel

            transform.position += normal * latOff;
            // Face along the tangent, then rotate off it by the heading error.
            transform.rotation  = Quaternion.LookRotation(tan, Vector3.up)
                                * Quaternion.Euler(0f, hdgOff, 0f);
        }
        else
        {
            // Global-frame fallback: lateral = world X, heading = yaw about up.
            transform.position += new Vector3(latOff, 0f, 0f);
            transform.Rotate(Vector3.up, hdgOff);
        }

        Debug.Log($"[TaxiAgent] Recovery bad-spawn: latOff={latOff:F2}m hdgOff={hdgOff:F1}deg");
    }

    // ── Observations (20 floats) ───────────────────────────────────────────────
    // Layout: [ego(4)] [obs0..2 (4 each, 12 total)] [goal(1)] [cbf_h(1)] [tangent(2)]
    // See class doc-comment for full slot descriptions.

    public override void CollectObservations(VectorSensor sensor)
    {
        // ── Ego state [0..3] ───────────────────────────────────────────────────
        TaxiwayPath egoPath = scenarioManager != null ? scenarioManager.EgoPath : null;
        Vector3     tangent = new Vector3(0f, 0f, 1f); // default: straight ahead
        float       ego0, ego1, ego2, goal_val;

        if (network != null && egoPath != null)
        {
            // Frenet frame
            PathState ps = network.GetRelativeState(transform.position, transform.forward, egoPath);
            ego0    = ps.s;
            ego1    = ps.d;
            ego2    = ps.thetaError;
            tangent = ps.tangent;
            goal_val = network.RemainingArcLength(transform.position, egoPath);
        }
        else
        {
            // Global frame (backward-compat)
            ego0 = transform.position.z;
            ego1 = transform.position.x;
            float th = transform.eulerAngles.y * Mathf.Deg2Rad;
            ego2 = Mathf.Atan2(Mathf.Sin(th), Mathf.Cos(th));
            goal_val = goalMarker != null
                ? goalMarker.position.z - transform.position.z
                : 999f;
        }

        sensor.AddObservation(ego0);
        sensor.AddObservation(ego1);
        sensor.AddObservation(ego2);
        sensor.AddObservation(_speed);

        // ── Nearest-K obstacle observations [4..15] ───────────────────────────
        var slots = new List<(float dx, float dy, float vx, float vy, float dist2)>();

        if (scenarioManager != null)
        {
            bool blindCorner = _scenarioType == (int)ScenarioType.BlindCorner
                               && occlusionMask.value != 0;
            foreach (var a in scenarioManager.ActiveAgents)
            {
                Vector3 pos = a.transform.position;

                // Task 7: skip obstacles whose line-of-sight to the ego is blocked by a
                // building (occlusionMask). The obstacle only enters the observation once
                // it clears the corner, so the controller reacts late (emergency stop).
                // A reveal-range floor keeps close obstacles always visible, so a permanent
                // occluder can't hide the threat until contact (which would record an
                // expert that never reacts). See occlusionRevealDist.
                if (blindCorner)
                {
                    float d2 = (pos - transform.position).sqrMagnitude;
                    if (d2 > occlusionRevealDist * occlusionRevealDist)
                    {
                        Vector3 eye = transform.position + Vector3.up * occlusionEyeHeight;
                        Vector3 tgt = pos               + Vector3.up * occlusionEyeHeight;
                        if (Physics.Linecast(eye, tgt, occlusionMask.value))
                            continue;   // occluded and still far — not yet visible to the ego
                    }
                }

                float   dx  = pos.z - transform.position.z;  // global delta
                float   dy  = pos.x - transform.position.x;
                Vector3 vel = a.Velocity;
                slots.Add((dx, dy, vel.z, vel.x, dx * dx + dy * dy));
            }
        }
        else if (incursionAgent != null)
        {
            Vector3 obsPos  = incursionAgent.position;
            _obsVel         = (obsPos - _obsPosPrev) / Time.fixedDeltaTime;
            _obsPosPrev     = obsPos;
            float dx = obsPos.z - transform.position.z;
            float dy = obsPos.x - transform.position.x;
            slots.Add((dx, dy, _obsVel.z, _obsVel.x, dx * dx + dy * dy));
        }

        slots.Sort((a, b) => a.dist2.CompareTo(b.dist2));

        float cbf_h_nearest = float.MaxValue;
        for (int i = 0; i < K_OBS; i++)
        {
            if (i < slots.Count)
            {
                var s = slots[i];
                sensor.AddObservation(s.dx);
                sensor.AddObservation(s.dy);
                sensor.AddObservation(s.vx);
                sensor.AddObservation(s.vy);
                if (i == 0) cbf_h_nearest = s.dist2 - dSafe * dSafe;
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(999f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }

        // ── Goal, CBF, tangent [16..19] ───────────────────────────────────────
        sensor.AddObservation(goal_val);
        sensor.AddObservation(cbf_h_nearest == float.MaxValue ? 999f : cbf_h_nearest);
        sensor.AddObservation(tangent.x);  // [18] tangent_x — Python rotates CBF constraint
        sensor.AddObservation(tangent.z);  // [19] tangent_z
    }

    // ── Actions received from Python ───────────────────────────────────────────

    public override void OnActionReceived(ActionBuffers actions)
    {
        float a_cmd     = actions.ContinuousActions[0];
        float delta_cmd = actions.ContinuousActions[1];

        ApplyBicycleDynamics(a_cmd, delta_cmd);
        AddReward(ComputeReward());

        _episodeTime += Time.fixedDeltaTime;

        // Goal distance: arc-length remaining when on network, global Z delta otherwise.
        TaxiwayPath egoPathAct = scenarioManager != null ? scenarioManager.EgoPath : null;
        float goal_dx = (network != null && egoPathAct != null)
            ? network.RemainingArcLength(transform.position, egoPathAct)
            : (goalMarker != null ? goalMarker.position.z - transform.position.z : 999f);

        // Off-road: cross-track error in map mode, global X otherwise.
        float lateralErr = LaneLateralError();

        bool reached = goal_dx < 2.0f;
        bool timeout = _episodeTime >= maxEpisodeSeconds;
        bool offRoad = Mathf.Abs(lateralErr) > taxiwayHalfWidth + 2f;

        if (reached)   AddReward( 10f);
        if (_collided) AddReward(-20f);

        if (reached || _collided || timeout || offRoad)
            EndEpisode();
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

        // 1 + 2 — rate-limited, speed-dependent nose-wheel steering
        float speedFraction  = Mathf.Clamp01(_speed / Mathf.Max(1f, steerRolloffSpeed));
        float effectiveLimit = maxSteer * Mathf.Lerp(1f, steerRolloffMin, speedFraction);
        float deltaTarget    = Mathf.Clamp(delta_cmd, -effectiveLimit, effectiveLimit);
        float maxDelta       = maxSteerRate * dt;
        _deltaActual = Mathf.MoveTowards(_deltaActual, deltaTarget, maxDelta);

        // 3 — first-order acceleration lag  (τ·ȧ + a = a_cmd)
        float a_clamped  = Mathf.Clamp(a_cmd, -maxBrake, maxAccel);
        if (accelTau > 1e-3f)
            _accelActual += (a_clamped - _accelActual) * (dt / accelTau);
        else
            _accelActual  = a_clamped;

        // 4 — aerodynamic + rolling drag (acts opposite to motion)
        float drag    = dragCoeff * _speed;
        float v_dot   = _accelActual - drag;
        _speed        = Mathf.Max(0f, _speed + v_dot * dt);

        // Bicycle yaw rate: dθ/dt = (v/L) * tan(δ)
        float dTheta = (_speed / wheelbase) * Mathf.Tan(_deltaActual) * dt;
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

    // Signed cross-track error to the ego path (map mode) or global X (fallback).
    float LaneLateralError()
    {
        TaxiwayPath egoPath = scenarioManager != null ? scenarioManager.EgoPath : null;
        if (network != null && egoPath != null)
            return network.GetRelativeState(transform.position, transform.forward, egoPath).d;
        return transform.position.x;
    }

    // Heading error vs path tangent (map mode) or global heading (fallback).
    float LaneHeadingError()
    {
        TaxiwayPath egoPath = scenarioManager != null ? scenarioManager.EgoPath : null;
        if (network != null && egoPath != null)
            return network.GetRelativeState(transform.position, transform.forward, egoPath).thetaError;
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
