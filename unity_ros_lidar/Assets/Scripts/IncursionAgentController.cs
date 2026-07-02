using UnityEngine;

/// <summary>
/// Trajectory mode for this incursion agent.
/// </summary>
public enum TrajectoryMode
{
    Straight,    // constant-velocity straight crossing
    Curved,      // constant-speed arc
    StopGo,      // stop near taxiway edge, wait, resume
    Erratic,     // random heading/speed perturbations
    Stationary,  // placed on taxiway and never moves (parked vehicle / FOD)
    Accelerating,// starts slow, accelerates toward conflict point
    FollowPath,  // drive along an assigned TaxiwayPath (GeoJSON map-based scenario)
    HoldShort,   // hold at a runway holding position; if willIncur, drive across when the ego nears
}

/// <summary>
/// Deterministic incursion crosser supporting six trajectory modes.
/// All modes share the same public API: ResetCrossing / StopCrossing / Velocity.
/// </summary>
public class IncursionAgentController : MonoBehaviour
{
    [Header("Crossing motion")]
    public Vector3 crossDirection = Vector3.right;
    public float   crossSpeed     = 5.0f;
    public bool    faceTravelDirection = true;
    public bool    frontIsNegativeZ    = true;

    [Header("Trajectory mode")]
    public TrajectoryMode trajectoryMode = TrajectoryMode.Straight;

    [Header("Curved")]
    public float curveRadius = 20f;

    [Header("StopGo")]
    public float     stopDistanceFromConflict = 6f;
    public float     stopWaitTime             = 2.0f;
    public Transform conflictPoint;

    [Header("Erratic")]
    public float erraticSpeedJitter      = 0.8f;
    public float erraticHeadingJitter    = 0.12f;
    public float erraticUpdateInterval   = 0.5f;

    [Header("Accelerating")]
    [Tooltip("Speed at spawn (fraction of crossSpeed). Agent accelerates to crossSpeed.")]
    public float accelStartFraction = 0.2f;
    [Tooltip("Acceleration rate [m/s²].")]
    public float accelRate          = 1.0f;

    [Header("FollowPath (GeoJSON map mode)")]
    [Tooltip("Set by TaxiScenarioManager. The agent follows these waypoints in order.")]
    public TaxiwayPath assignedPath;
    [Tooltip("Waypoint-reached radius [m].")]
    public float pathWaypointRadius = 3f;
    [Tooltip("Follow the assigned path in REVERSE waypoint order. Used by the head-on scenario " +
             "so an agent placed ahead on the EGO's own path drives back TOWARD the ego.")]
    public bool followReverse = false;
    [Tooltip("Track a line offset this many metres to the RIGHT of travel (perpendicular to the " +
             "path), instead of the centreline. Head-on traffic uses this to hold its own side " +
             "of the taxiway so the ego can pass. 0 = follow the centreline.")]
    public float pathLateralOffset = 0f;
    [Tooltip("Park at the final waypoint instead of driving straight off the end. The follow-" +
             "vehicle task uses this so the lead vehicle stops at the SHARED goal (path end) and " +
             "the ego arrives behind it, rather than the lead clearing out of the scene.")]
    public bool stopAtPathEnd = false;

    [Header("FollowVehicle stochastic events (Task 5)")]
    [Tooltip("While following its path, randomly STOP (brake) or ACCELERATE (speed burst). " +
             "The follow-vehicle scenario enables this so the ego's MPPI must adapt to a lead " +
             "that unpredictably slows or speeds up. Off = plain constant-speed FollowPath.")]
    public bool  stochasticFollow      = false;
    [Tooltip("Per-decision probability [0..1] of triggering a stop/accelerate event. Set from " +
             "difficulty by TaxiScenarioManager — harder episodes make the lead act up more often.")]
    public float followEventProbability = 0f;
    [Tooltip("Seconds between event rolls while cruising.")]
    public float followEventInterval    = 2f;
    [Tooltip("How long a stop (brake) event holds the lead at zero speed [s].")]
    public float followStopDuration      = 2.5f;
    [Tooltip("Speed multiplier applied to the base follow speed during an accelerate burst.")]
    public float followAccelFactor       = 2.5f;
    [Tooltip("How long an accelerate burst lasts [s].")]
    public float followAccelDuration     = 2f;

    [Header("HoldShort / runway incursion (Task 9)")]
    [Tooltip("If true, this agent commits a runway incursion: it holds at its spawn (the holding " +
             "position) and then drives forward across the runway once the ego comes within " +
             "incursionTriggerDist. If false it holds indefinitely (correct hold-short — the ego " +
             "passes safely). Set per-episode by TaxiScenarioManager from the incursion probability.")]
    public bool      willIncur           = false;
    [Tooltip("The ego transform. When willIncur, the hold is released once the ego is within " +
             "incursionTriggerDist of this agent, so the incursion is timed to the ego's approach.")]
    public Transform incursionTrigger    = null;
    [Tooltip("Ego proximity [m] that releases the hold and starts the incursion.")]
    public float     incursionTriggerDist = 35f;

    // ── private ────────────────────────────────────────────────────────────────

    bool    _moving;
    Vector3 _dir;
    float   _speed;        // current speed (may change in Accelerating/Erratic)
    float   _topSpeed;     // target speed for Accelerating mode

    // StopGo
    bool  _stopped;
    float _stopTimer;

    // Erratic
    float _erraticTimer;
    float _erraticSpeedOffset;

    // FollowPath
    int _pathWpIndex;

    // FollowVehicle stochastic events
    enum FollowState { Cruise, Stopped, Accel }
    FollowState _followState;
    float       _followTimer;      // time accumulated toward the next event roll
    float       _followStateTimer; // remaining duration of the active stop/accel event
    float       _followEffSpeed;   // speed actually applied this frame (reported to observers)
    bool        _followParked;     // reached the end of its path (stopAtPathEnd) and holding there

    // HoldShort
    bool        _released;         // incursion started (agent has left the holding position)

    // ── public API ─────────────────────────────────────────────────────────────

    /// True once a stopAtPathEnd agent (the follow-vehicle lead) has reached the end of its
    /// path and parked at the shared goal. Lets the ego count "arrived safely behind the lead"
    /// as episode success instead of deadlocking the final few metres to the goal.
    public bool ReachedPathEnd => _followParked;

    public Vector3 CrossDirectionNormalized =>
        crossDirection.sqrMagnitude > 1e-6f ? crossDirection.normalized : Vector3.right;

    public Vector3 Velocity
    {
        get
        {
            if (!_moving) return Vector3.zero;
            // Follow vehicle: report the effective (possibly braked/boosted) speed so the ego's
            // CBF/MPPI sees the lead actually stop or accelerate.
            if (trajectoryMode == TrajectoryMode.FollowPath && stochasticFollow)
                return _dir * _followEffSpeed;
            // HoldShort: zero while holding at the marker (off the runway), real speed once the
            // incursion is released — so the ego only "sees" motion when the vehicle intrudes.
            if (trajectoryMode == TrajectoryMode.HoldShort)
                return _released ? _dir * _speed : Vector3.zero;
            return _stopped ? Vector3.zero : _dir * (_speed + _erraticSpeedOffset);
        }
    }

    public void ResetCrossing(Vector3 startPos, float speed = -1f)
    {
        transform.position  = startPos;
        _topSpeed           = speed > 0f ? speed : crossSpeed;
        _speed              = trajectoryMode == TrajectoryMode.Accelerating
                                ? _topSpeed * accelStartFraction
                                : _topSpeed;
        _moving             = true;
        _stopped            = false;
        _stopTimer          = 0f;
        _erraticTimer       = 0f;
        _erraticSpeedOffset = 0f;
        _pathWpIndex        = 0;
        _followState        = FollowState.Cruise;
        _followTimer        = 0f;
        _followStateTimer   = 0f;
        _followEffSpeed     = _speed;
        _followParked       = false;
        _released           = false;

        if ((trajectoryMode == TrajectoryMode.FollowPath || trajectoryMode == TrajectoryMode.HoldShort)
            && assignedPath != null
            && assignedPath.Waypoints.Count > 0)
        {
            // Start toward the next waypoint in the travel direction: forward (increasing
            // index) normally, or backward (decreasing index) for a reverse head-on agent.
            _pathWpIndex = followReverse
                ? FindNearestWaypointBehind(startPos)
                : FindNearestWaypointAhead(startPos);
            Vector3 toNext = assignedPath.Waypoints[_pathWpIndex] - startPos;
            toNext.y = 0f;
            _dir = toNext.sqrMagnitude > 1e-6f ? toNext.normalized : Vector3.forward;
        }
        else
        {
            _dir = CrossDirectionNormalized;
        }

        if (faceTravelDirection && _dir.sqrMagnitude > 1e-6f)
        {
            Vector3 faceDir = frontIsNegativeZ ? -_dir : _dir;
            transform.rotation = Quaternion.LookRotation(faceDir, Vector3.up);
        }
    }

    public void StopCrossing() => _moving = false;

    // ── FixedUpdate ────────────────────────────────────────────────────────────

    void FixedUpdate()
    {
        if (!_moving) return;
        switch (trajectoryMode)
        {
            case TrajectoryMode.Straight:     StepStraight();     break;
            case TrajectoryMode.Curved:       StepCurved();       break;
            case TrajectoryMode.StopGo:       StepStopGo();       break;
            case TrajectoryMode.Erratic:      StepErratic();      break;
            case TrajectoryMode.Stationary:                       break; // intentionally idle
            case TrajectoryMode.Accelerating: StepAccelerating(); break;
            case TrajectoryMode.FollowPath:   StepFollowPath();   break;
            case TrajectoryMode.HoldShort:    StepHoldShort();    break;
        }
    }

    // Hold at the runway holding position until released, then drive across the runway.
    // Release condition: willIncur AND the ego is within incursionTriggerDist. A non-incurring
    // agent never releases — it holds short correctly and the ego passes on the runway safely.
    void StepHoldShort()
    {
        if (!_released)
        {
            if (!willIncur || incursionTrigger == null) return;   // hold indefinitely
            Vector3 d = incursionTrigger.position - transform.position; d.y = 0f;
            if (d.sqrMagnitude <= incursionTriggerDist * incursionTriggerDist)
                _released = true;
            else
                return;
        }
        // Released: cross the runway by following the assigned taxiway through the intersection.
        StepFollowPath();
    }

    // ── modes ──────────────────────────────────────────────────────────────────

    void StepStraight()
    {
        transform.position += _dir * (_speed * Time.fixedDeltaTime);
    }

    void StepCurved()
    {
        if (curveRadius > 0.1f)
        {
            float omega = _speed / curveRadius * Time.fixedDeltaTime;
            _dir = Quaternion.AngleAxis(omega * Mathf.Rad2Deg, Vector3.up) * _dir;
        }
        transform.position += _dir * (_speed * Time.fixedDeltaTime);
        if (faceTravelDirection && _dir.sqrMagnitude > 1e-6f)
            transform.rotation = Quaternion.LookRotation(frontIsNegativeZ ? -_dir : _dir, Vector3.up);
    }

    void StepStopGo()
    {
        if (_stopped)
        {
            _stopTimer += Time.fixedDeltaTime;
            if (_stopTimer >= stopWaitTime) { _stopped = false; _stopTimer = 0f; }
            return;
        }
        if (conflictPoint != null)
        {
            float along = Vector3.Dot(conflictPoint.position - transform.position, _dir);
            if (along > 0f && along < stopDistanceFromConflict)
            { _stopped = true; _stopTimer = 0f; return; }
        }
        transform.position += _dir * (_speed * Time.fixedDeltaTime);
    }

    void StepErratic()
    {
        _erraticTimer += Time.fixedDeltaTime;
        if (_erraticTimer >= erraticUpdateInterval)
        {
            _erraticTimer       = 0f;
            _erraticSpeedOffset = Random.Range(-erraticSpeedJitter, erraticSpeedJitter);
            float hdg           = Random.Range(-erraticHeadingJitter, erraticHeadingJitter);
            _dir = Quaternion.AngleAxis(hdg * Mathf.Rad2Deg, Vector3.up) * _dir;
        }
        float eff = Mathf.Max(0f, _speed + _erraticSpeedOffset);
        transform.position += _dir * (eff * Time.fixedDeltaTime);
        if (faceTravelDirection && _dir.sqrMagnitude > 1e-6f)
            transform.rotation = Quaternion.LookRotation(frontIsNegativeZ ? -_dir : _dir, Vector3.up);
    }

    void StepAccelerating()
    {
        _speed = Mathf.Min(_topSpeed, _speed + accelRate * Time.fixedDeltaTime);
        transform.position += _dir * (_speed * Time.fixedDeltaTime);
    }

    void StepFollowPath()
    {
        if (assignedPath == null || assignedPath.Waypoints.Count == 0) return;

        var wps = assignedPath.Waypoints;
        int step = followReverse ? -1 : +1;   // travel direction along the waypoint list

        // Ran off the end of the path (start end when reversing):
        //  • Follow vehicle (stopAtPathEnd): PARK at the shared goal so the ego arrives behind it.
        //  • Everyone else: keep driving straight in the current heading so the agent CLEARS the
        //    area. Parking here (the old default) left a permanent phantom blocker that stops the
        //    ego and deadlocks it at v=0.
        if (PastPathEnd(wps.Count))
        {
            if (stopAtPathEnd) { _followParked = true; _followEffSpeed = 0f; return; }
            DriveStraight();
            return;
        }

        // Aim at the waypoint shifted sideways by pathLateralOffset, so the agent tracks a
        // line PARALLEL to the path (its own side of the taxiway) rather than the centreline.
        Vector3 target = OffsetTarget(wps[_pathWpIndex]);
        Vector3 toTarget = target - transform.position;
        toTarget.y = 0f;

        if (toTarget.sqrMagnitude < pathWaypointRadius * pathWaypointRadius)
        {
            _pathWpIndex += step;
            if (PastPathEnd(wps.Count))
            {
                if (stopAtPathEnd) { _followParked = true; _followEffSpeed = 0f; return; }
                DriveStraight();
                return;
            }
            toTarget = OffsetTarget(wps[_pathWpIndex]) - transform.position;
            toTarget.y = 0f;
        }

        if (toTarget.sqrMagnitude > 1e-6f) _dir = toTarget.normalized;

        // Stochastic stop/accelerate modulation for the follow-vehicle task. Plain FollowPath
        // (head-on, intersection) leaves _followEffSpeed == _speed.
        float eff = FollowEffectiveSpeed();
        transform.position += _dir * (eff * Time.fixedDeltaTime);

        if (faceTravelDirection && _dir.sqrMagnitude > 1e-6f)
            transform.rotation = Quaternion.LookRotation(frontIsNegativeZ ? -_dir : _dir, Vector3.up);
    }

    // Effective follow speed with random stop (brake) / accelerate (burst) events. Probability
    // per roll and durations are configured by TaxiScenarioManager from episode difficulty.
    // Returns _speed unchanged when stochasticFollow is off.
    float FollowEffectiveSpeed()
    {
        if (!stochasticFollow) { _followEffSpeed = _speed; return _speed; }

        switch (_followState)
        {
            case FollowState.Stopped:
                _followStateTimer -= Time.fixedDeltaTime;
                if (_followStateTimer <= 0f) _followState = FollowState.Cruise;
                _followEffSpeed = 0f;
                break;

            case FollowState.Accel:
                _followStateTimer -= Time.fixedDeltaTime;
                if (_followStateTimer <= 0f) _followState = FollowState.Cruise;
                _followEffSpeed = _speed * followAccelFactor;
                break;

            default: // Cruise
                _followTimer += Time.fixedDeltaTime;
                if (_followTimer >= followEventInterval)
                {
                    _followTimer = 0f;
                    if (Random.value < followEventProbability)
                    {
                        // 50/50 stop vs accelerate — the ego can't predict which.
                        if (Random.value < 0.5f)
                        { _followState = FollowState.Stopped; _followStateTimer = followStopDuration; }
                        else
                        { _followState = FollowState.Accel;   _followStateTimer = followAccelDuration; }
                    }
                }
                _followEffSpeed = _followState == FollowState.Stopped ? 0f
                                : _followState == FollowState.Accel   ? _speed * followAccelFactor
                                : _speed;
                break;
        }
        return _followEffSpeed;
    }

    // Shift a centreline waypoint sideways (perpendicular to the current heading) by
    // pathLateralOffset, so FollowPath tracks a line parallel to the path.
    Vector3 OffsetTarget(Vector3 wp)
    {
        if (Mathf.Abs(pathLateralOffset) < 1e-3f) return wp;
        Vector3 perp = Vector3.Cross(Vector3.up, _dir).normalized;   // right of travel
        return wp + perp * pathLateralOffset;
    }

    // Continue in the last heading (used once a FollowPath agent runs off the end of its
    // path — it keeps moving out of the conflict zone instead of parking as a blocker).
    void DriveStraight()
    {
        transform.position += _dir * (_speed * Time.fixedDeltaTime);
    }

    // True when _pathWpIndex has walked off the travel end of the path (past the last
    // waypoint going forward, or before the first going reverse).
    bool PastPathEnd(int count) =>
        followReverse ? _pathWpIndex < 0 : _pathWpIndex >= count;

    // Index of the waypoint nearest to pos, stepped one ahead (forward travel).
    int FindNearestWaypointAhead(Vector3 pos)
    {
        int idx = NearestWaypoint(pos);
        return Mathf.Min(idx + 1, assignedPath.Waypoints.Count - 1);
    }

    // Index of the waypoint nearest to pos, stepped one back (reverse travel).
    int FindNearestWaypointBehind(Vector3 pos)
    {
        int idx = NearestWaypoint(pos);
        return Mathf.Max(idx - 1, 0);
    }

    int NearestWaypoint(Vector3 pos)
    {
        if (assignedPath == null || assignedPath.Waypoints.Count == 0) return 0;
        float best = float.MaxValue;
        int   idx  = 0;
        var   wps  = assignedPath.Waypoints;
        for (int i = 0; i < wps.Count; i++)
        {
            float d = Vector3.Distance(pos, wps[i]);
            if (d < best) { best = d; idx = i; }
        }
        return idx;
    }

    // ── Gizmo ─────────────────────────────────────────────────────────────────

    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.magenta;
        Vector3 d = CrossDirectionNormalized;
        Gizmos.DrawLine(transform.position - d * 20f, transform.position + d * 20f);
        Gizmos.DrawWireSphere(transform.position, 1f);
    }
}
