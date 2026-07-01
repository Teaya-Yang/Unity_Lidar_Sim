using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// Scenario types pushed by Python via "scenario_type" side channel.
/// </summary>
public enum ScenarioType
{
    Standard    = 0,   // difficulty-based layout, perpendicular crossing
    Stationary  = 1,   // one agent placed on the taxiway, never moves
    HeadOn      = 2,   // agents travel along taxiway axis (±Z) toward airplane
    HighSpeed   = 3,   // standard layout, airplane desired_speed pushed separately
    Accelerating= 4,   // agents start slow and accelerate into the conflict zone
}

/// <summary>
/// Centralized per-episode scenario configurator.
///
/// SETUP:
///   - Assign one incursionPrefab (IncursionAgentController + Rigidbody + Collider).
///   - Assign one or more conflictPoints (empty GameObjects at taxiway intersections).
///   - Set maxAgents (copies pre-instantiated at Awake).
///
/// Python pushes "scenario_type" (int cast to float) each episode to select the
/// active scenario. Within each type, "difficulty" further controls agent count/modes.
/// </summary>
public class TaxiScenarioManager : MonoBehaviour
{
    [Header("Prefab")]
    public GameObject incursionPrefab;
    public int        maxAgents = 3;

    [Header("Conflict points — one per taxiway intersection")]
    public List<Transform> conflictPoints = new List<Transform>();

    [Header("Defaults")]
    public float defaultDifficulty     = 0.0f;
    public float defaultIncursionDt    = 0.0f;
    public float defaultAmbulanceSpeed = 5.0f;

    [Header("Spawn randomisation")]
    public float conflictZJitter         = 10f;
    public bool  randomiseCrossDirection = true;
    public float lateralSpread           = 3f;
    public float speedVariation          = 0.05f;

    [Header("Compound conflicts (Lever 2)")]
    [Tooltip("Enable multi-agent conflicts that share a timing window so avoiding one " +
             "obstacle forces the ego toward another. When off, secondary agents are " +
             "staggered 1.5 s apart (ambient traffic, the legacy behaviour).")]
    public bool  compoundConflicts   = true;
    [Tooltip("Difficulty at/above which secondary agents converge on the conflict window " +
             "instead of being staggered out of it.")]
    public float compoundDifficulty  = 0.7f;
    [Tooltip("Half-width of the converging arrival window [s]. Secondary agents arrive within " +
             "±this of the ego, from alternating sides, creating a genuine go/stop dilemma.")]
    public float compoundDtWindow     = 0.6f;

    [Header("GeoJSON map integration (optional)")]
    [Tooltip("Assign to enable map-based ego spawning and path-following obstacles. " +
             "Leave null to use the legacy conflict-point layout.")]
    public TaxiwayNetwork network;
    [Tooltip("Skip ego paths whose sharpest corner exceeds this [deg]. The aircraft cannot " +
             "steer through tight corners at taxi speed; such paths make it leave the lane and stall.")]
    public float maxEgoTurnDeg    = 35f;
    [Tooltip("Skip ego paths shorter than this [m] so episodes have room to run.")]
    public float minEgoPathLength = 60f;
    [Tooltip("Never spawn an obstacle closer than this [m] to the ego at episode start.")]
    public float minObstacleSpawnDist = 15f;
    [Tooltip("How far ahead (in seconds of ego travel) a conflict point may be and still " +
             "be reachable within the episode. Lower values force agents to spawn close " +
             "to the ego so they actually arrive during the episode. Overridden per-episode " +
             "by Python via the 'episode_reach_seconds' side-channel parameter.")]
    public float episodeReachSeconds = 20f;

    // ── public read-only ───────────────────────────────────────────────────────
    public IReadOnlyList<IncursionAgentController> ActiveAgents => _active;

    // The path assigned to the ego this episode (null when network is not used).
    public TaxiwayPath EgoPath { get; private set; }

    // ── internal ───────────────────────────────────────────────────────────────
    readonly List<IncursionAgentController> _pool   = new List<IncursionAgentController>();
    readonly List<IncursionAgentController> _active = new List<IncursionAgentController>();

    // ── Awake ──────────────────────────────────────────────────────────────────

    void Awake()
    {
        if (incursionPrefab == null || conflictPoints == null || conflictPoints.Count == 0)
        {
            Debug.LogError("[TaxiScenarioManager] incursionPrefab or conflictPoints not assigned.", this);
            return;
        }
        for (int i = 0; i < maxAgents; i++)
        {
            var go   = Instantiate(incursionPrefab, Vector3.zero, Quaternion.identity, transform);
            go.name  = $"IncursionAgent_{i}";
            go.SetActive(false);
            var ctrl = go.GetComponent<IncursionAgentController>();
            if (ctrl == null) { Destroy(go); continue; }
            _pool.Add(ctrl);
        }
        Debug.Log($"[TaxiScenarioManager] {_pool.Count} agents, {conflictPoints.Count} conflict point(s).");
    }

    // ── per-episode reset ──────────────────────────────────────────────────────

    public void ResetEpisode(float difficulty,
                             float aircraftSpeed,
                             Transform aircraftTransform,
                             float baseDt,
                             float ambulanceSpeed,
                             float conflictZOffset  = float.NaN,
                             float crossDirSign     = 0f,
                             int   scenarioType     = 0,
                             float headOnProb       = 0f)
    {
        _active.Clear();
        foreach (var a in _pool) a.gameObject.SetActive(false);
        EgoPath = null;

        // ── Map-based episode (network assigned and has paths) ─────────────────
        if (network != null && network.Paths.Count > 0)
        {
            ResetEpisodeFromNetwork(difficulty, aircraftSpeed, aircraftTransform, baseDt,
                                    ambulanceSpeed, scenarioType);
            return;
        }

        if (_pool.Count == 0 || conflictPoints == null || conflictPoints.Count == 0) return;

        // ── Z jitter ──────────────────────────────────────────────────────────
        float zOffset = float.IsNaN(conflictZOffset)
            ? Random.Range(-conflictZJitter, conflictZJitter)
            : conflictZOffset;

        // ── Crossing direction ─────────────────────────────────────────────────
        float dirSign = crossDirSign != 0f
            ? Mathf.Sign(crossDirSign)
            : (randomiseCrossDirection && Random.value < 0.5f ? -1f : 1f);

        // ── Layout from scenario type ──────────────────────────────────────────
        int              nActive;
        TrajectoryMode[] modes;
        ResolveLayout((ScenarioType)scenarioType, difficulty, _pool.Count, out nActive, out modes);

        for (int i = 0; i < nActive; i++)
        {
            var agent = _pool[i];
            agent.gameObject.SetActive(true);

            // Assign conflict point round-robin
            Transform cp          = conflictPoints[i % conflictPoints.Count];
            Vector3   effectiveCP = cp.position + new Vector3(0f, 0f, zOffset);
            agent.conflictPoint   = cp;

            float egoTtc = Mathf.Max(0f,
                (effectiveCP.z - aircraftTransform.position.z) / Mathf.Max(1f, aircraftSpeed));

            float dtOffset = CompoundDtOffset(baseDt, i, difficulty);
            float spd      = ambulanceSpeed * (1f + i * speedVariation);

            // ── Head-on scenario: some agents travel along taxiway axis (±Z) ──
            Vector3 baseDir;
            bool isHeadOn = (ScenarioType)scenarioType == ScenarioType.HeadOn
                            || Random.value < headOnProb;

            if (isHeadOn)
            {
                // Head-on: travel along -Z (toward airplane) or +Z (away)
                baseDir = dirSign < 0f ? -Vector3.forward : Vector3.forward;
            }
            else
            {
                float agentSign = (i % 2 == 0) ? dirSign : -dirSign;
                baseDir = agent.CrossDirectionNormalized * agentSign;
            }

            // ── Stationary: place directly on taxiway centreline ───────────────
            Vector3 start;
            if (modes[i] == TrajectoryMode.Stationary)
            {
                // Place the stationary obstacle at the effective conflict point
                start = effectiveCP;
            }
            else
            {
                Vector3 lateralAxis = Vector3.Cross(baseDir, Vector3.up).normalized;
                float   lateralOff  = (i - (nActive - 1) * 0.5f) * lateralSpread;
                start = effectiveCP - baseDir * spd * (egoTtc + dtOffset)
                      + lateralAxis * lateralOff;
            }

            agent.crossDirection = baseDir;
            agent.trajectoryMode = modes[i];
            agent.ResetCrossing(start, spd);
            _active.Add(agent);

            Debug.Log($"[TaxiScenarioManager] Agent {i}: {modes[i]} cp='{cp.name}' " +
                      $"headOn={isHeadOn} spd={spd:F1} dtOff={dtOffset:+.1f}");
        }

        Debug.Log($"[TaxiScenarioManager] reset type={(ScenarioType)scenarioType} " +
                  $"diff={difficulty:F2} n={nActive} zOff={zOffset:+.1f}");
    }

    // ── Map-based (GeoJSON network) episode reset ─────────────────────────────

    void ResetEpisodeFromNetwork(float difficulty, float aircraftSpeed,
                                  Transform aircraftTransform, float baseDt,
                                  float ambulanceSpeed, int scenarioType)
    {
        var paths = network.Paths;

        // Pick a random NAVIGABLE path for the ego. Constraints:
        //   • must be a taxiway (not a runway or apron — those are wrong geometry/scale
        //     for a 8 m/s taxi sim: runways are 3.6 km straights, aprons are area perimeters);
        //   • corners must be gentle enough for the aircraft to steer through;
        //   • long enough to be a meaningful route.
        var navigable = new List<int>();
        for (int pi = 0; pi < paths.Count; pi++)
            if (paths[pi].IsTaxiway &&
                paths[pi].MaxTurnDeg <= maxEgoTurnDeg &&
                paths[pi].TotalLength >= minEgoPathLength)
                navigable.Add(pi);

        int egoPathIdx = navigable.Count > 0
            ? navigable[Random.Range(0, navigable.Count)]
            : Random.Range(0, paths.Count);   // fallback: nothing qualifies
        EgoPath = paths[egoPathIdx];
        var egoWps = EgoPath.Waypoints;
        if (egoWps.Count > 0)
        {
            aircraftTransform.position    = egoWps[0];
            Vector3 initDir = egoWps.Count > 1
                ? (egoWps[1] - egoWps[0]).normalized
                : Vector3.forward;
            if (initDir.sqrMagnitude > 1e-6f)
                aircraftTransform.rotation = Quaternion.LookRotation(initDir, Vector3.up);
        }

        // Determine how many obstacle agents to activate
        int nActive;
        TrajectoryMode[] modes;
        ResolveLayout((ScenarioType)scenarioType, difficulty, _pool.Count, out nActive, out modes);

        // Ego's current arc position and how far it can travel this episode.
        float egoArcStart  = network.GetRelativeState(
                                 aircraftTransform.position, Vector3.forward, EgoPath).s;
        float reachableArc = aircraftSpeed * episodeReachSeconds;

        // Collect candidate intersecting paths (exclude the ego's own path), keeping the
        // ego-arc of each conflict point so we can reject ones the ego can't reach in time.
        var candidatePaths = new List<(TaxiwayPath path, Vector3 intersection, float egoArc)>();
        for (int pi = 0; pi < paths.Count; pi++)
        {
            if (pi == egoPathIdx) continue;
            if (!network.TryFindIntersection(EgoPath, paths[pi], out Vector3 ix)) continue;

            float egoArc  = network.GetRelativeState(ix, Vector3.forward, EgoPath).s;
            float ahead   = egoArc - egoArcStart;
            // Only keep conflicts ahead of the ego and reachable before timeout.
            if (ahead < minObstacleSpawnDist || ahead > reachableArc) continue;
            candidatePaths.Add((paths[pi], ix, egoArc));
        }

        // Nearest reachable conflicts first, so the assigned obstacles reliably produce an
        // in-window encounter rather than a fly-by the ego never reaches.
        candidatePaths.Sort((a, b) => a.egoArc.CompareTo(b.egoArc));

        if (candidatePaths.Count < nActive)
            Debug.Log($"[TaxiScenarioManager] (map) only {candidatePaths.Count} reachable " +
                      $"conflict(s) within {reachableArc:F0}m for {nActive} agent(s).");

        for (int i = 0; i < nActive; i++)
        {
            var agent = _pool[i];
            agent.gameObject.SetActive(true);

            TaxiwayPath obsPath;
            Vector3 conflictPt;
            float   egoArcConflict;

            bool compound = compoundConflicts && difficulty >= compoundDifficulty;

            if (compound && i > 0 && candidatePaths.Count > 0)
            {
                // Lever 2 co-location: secondary arrives at the PRIMARY conflict point,
                // timed within ±compoundDtWindow so the ego can't resolve both sequentially.
                //
                // Fix C — head-on special case: for a head-on primary, candidatePaths[0]
                // is the counterflow (same-axis) path. A second head-on agent is a convoy,
                // not a trap — the ego clears both at once. Instead, prefer candidatePaths[1]
                // (a LATERAL crossing path, different axis through the same conflict point)
                // so evading the oncoming agent steers the ego into a crosser.
                bool isHeadOn = (ScenarioType)scenarioType == ScenarioType.HeadOn;
                if (isHeadOn && candidatePaths.Count > 1)
                    (obsPath, conflictPt, egoArcConflict) = candidatePaths[1];
                else
                    (obsPath, conflictPt, egoArcConflict) = candidatePaths[0];
            }
            else if (i < candidatePaths.Count)
            {
                (obsPath, conflictPt, egoArcConflict) = candidatePaths[i];
            }
            else
            {
                // No reachable intersecting path for this slot — fall back to a random path.
                // (This agent likely won't produce a conflict; it pads the obstacle count.)
                obsPath    = paths[Random.Range(0, paths.Count)];
                conflictPt = obsPath.Waypoints.Count > 0
                    ? obsPath.Waypoints[obsPath.Waypoints.Count / 2]
                    : Vector3.zero;
                egoArcConflict = network.GetRelativeState(conflictPt, Vector3.forward, EgoPath).s;
            }

            // Time for the EGO to reach the intersection: arc-length from the ego's current
            // position to the conflict point, along the EGO path — NOT to the path end.
            float egoTtc = Mathf.Max(0f,
                (egoArcConflict - egoArcStart) / Mathf.Max(1f, aircraftSpeed));

            float dtOffset = CompoundDtOffset(baseDt, i, difficulty);
            float spd      = ambulanceSpeed * (1f + i * speedVariation);

            PathState obsState = network.GetRelativeState(conflictPt, Vector3.forward, obsPath);

            // Where to place the obstacle along its own path (arc-length).
            float obsArc;
            if (modes[i] == TrajectoryMode.Stationary)
                // Disabled vehicle / FOD blocking the ego's taxiway: park it ON the
                // conflict point (which lies on the EGO path) so the ego must detect
                // it and stop — not off on a side taxiway, where the upstream offset
                // below would otherwise strand a never-moving obstacle.
                obsArc = obsState.s;
            else
                // Moving obstacle: place UPSTREAM so it reaches the conflict point at
                // the same time as the ego (offset by dtOffset for staggering).
                obsArc = obsState.s - spd * (egoTtc + dtOffset);
            obsArc = Mathf.Clamp(obsArc, 0f, obsPath.TotalLength);

            // Walk the arc to find world position
            Vector3 spawnPos = ArcToWorldPosition(obsPath, obsArc);

            // Guard: never spawn an obstacle on top of the ego. If the computed spawn is
            // too close, push it upstream along its path; if still too close, skip this agent.
            if (Vector3.Distance(spawnPos, aircraftTransform.position) < minObstacleSpawnDist)
            {
                float backedArc = Mathf.Clamp(obsArc - minObstacleSpawnDist * 2f, 0f, obsPath.TotalLength);
                spawnPos = ArcToWorldPosition(obsPath, backedArc);
                if (Vector3.Distance(spawnPos, aircraftTransform.position) < minObstacleSpawnDist)
                {
                    agent.gameObject.SetActive(false);
                    Debug.Log($"[TaxiScenarioManager] (map) Agent {i}: skipped (too close to ego).");
                    continue;
                }
            }

            agent.assignedPath   = obsPath;
            agent.trajectoryMode = modes[i];
            agent.crossDirection  = obsState.tangent;
            agent.ResetCrossing(spawnPos, spd);
            _active.Add(agent);

            Debug.Log($"[TaxiScenarioManager] (map) Agent {i}: mode={modes[i]} " +
                      $"spd={spd:F1} arc={obsArc:F0}m");
        }

        Debug.Log($"[TaxiScenarioManager] (map) reset egoPath={egoPathIdx} n={nActive}");
    }

    // Returns the world position at a given arc-length along a path.
    static Vector3 ArcToWorldPosition(TaxiwayPath path, float arc)
    {
        arc = Mathf.Clamp(arc, 0f, path.TotalLength);
        var  cl  = path.CumulativeLength;
        var  wps = path.Waypoints;
        for (int i = 0; i < cl.Length - 1; i++)
        {
            if (arc <= cl[i + 1])
            {
                float segLen = cl[i + 1] - cl[i];
                float t = segLen > 1e-6f ? (arc - cl[i]) / segLen : 0f;
                return Vector3.Lerp(wps[i], wps[i + 1], t);
            }
        }
        return wps[wps.Count - 1];
    }

    // ── compound timing (Lever 2) ───────────────────────────────────────────────

    /// <summary>
    /// Arrival-time offset for agent i. Legacy behaviour staggers each secondary
    /// agent 1.5 s later (ambient traffic that the ego never co-occupies). When
    /// compound conflicts are enabled and difficulty is high enough, secondaries
    /// instead CONVERGE within ±compoundDtWindow of the ego's arrival, from
    /// alternating sides — so evading the primary steers into a secondary.
    /// </summary>
    float CompoundDtOffset(float baseDt, int i, float difficulty)
    {
        bool compound = compoundConflicts && difficulty >= compoundDifficulty;
        if (!compound) return baseDt + i * 1.5f;   // legacy staggered ambient traffic
        if (i == 0)    return baseDt;              // primary arrives with the ego
        float side = (i % 2 == 1) ? 1f : -1f;      // alternate which side arrives first
        int   step = (i + 1) / 2;                  // 1,1,2,2,… grows the window slowly
        return baseDt + side * compoundDtWindow * step;
    }

    // ── layout table ──────────────────────────────────────────────────────────

    static void ResolveLayout(ScenarioType type, float d, int poolSize,
                              out int nActive, out TrajectoryMode[] modes)
    {
        switch (type)
        {
            case ScenarioType.Stationary:
                // Lever 2: at high difficulty, pair the parked blockage with a crosser.
                // The ego's go-around swerve around the stationary obstacle then has to
                // contend with a moving agent crossing the swerve path — the compound
                // "avoid A forces you toward B" dilemma.
                if (d >= 0.7f)
                {
                    nActive = 2;
                    modes   = new[] { TrajectoryMode.Stationary, TrajectoryMode.Straight };
                }
                else
                {
                    nActive = 1;
                    modes   = new[] { TrajectoryMode.Stationary };
                }
                break;

            case ScenarioType.HeadOn:
                // Lever 2: head-on blocker plus a crosser at the same window at high difficulty,
                // so sidestepping the oncoming agent steers into a crossing conflict.
                if (d >= 0.7f)
                {
                    nActive = 2;
                    modes   = new[] { TrajectoryMode.Straight, TrajectoryMode.Straight };
                }
                else
                {
                    nActive = 1;
                    modes   = new[] { TrajectoryMode.Straight };
                }
                break;

            case ScenarioType.Accelerating:
                nActive = d < 0.5f ? 1 : 2;
                modes   = nActive == 1
                    ? new[] { TrajectoryMode.Accelerating }
                    : new[] { TrajectoryMode.Accelerating, TrajectoryMode.Straight };
                break;

            case ScenarioType.HighSpeed:
            case ScenarioType.Standard:
            default:
                // Standard difficulty ramp
                if (d < 0.33f)      { nActive = 1; modes = new[] { TrajectoryMode.Straight }; }
                else if (d < 0.55f) { nActive = 1; modes = new[] { TrajectoryMode.StopGo }; }
                else if (d < 0.70f) { nActive = 2; modes = new[] { TrajectoryMode.Straight, TrajectoryMode.StopGo }; }
                else if (d < 0.85f) { nActive = 2; modes = new[] { TrajectoryMode.Straight, TrajectoryMode.Curved }; }
                else                { nActive = 3; modes = new[] { TrajectoryMode.Straight, TrajectoryMode.Curved, TrajectoryMode.Erratic }; }
                break;
        }

        nActive = Mathf.Min(nActive, poolSize);
        if (modes.Length > nActive)
        {
            var t = new TrajectoryMode[nActive];
            System.Array.Copy(modes, t, nActive); modes = t;
        }
    }

    // ── Gizmos ────────────────────────────────────────────────────────────────

    void OnDrawGizmosSelected()
    {
        if (conflictPoints == null) return;
        for (int i = 0; i < conflictPoints.Count; i++)
        {
            if (conflictPoints[i] == null) continue;
            Gizmos.color = Color.HSVToRGB(i / (float)Mathf.Max(conflictPoints.Count, 1), 0.9f, 1f);
            Gizmos.DrawWireSphere(conflictPoints[i].position, 6f);
            Gizmos.DrawLine(conflictPoints[i].position - Vector3.right * 40f,
                            conflictPoints[i].position + Vector3.right * 40f);
            Gizmos.color = Color.cyan;
            Gizmos.DrawLine(conflictPoints[i].position - Vector3.forward * conflictZJitter,
                            conflictPoints[i].position + Vector3.forward * conflictZJitter);
        }
    }
}
