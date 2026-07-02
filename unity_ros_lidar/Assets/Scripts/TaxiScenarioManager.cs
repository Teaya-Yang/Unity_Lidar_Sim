using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Serialization;

/// <summary>
/// Scenario types pushed by Python via "scenario_type" side channel.
/// </summary>
// Kept in sync with Python's SCENARIO_* ids in taxi_controller.py (pushed via the
// 'scenario_type' side channel). IDs are contiguous; removed scenarios (stationary,
// highspeed, accelerating, recovery, blind-corner) will be rebuilt with new ids later.
public enum ScenarioType
{
    Standard        = 0,   // difficulty-based layout, perpendicular crossing
    HeadOn          = 1,   // agents travel along the taxiway axis toward the airplane
    FollowVehicle   = 2,   // follow a vehicle to the SAME goal; the lead randomly stops/accelerates (difficulty-scaled)
    Intersection    = 3,   // ego on one taxiway branch, ONE agent on the crossing branch — meet at the node
    RunwayIncursion = 4,   // ego drives ON a runway toward a goal; a vehicle holds short on a crossing taxiway and may incur
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

    [Header("Head-on (oncoming traffic on the ego's own taxiway)")]
    [Tooltip("Meet point ahead of the ego, along the EGO path, where an oncoming head-on agent " +
             "is timed to converge with the ego [m]. Must be within reachable arc this episode.")]
    public float headOnApproachGap = 50f;
    [Tooltip("Lateral offset [m] the oncoming head-on agent holds to ITS side of the taxiway at " +
             "difficulty 0 (EASY): wide, generous clearance so the ego passes comfortably. Above " +
             "~half of dSafe, ego + agent both holding their side clears the safety distance.")]
    public float headOnLateralOffsetEasy = 4.5f;
    [Tooltip("Lateral offset [m] at difficulty 1 (HARD): tight — approaches a dead-centre head-on " +
             "where the pass barely clears (or fails), forcing decisive avoidance. The per-episode " +
             "offset is lerped between Easy and Hard by difficulty.")]
    public float headOnLateralOffsetHard = 1.5f;

    [Header("Task 5 — Follow Vehicle (follow a lead to the SAME goal)")]
    [Tooltip("Gap ahead of the ego, along the EGO path, where the lead vehicle spawns [m].")]
    [FormerlySerializedAs("leadVehicleGap")]
    public float followVehicleGap   = 40f;
    [Tooltip("Cruising speed of the lead vehicle [m/s]. Kept below the ego's taxi speed so the " +
             "ego catches up and must maintain a safe following gap.")]
    [FormerlySerializedAs("leadVehicleSpeed")]
    public float followVehicleSpeed = 4f;
    [Tooltip("Probability [0..1] per event roll that the lead stops or accelerates, at difficulty 0 " +
             "(EASY): the lead mostly cruises steadily.")]
    public float followEventProbEasy = 0.05f;
    [Tooltip("Probability [0..1] per event roll at difficulty 1 (HARD): the lead frequently brakes " +
             "or lurches forward, so the ego's MPPI must constantly adapt its following distance. " +
             "Kept below 0.5 so the lead still makes net progress and the ego never stalls out.")]
    public float followEventProbHard = 0.45f;
    [Tooltip("Seconds between the lead's stop/accelerate decisions. MUST exceed followStopDuration " +
             "so every stop is followed by a moving window — otherwise stops chain and the lead " +
             "(and the ego behind it) never progresses.")]
    public float followEventInterval = 2.5f;
    [Tooltip("How long a lead stop (brake) event lasts [s]. Keep < followEventInterval.")]
    public float followStopDuration  = 1.2f;
    [Tooltip("Speed multiplier during a lead accelerate burst.")]
    public float followAccelFactor   = 2.0f;
    [Tooltip("How long a lead accelerate burst lasts [s].")]
    public float followAccelDuration = 1.5f;

    [Header("Task 9 — Runway Incursion")]
    [Tooltip("Probability the holding vehicle commits an incursion (drives onto the runway) at " +
             "difficulty 0 (EASY). Otherwise it holds short correctly and the ego passes safely.")]
    public float incursionProbEasy   = 0.15f;
    [Tooltip("Incursion probability at difficulty 1 (HARD). Ramped with difficulty so harder " +
             "episodes are more likely to force the ego to react to a runway incursion.")]
    public float incursionProbHard   = 0.85f;
    [Tooltip("Speed of the incursion vehicle once it leaves the holding position [m/s].")]
    public float incursionSpeed      = 6f;
    [Tooltip("Ego proximity [m] that triggers the incursion (the hold is released).")]
    public float incursionTriggerDist = 40f;
    [Tooltip("How far back along the taxiway from the runway the vehicle holds [m]. Should keep it " +
             "clear of the runway so a non-incurring hold never blocks the ego. Larger = more " +
             "lateral clearance for oblique (high-speed-exit) taxiways.")]
    public float holdShortBackoff    = 25f;
    [Tooltip("Distance the ego spawns before the runway crossing [m]. Sets the reaction window.")]
    public float runwayApproachGap   = 55f;
    [Tooltip("How far past the crossing, along the runway, the ego's goal sits [m]. Bounds the " +
             "episode so the ego doesn't have to taxi the whole (km-long) runway.")]
    public float runwayGoalPastCrossing = 35f;

    // ── public read-only ───────────────────────────────────────────────────────
    public IReadOnlyList<IncursionAgentController> ActiveAgents => _active;

    // The path assigned to the ego this episode (null when network is not used).
    public TaxiwayPath EgoPath { get; private set; }

    // Arc-length along EgoPath where the ego's goal sits. Normally the path end; the runway-
    // incursion scenario sets it just past the crossing so the episode is bounded on a long
    // runway. TaxiAgent reads this for the goal observation and the reached-goal test.
    public float EgoGoalS { get; private set; }

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
        ResolveLayout((ScenarioType)scenarioType, difficulty, _pool.Count, false, out nActive, out modes);

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

        // Runway incursion is geometrically distinct (ego ON a runway, one hold-short vehicle on a
        // crossing taxiway) — handle it in a dedicated routine rather than the shared crosser loop.
        if ((ScenarioType)scenarioType == ScenarioType.RunwayIncursion)
        {
            SetupRunwayIncursion(difficulty, aircraftSpeed, aircraftTransform);
            return;
        }

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

        // Ego path selection. The dedicated Intersection scenario uses intersection-first
        // selection — an ego path guaranteed to cross another taxiway within reach, so the
        // agent lands on the other branch and a real crossing conflict is certain. All other
        // scenarios keep the plain random-navigable pick (their obstacle logic differs).
        bool intersectionScenario = (ScenarioType)scenarioType == ScenarioType.Intersection;
        int  egoPathIdx;
        if (navigable.Count == 0)
            egoPathIdx = Random.Range(0, paths.Count);          // fallback: nothing qualifies
        else if (intersectionScenario)
            egoPathIdx = SelectEgoPathWithIntersection(navigable, aircraftSpeed);
        else
            egoPathIdx = navigable[Random.Range(0, navigable.Count)];
        EgoPath  = paths[egoPathIdx];
        EgoGoalS = EgoPath.TotalLength;          // goal at the path end (runway incursion overrides)
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
        ResolveLayout((ScenarioType)scenarioType, difficulty, _pool.Count, true, out nActive, out modes);

        // Ego's current arc position and how far it can travel this episode.
        float egoArcStart  = network.GetRelativeState(
                                 aircraftTransform.position, Vector3.forward, EgoPath).s;
        float reachableArc = aircraftSpeed * episodeReachSeconds;

        // Collect candidate intersecting paths (exclude the ego's own path), keeping the
        // ego-arc of each conflict point so we can reject ones the ego can't reach in time.
        // For the Intersection scenario the crosser must be on ANOTHER TAXIWAY: a FollowPath
        // agent assigned to a runway (a 3.6 km straight) or an apron (a closed polygon
        // outline) would trace that wrong geometry instead of crossing the taxiway, producing
        // a nonsensical path and no real conflict. (Other scenarios keep any intersecting
        // feature — their agents cross straight and don't follow the path.)
        bool xsnScenario = (ScenarioType)scenarioType == ScenarioType.Intersection;
        var candidatePaths = new List<(TaxiwayPath path, Vector3 intersection, float egoArc)>();
        for (int pi = 0; pi < paths.Count; pi++)
        {
            if (pi == egoPathIdx) continue;
            if (xsnScenario && !paths[pi].IsTaxiway) continue;   // crosser must be a taxiway
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
            bool isFollow = (ScenarioType)scenarioType == ScenarioType.FollowVehicle;
            bool isXsn    = (ScenarioType)scenarioType == ScenarioType.Intersection;
            bool isHeadOn = (ScenarioType)scenarioType == ScenarioType.HeadOn;

            if (isFollow)
            {
                // Task 5: the lead is a vehicle on the ego's OWN path, ahead, heading to the SAME
                // goal (the path end). No intersecting path is involved — spawn it followVehicleGap
                // metres up the ego route and let it drive forward (FollowPath) so the ego catches
                // up and must keep a safe following gap as the lead randomly stops/accelerates.
                obsPath        = EgoPath;
                egoArcConflict = egoArcStart + followVehicleGap;
                conflictPt     = ArcToWorldPosition(EgoPath, egoArcConflict);
            }
            else if (isHeadOn)
            {
                // Head-on: oncoming traffic on the EGO's OWN path. The meet point is
                // headOnApproachGap ahead of the ego; the agent is spawned FURTHER ahead
                // (below) and drives back along the path (FollowPath reverse) toward the ego.
                obsPath        = EgoPath;
                egoArcConflict = egoArcStart + headOnApproachGap;
                conflictPt     = ArcToWorldPosition(EgoPath, egoArcConflict);
            }
            else if (isXsn && candidatePaths.Count > 0)
            {
                // Intersection: give each agent a DISTINCT crossing branch when the ego route
                // has several reachable ones (cross traffic from multiple directions); once the
                // branches run out, stack the extras on the primary node so the intersection
                // just gets busier with difficulty — never a random-path dud.
                int idx = Mathf.Min(i, candidatePaths.Count - 1);
                (obsPath, conflictPt, egoArcConflict) = candidatePaths[idx];
            }
            else if (compound && i > 0 && candidatePaths.Count > 0)
            {
                // Lever 2 co-location: secondary arrives at the PRIMARY conflict point,
                // timed within ±compoundDtWindow so the ego can't resolve both sequentially.
                // (Head-on is handled earlier — oncoming traffic on the ego path — so it never
                // reaches here.)
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
            float spd      = isFollow
                ? followVehicleSpeed                     // lead vehicle cruise speed (follow task)
                : ambulanceSpeed * (1f + i * speedVariation);

            PathState obsState = network.GetRelativeState(conflictPt, Vector3.forward, obsPath);

            // Where to place the obstacle along its own path (arc-length).
            float obsArc;
            if (modes[i] == TrajectoryMode.Stationary || isFollow)
                // Place the obstacle AT the conflict point rather than upstream of it.
                //  • Stationary: a disabled vehicle / FOD parked on the ego's taxiway, so
                //    the ego must detect it and stop (not stranded off on a side taxiway).
                //  • Follow vehicle (FollowPath, Task 5): spawn it followVehicleGap ahead on the
                //    ego's own path, then it drives forward to the shared goal and the ego follows.
                obsArc = obsState.s;
            else if (isHeadOn)
                // Head-on: spawn AHEAD of the meet point (higher arc) so that, driving BACK
                // toward the ego at spd, it reaches the meet point exactly when the ego does.
                obsArc = obsState.s + spd * (egoTtc + dtOffset);
            else
                // Moving obstacle (incl. an Intersection FollowPath crosser, which then drives
                // FORWARD along the crossing taxiway to the node): place UPSTREAM so it reaches
                // the conflict point when the ego does (offset by dtOffset for staggering).
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

            // Head-on agents hold their own side of the taxiway: track a line offset laterally
            // from the centreline (persisted via pathLateralOffset), and spawn already on that
            // side so the offset doesn't have to be "steered into". Everyone else follows the
            // centreline. The offset shrinks with difficulty (wide/easy pass at diff 0 → tight,
            // near-centre head-on at diff 1). Set explicitly every episode (no stale state).
            float lateralOff = isHeadOn
                ? Mathf.Lerp(headOnLateralOffsetEasy, headOnLateralOffsetHard, difficulty)
                : 0f;
            if (isHeadOn && Mathf.Abs(lateralOff) > 1e-3f)
            {
                Vector3 travelDir = -obsState.tangent;                       // reverse travel
                Vector3 perp = Vector3.Cross(Vector3.up, travelDir).normalized; // right of travel
                spawnPos += perp * lateralOff;
            }

            agent.assignedPath      = obsPath;
            agent.trajectoryMode    = modes[i];
            agent.crossDirection    = obsState.tangent;
            // Head-on agents traverse the ego path in reverse (toward the ego); everyone else
            // forward. Set explicitly every episode so a pooled agent never keeps a stale flag.
            agent.followReverse     = isHeadOn;
            agent.pathLateralOffset = lateralOff;

            // Follow-vehicle task: the lead shares the ego's goal (parks at path end) and randomly
            // stops/accelerates with a probability that ramps with difficulty. Reset the flags every
            // episode (agents are pooled) so a non-follow agent never inherits stale event state.
            agent.stochasticFollow = isFollow;
            agent.stopAtPathEnd    = isFollow;
            if (isFollow)
            {
                agent.followEventProbability = Mathf.Lerp(followEventProbEasy, followEventProbHard, difficulty);
                agent.followEventInterval    = followEventInterval;
                agent.followStopDuration     = followStopDuration;
                agent.followAccelFactor      = followAccelFactor;
                agent.followAccelDuration    = followAccelDuration;
            }

            agent.ResetCrossing(spawnPos, spd);
            _active.Add(agent);

            Debug.Log($"[TaxiScenarioManager] (map) Agent {i}: mode={modes[i]} " +
                      $"spd={spd:F1} arc={obsArc:F0}m");
        }

        Debug.Log($"[TaxiScenarioManager] (map) reset egoPath={egoPathIdx} n={nActive}");
    }

    // ── Runway incursion (Task 9) ─────────────────────────────────────────────
    //
    // The ego drives ALONG a runway toward a goal just past a taxiway crossing. A single vehicle
    // waits at the holding position on that crossing taxiway; with a difficulty-scaled probability
    // it commits an incursion — driving across the runway as the ego approaches — otherwise it
    // holds short and the ego passes safely.
    void SetupRunwayIncursion(float difficulty, float aircraftSpeed, Transform aircraftTransform)
    {
        var paths = network.Paths;

        // Collect every (runway, crossing-taxiway, intersection) triple where the crossing sits far
        // enough from the runway start to spawn the ego before it, and far enough from the end to
        // place the goal past it. Runways must be long enough to hold the whole approach+goal span.
        float minRunwayLen = runwayApproachGap + runwayGoalPastCrossing + minObstacleSpawnDist + 20f;
        var candidates = new List<(int runwayIdx, int taxiIdx, Vector3 ix, float ixArc)>();
        for (int ri = 0; ri < paths.Count; ri++)
        {
            if (!paths[ri].IsRunway || paths[ri].TotalLength < minRunwayLen) continue;
            for (int ti = 0; ti < paths.Count; ti++)
            {
                if (ti == ri || !paths[ti].IsTaxiway) continue;
                if (!network.TryFindIntersection(paths[ri], paths[ti], out Vector3 ix)) continue;
                float ixArc = network.GetRelativeState(ix, Vector3.forward, paths[ri]).s;
                if (ixArc < runwayApproachGap + minObstacleSpawnDist) continue;
                if (ixArc > paths[ri].TotalLength - runwayGoalPastCrossing) continue;
                candidates.Add((ri, ti, ix, ixArc));
            }
        }

        if (candidates.Count == 0)
        {
            Debug.LogWarning("[TaxiScenarioManager] (runway incursion) no runway with a reachable " +
                             "taxiway crossing found — placing ego on a runway with no incursion.");
            int rIdx = 0;
            for (int ri = 0; ri < paths.Count; ri++) if (paths[ri].IsRunway) { rIdx = ri; break; }
            EgoPath  = paths[rIdx];
            EgoGoalS = EgoPath.TotalLength;
            PlaceEgoOnPath(aircraftTransform, EgoPath, 0f);
            return;
        }

        var pick    = candidates[Random.Range(0, candidates.Count)];
        var runway  = paths[pick.runwayIdx];
        var taxiway = paths[pick.taxiIdx];

        // Ego: spawn on the runway runwayApproachGap before the crossing, heading toward it, with
        // the goal a short way past the crossing so the episode is bounded on a km-long runway.
        EgoPath  = runway;
        float egoStartArc = Mathf.Max(0f, pick.ixArc - runwayApproachGap);
        EgoGoalS = Mathf.Min(runway.TotalLength, pick.ixArc + runwayGoalPastCrossing);
        PlaceEgoOnPath(aircraftTransform, runway, egoStartArc);

        // Holding position on the taxiway, short of the runway. Prefer a real holding-position
        // marker near the crossing (snapped to the taxiway centreline); else back off geometrically.
        float ixTaxiArc = network.GetRelativeState(pick.ix, Vector3.forward, taxiway).s;
        float holdArc;
        if (network.TryNearestHoldingPosition(pick.ix, holdShortBackoff * 2.5f, out HoldingPosition hp))
        {
            PathState hs = network.GetRelativeState(hp.Position, Vector3.forward, taxiway);
            // Only trust the marker if it actually lies on this taxiway and short of the runway.
            holdArc = (Mathf.Abs(hs.d) < 12f && Mathf.Abs(hs.s - ixTaxiArc) > 3f)
                ? hs.s
                : (ixTaxiArc >= holdShortBackoff ? ixTaxiArc - holdShortBackoff : ixTaxiArc + holdShortBackoff);
        }
        else
        {
            holdArc = ixTaxiArc >= holdShortBackoff ? ixTaxiArc - holdShortBackoff
                                                    : ixTaxiArc + holdShortBackoff;
        }
        holdArc = Mathf.Clamp(holdArc, 0f, taxiway.TotalLength);
        // Travel forward (increasing waypoint index) crosses the runway when the hold is on the
        // low-arc side; hold beyond the crossing → drive in reverse to cross back over it.
        bool    reverse  = holdArc > ixTaxiArc;
        Vector3 spawnPos = ArcToWorldPosition(taxiway, holdArc);

        if (_pool.Count == 0)
        {
            Debug.LogWarning("[TaxiScenarioManager] (runway incursion) no agent in pool — ego placed, no incursion.");
            return;
        }

        var agent = _pool[0];
        agent.gameObject.SetActive(true);
        agent.assignedPath         = taxiway;
        agent.trajectoryMode       = TrajectoryMode.HoldShort;
        agent.followReverse        = reverse;
        agent.pathLateralOffset    = 0f;
        agent.stochasticFollow     = false;
        agent.stopAtPathEnd        = false;
        agent.willIncur            = Random.value < Mathf.Lerp(incursionProbEasy, incursionProbHard, difficulty);
        agent.incursionTrigger     = aircraftTransform;
        agent.incursionTriggerDist = incursionTriggerDist;
        agent.crossDirection       = network.GetRelativeState(pick.ix, Vector3.forward, taxiway).tangent;
        agent.ResetCrossing(spawnPos, incursionSpeed);
        _active.Add(agent);

        Debug.Log($"[TaxiScenarioManager] (runway incursion) runway='{runway.Ref}' " +
                  $"taxiway='{taxiway.Ref}' ixArc={pick.ixArc:F0} willIncur={agent.willIncur} " +
                  $"holdArc={holdArc:F0} reverse={reverse}");
    }

    // Place the ego at a given arc-length along a path, headed toward increasing arc.
    void PlaceEgoOnPath(Transform ego, TaxiwayPath path, float arc)
    {
        ego.position = ArcToWorldPosition(path, arc);
        Vector3 tan = network.GetRelativeState(ego.position, Vector3.forward, path).tangent;
        if (tan.sqrMagnitude > 1e-6f)
            ego.rotation = Quaternion.LookRotation(tan, Vector3.up);
    }

    // ── Intersection-first ego path selection ─────────────────────────────────
    //
    // Picks an ego path that genuinely crosses another taxiway within reach this episode,
    // so the crossing agent can be placed on the OTHER branch of that intersection and a
    // real conflict is guaranteed. Navigable paths are tried in random order and the first
    // with a reachable taxiway crossing is returned; if none qualify (rare — a taxiway with
    // no reachable neighbour), it falls back to a random navigable path.
    int SelectEgoPathWithIntersection(List<int> navigable, float aircraftSpeed)
    {
        var   paths        = network.Paths;
        float reachableArc = aircraftSpeed * episodeReachSeconds;

        // Fisher–Yates shuffle (Unity RNG, so it honours the per-episode seed).
        var order = new List<int>(navigable);
        for (int k = order.Count - 1; k > 0; k--)
        {
            int j   = Random.Range(0, k + 1);
            int tmp = order[k]; order[k] = order[j]; order[j] = tmp;
        }

        foreach (int egoIdx in order)
        {
            TaxiwayPath ego = paths[egoIdx];
            if (ego.Waypoints.Count == 0) continue;
            float startArc = network.GetRelativeState(ego.Waypoints[0], Vector3.forward, ego).s;

            for (int pi = 0; pi < paths.Count; pi++)
            {
                if (pi == egoIdx) continue;
                if (!paths[pi].IsTaxiway) continue;   // crosser must be another taxiway "street"
                if (!network.TryFindIntersection(ego, paths[pi], out Vector3 ix)) continue;

                float ahead = network.GetRelativeState(ix, Vector3.forward, ego).s - startArc;
                if (ahead >= minObstacleSpawnDist && ahead <= reachableArc)
                    return egoIdx;                    // genuine reachable intersection found
            }
        }

        // Nothing had a reachable crossing — keep the episode alive with a random navigable path.
        Debug.Log("[TaxiScenarioManager] (map) no ego path with a reachable intersection — " +
                  "falling back to a random navigable path (episode may be a dud).");
        return navigable[Random.Range(0, navigable.Count)];
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

    static void ResolveLayout(ScenarioType type, float d, int poolSize, bool networkMode,
                              out int nActive, out TrajectoryMode[] modes)
    {
        switch (type)
        {
            case ScenarioType.HeadOn:
                // Oncoming traffic on the EGO's OWN taxiway: agents follow the ego path in
                // REVERSE (driving toward the ego). More oncoming agents at higher difficulty.
                // FollowPath (network) tracks the real taxiway; legacy mode falls back to a
                // Straight agent driving along the taxiway axis.
                TrajectoryMode hoMode = networkMode ? TrajectoryMode.FollowPath
                                                    : TrajectoryMode.Straight;
                nActive = d >= 0.7f ? 2 : 1;
                modes   = new TrajectoryMode[nActive];
                for (int k = 0; k < nActive; k++) modes[k] = hoMode;
                break;

            case ScenarioType.FollowVehicle:
                // One lead vehicle on the ego's OWN path ahead, heading to the SAME goal.
                // FollowPath drives it along the (ego) path; the stochastic stop/accelerate
                // behaviour (configured per-episode from difficulty) forces the ego to adapt.
                nActive = 1;
                modes   = new[] { TrajectoryMode.FollowPath };
                break;

            case ScenarioType.Intersection:
                // The ego path is chosen (intersection-first) to guarantee a crossing branch
                // exists. Agent count ramps with difficulty (like Standard) so the node gets
                // busier — richer, more general crossing conflicts. Every agent FOLLOWS the
                // crossing taxiway through the node (tracks the real, possibly curved geometry).
                // In legacy conflict-point mode there is no path to follow, so fall back to a
                // straight crossing there.
                if      (d < 0.33f) nActive = 1;
                else if (d < 0.66f) nActive = 2;
                else                nActive = 3;

                TrajectoryMode xsnMode = networkMode ? TrajectoryMode.FollowPath
                                                     : TrajectoryMode.Straight;
                modes = new TrajectoryMode[nActive];
                for (int k = 0; k < nActive; k++) modes[k] = xsnMode;
                break;

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
