using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json.Linq;

/// <summary>
/// Runtime state of the ego aircraft relative to a TaxiwayPath (Frenet frame).
///   s         — arc-length from path start [m]
///   d         — signed cross-track error [m]  (+ = left of path direction)
///   thetaError — heading error vs path tangent [rad]
///   tangent    — world-space unit vector along the path at the closest point
/// </summary>
public struct PathState
{
    public float   s;
    public float   d;
    public float   thetaError;
    public Vector3 tangent;
}

/// <summary>
/// A runtime-only taxiway path loaded from GeoJSON (or built programmatically).
/// Not a MonoBehaviour — lives purely in memory, held by TaxiwayNetwork.
/// </summary>
public class TaxiwayPath
{
    public readonly List<Vector3> Waypoints = new List<Vector3>();
    public float[] CumulativeLength { get; private set; }
    public float   TotalLength      => CumulativeLength != null && CumulativeLength.Length > 0
                                        ? CumulativeLength[CumulativeLength.Length - 1] : 0f;

    // GeoJSON feature metadata (OSM aeroway tags). Empty/0 when absent.
    public string AerowayType = "";   // "taxiway", "apron", "runway"
    public string Ref         = "";   // taxiway letter / runway designation, e.g. "F", "10L/28R"
    public string Name        = "";   // human name, e.g. "Boarding Area E"
    public float  Width       = 0f;   // metres, 0 = unknown

    // True only for paths the ego may be assigned to (linear taxi routes).
    public bool IsTaxiway => AerowayType == "taxiway";

    // True for runway centrelines (the ego drives ON one in the runway-incursion scenario).
    public bool IsRunway => AerowayType == "runway";

    // Sharpest turn between consecutive segments [deg]. Used to skip un-navigable
    // paths (corners tighter than the aircraft's steering can follow).
    public float MaxTurnDeg { get; private set; }

    public void Precompute()
    {
        CumulativeLength = new float[Waypoints.Count];
        CumulativeLength[0] = 0f;
        for (int i = 1; i < Waypoints.Count; i++)
            CumulativeLength[i] = CumulativeLength[i - 1]
                                 + Vector3.Distance(Waypoints[i - 1], Waypoints[i]);

        MaxTurnDeg = 0f;
        for (int i = 1; i < Waypoints.Count - 1; i++)
        {
            Vector3 a = (Waypoints[i]     - Waypoints[i - 1]); a.y = 0f;
            Vector3 b = (Waypoints[i + 1] - Waypoints[i]);     b.y = 0f;
            if (a.sqrMagnitude < 1e-6f || b.sqrMagnitude < 1e-6f) continue;
            float ang = Vector3.Angle(a, b);
            if (ang > MaxTurnDeg) MaxTurnDeg = ang;
        }
    }
}

/// <summary>
/// A runway-holding position parsed from a GeoJSON Point (OSM aeroway=holding_position).
/// Marks where a taxiing vehicle must stop short of a runway. The runway-incursion scenario
/// spawns the incursion vehicle here so it either HOLDS (safe) or drives past (incursion).
/// </summary>
public struct HoldingPosition
{
    public Vector3 Position;
    public string  Ref;          // taxiway/runway designation from the "ref" property
    public bool    IsRunwayHold; // holding_position:type == "runway" (vs ILS/intermediate)
}

/// <summary>
/// Parses a GeoJSON file from StreamingAssets and exposes:
///   • A list of TaxiwayPaths (runtime waypoint chains).
///   • GetRelativeState() — Frenet-frame position/heading error for the ego.
///   • TryFindIntersection() — closest crossing point between two paths.
///
/// Coordinate mapping: GeoJSON lon/lat → flat-earth Mercator → Unity (X=East, Z=North).
/// Set originLat / originLon to the map's geographic centre so local coords stay small.
///
/// If no GeoJSON file is present the network starts empty; callers must null-check
/// or use the fallback straight-line path in TaxiAgent.
/// </summary>
public class TaxiwayNetwork : MonoBehaviour
{
    [Header("GeoJSON source")]
    [Tooltip("File name relative to Application.streamingAssetsPath (e.g. 'taxiway.geojson').")]
    public string geoJsonFileName = "taxiway.geojson";

    [Header("Geographic origin (map centre)")]
    [Tooltip("Latitude of the coordinate origin. All positions are relative to this.")]
    public double originLat = 0.0;
    [Tooltip("Longitude of the coordinate origin.")]
    public double originLon = 0.0;

    [Header("Intersection detection")]
    [Tooltip("Two path segments are considered intersecting when their closest approach is within this distance [m].")]
    public float intersectionThreshold = 5f;

    [Header("Runtime visualisation")]
    [Tooltip("Draw coloured LineRenderers for each path at startup (visible in Game view and builds).")]
    public bool drawAtRuntime = true;
    [Tooltip("Width of the rendered centreline [m].")]
    public float lineWidth = 1.0f;
    [Tooltip("Y height of the drawn lines above ground [m].")]
    public float lineHeight = 0.3f;

    public IReadOnlyList<TaxiwayPath> Paths => _paths;
    readonly List<TaxiwayPath> _paths = new List<TaxiwayPath>();

    public IReadOnlyList<HoldingPosition> HoldingPositions => _holds;
    readonly List<HoldingPosition> _holds = new List<HoldingPosition>();

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void Awake()
    {
        LoadMapData();
        if (drawAtRuntime) DrawNetworkRuntime();
    }

    // ── Public API ────────────────────────────────────────────────────────────

    // Right-click the component header in the Inspector → "Load Map Data" to parse the
    // GeoJSON in EDIT mode, so the network gizmos are visible without pressing Play
    // (e.g. while hand-placing occluders/buildings along a known taxiway).
    [ContextMenu("Load Map Data")]
    public void LoadMapData()
    {
        _paths.Clear();
        _holds.Clear();
        string filePath = Path.Combine(Application.streamingAssetsPath, geoJsonFileName);

        if (!File.Exists(filePath))
        {
            Debug.LogWarning(
                $"[TaxiwayNetwork] '{filePath}' not found — network is empty. " +
                "Place a GeoJSON file in Assets/StreamingAssets/ and re-enter Play.", this);
            return;
        }

        JObject root;
        try { root = JObject.Parse(File.ReadAllText(filePath)); }
        catch (System.Exception ex)
        {
            Debug.LogError($"[TaxiwayNetwork] Failed to parse GeoJSON: {ex.Message}", this);
            return;
        }

        var features = root["features"] as JArray;
        if (features == null) { Debug.LogError("[TaxiwayNetwork] No 'features' array found.", this); return; }

        foreach (var feature in features)
        {
            var geom = feature?["geometry"];
            if (geom == null) continue;

            string geomType = (string)geom["type"];
            var props = feature?["properties"];

            // Point features carry holding positions (aeroway=holding_position) — parked
            // stop-short markers, not drivable paths. Collect them separately.
            if (geomType == "Point")
            {
                if (props != null && (string)props["aeroway"] == "holding_position")
                {
                    var c = geom["coordinates"] as JArray;
                    if (c != null && c.Count >= 2)
                        _holds.Add(new HoldingPosition {
                            Position     = LatLonToUnity((double)c[1], (double)c[0]),
                            Ref          = (string)props["ref"] ?? "",
                            IsRunwayHold = (string)props["holding_position:type"] == "runway",
                        });
                }
                continue;
            }

            JArray rawCoords = null;

            if (geomType == "LineString")
                rawCoords = geom["coordinates"] as JArray;
            else if (geomType == "Polygon")
                rawCoords = (geom["coordinates"] as JArray)?[0] as JArray; // outer ring only

            var path = BuildPath(rawCoords);
            if (path == null) continue;

            // Attach OSM aeroway metadata from the feature's properties.
            if (props != null)
            {
                path.AerowayType = (string)props["aeroway"] ?? "";
                path.Ref         = (string)props["ref"]     ?? "";
                path.Name        = (string)props["name"]    ?? "";
                if (props["width"] != null &&
                    float.TryParse((string)props["width"],
                                   System.Globalization.NumberStyles.Float,
                                   System.Globalization.CultureInfo.InvariantCulture,
                                   out float w))
                    path.Width = w;
            }

            _paths.Add(path);
        }

        int nTaxi = 0, nApron = 0, nRunway = 0;
        foreach (var p in _paths)
        {
            if      (p.AerowayType == "taxiway") nTaxi++;
            else if (p.AerowayType == "apron")   nApron++;
            else if (p.AerowayType == "runway")  nRunway++;
        }
        Debug.Log($"[TaxiwayNetwork] Types: {nTaxi} taxiway, {nApron} apron, {nRunway} runway.", this);

        Debug.Log($"[TaxiwayNetwork] Loaded {_paths.Count} path(s) and {_holds.Count} holding " +
                  $"position(s) from '{geoJsonFileName}'.", this);
    }

    /// <summary>
    /// Nearest holding position to a world point, within maxDist metres. Returns false if none
    /// qualifies. Used to snap the runway-incursion vehicle's spawn onto a real hold-short marker.
    /// </summary>
    public bool TryNearestHoldingPosition(Vector3 near, float maxDist, out HoldingPosition hold)
    {
        hold = default;
        float best = maxDist;
        bool  found = false;
        for (int i = 0; i < _holds.Count; i++)
        {
            float dxz = Vector2.Distance(new Vector2(near.x, near.z),
                                         new Vector2(_holds[i].Position.x, _holds[i].Position.z));
            if (dxz < best) { best = dxz; hold = _holds[i]; found = true; }
        }
        return found;
    }

    /// <summary>
    /// Returns the Frenet-frame state of the agent relative to the given path.
    /// agentForward should be the world-space forward direction of the agent (Y ignored).
    ///
    /// sHint/searchWindow (optional): restrict the nearest-segment search to arc-lengths within
    /// searchWindow of sHint. Without this, a PURE global nearest-point search can snap onto a
    /// completely different (and oppositely-oriented) segment of the path whenever the agent is
    /// pushed far enough off-lane — e.g. swerving wide around a building via the LiDAR static-
    /// avoidance cost — because some unrelated stretch of the same taxiway polyline happens to
    /// pass physically closer than the segment the agent actually should be tracking. That snap
    /// flips the tangent/heading and can jump the arc-length backward, which is why progress
    /// toward the goal can stall or reverse right after a detour. Pass sHint &lt; 0 (default) for
    /// the old unrestricted behaviour; callers that track continuity (TaxiAgent) should pass the
    /// previous frame's s. Falls back to an unrestricted search if nothing falls in the window
    /// (e.g. first call, or the path changed).
    /// </summary>
    public PathState GetRelativeState(Vector3 agentPos, Vector3 agentForward, TaxiwayPath path,
                                      float sHint = -1f, float searchWindow = 40f)
    {
        var wps = path.Waypoints;
        var cum = path.CumulativeLength;
        bool useWindow = sHint >= 0f;

        // Find closest point on any segment (optionally restricted to an arc-length window
        // around sHint, for continuity — see doc comment above).
        int   bestSeg  = 0;
        float bestT    = 0f;
        float bestDist2 = float.MaxValue;
        Vector3 bestProj = wps[0];

        for (int i = 0; i < wps.Count - 1; i++)
        {
            if (useWindow && (cum[i] < sHint - searchWindow || cum[i] > sHint + searchWindow))
                continue;

            Vector3 a = wps[i], b = wps[i + 1];
            Vector3 ab = b - a;
            float len2 = ab.sqrMagnitude;
            float t = len2 > 1e-6f
                ? Mathf.Clamp01(Vector3.Dot(agentPos - a, ab) / len2)
                : 0f;
            Vector3 proj = a + ab * t;
            float dist2  = (agentPos - proj).sqrMagnitude;
            if (dist2 < bestDist2)
            {
                bestDist2 = dist2; bestSeg = i; bestT = t; bestProj = proj;
            }
        }

        // Nothing fell inside the window (stale/first hint, or the path changed) — fall back to
        // an unrestricted search rather than returning a meaningless default.
        if (useWindow && bestDist2 == float.MaxValue)
        {
            for (int i = 0; i < wps.Count - 1; i++)
            {
                Vector3 a = wps[i], b = wps[i + 1];
                Vector3 ab = b - a;
                float len2 = ab.sqrMagnitude;
                float t = len2 > 1e-6f
                    ? Mathf.Clamp01(Vector3.Dot(agentPos - a, ab) / len2)
                    : 0f;
                Vector3 proj = a + ab * t;
                float dist2  = (agentPos - proj).sqrMagnitude;
                if (dist2 < bestDist2)
                {
                    bestDist2 = dist2; bestSeg = i; bestT = t; bestProj = proj;
                }
            }
        }

        Vector3 tangent = (wps[bestSeg + 1] - wps[bestSeg]).normalized;

        // Arc-length s
        float segLen = Vector3.Distance(wps[bestSeg], wps[bestSeg + 1]);
        float s = path.CumulativeLength[bestSeg] + bestT * segLen;

        // Signed cross-track error: positive = left of path direction
        Vector3 toAgent = agentPos - bestProj;
        toAgent.y = 0f;
        float dSign = Mathf.Sign(Vector3.Dot(Vector3.Cross(tangent, toAgent), Vector3.up));
        float d = dSign * Mathf.Sqrt(bestDist2);

        // Signed heading error
        Vector3 fwd = agentForward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude > 1e-6f) fwd.Normalize();
        float dot   = Mathf.Clamp(Vector3.Dot(fwd, tangent), -1f, 1f);
        float cross = Vector3.Dot(Vector3.Cross(fwd, tangent), Vector3.up);
        float thetaErr = Mathf.Atan2(cross, dot);

        return new PathState { s = s, d = d, thetaError = thetaErr, tangent = tangent };
    }

    /// <summary>
    /// Arc-length remaining along path from agentPos to the end (clamped to 0).
    /// </summary>
    public float RemainingArcLength(Vector3 agentPos, TaxiwayPath path)
    {
        PathState ps = GetRelativeState(agentPos, Vector3.forward, path);
        return Mathf.Max(0f, path.TotalLength - ps.s);
    }

    /// <summary>
    /// Finds the closest world-space intersection point between pathA and pathB.
    /// Returns false if no segment pair comes within intersectionThreshold metres.
    /// </summary>
    public bool TryFindIntersection(TaxiwayPath pathA, TaxiwayPath pathB,
                                    out Vector3 intersection)
    {
        intersection = Vector3.zero;
        float best = intersectionThreshold;
        bool found = false;

        var wa = pathA.Waypoints;
        var wb = pathB.Waypoints;

        for (int i = 0; i < wa.Count - 1; i++)
        {
            for (int j = 0; j < wb.Count - 1; j++)
            {
                SegmentsClosestPoints(wa[i], wa[i+1], wb[j], wb[j+1],
                                      out Vector3 pA, out Vector3 pB);
                float d = Vector3.Distance(pA, pB);
                if (d < best) { best = d; intersection = (pA + pB) * 0.5f; found = true; }
            }
        }
        return found;
    }

    /// <summary>
    /// Builds a single continuous taxiway path from startPos to goalPos, stitching across
    /// intersections when the two points sit on different TaxiwayPath features (e.g. the goal
    /// is on the far side of an intersection from the start). BFS over the taxiway adjacency
    /// graph (edges = intersections found by TryFindIntersection), then slices/concatenates the
    /// waypoints of each path on the route between its entry and exit intersection.
    ///
    /// Returns null if start/goal aren't on any taxiway, or no connected route exists — callers
    /// should fall back to single-path behaviour in that case (e.g. via NearestTaxiway).
    /// startArc/goalArc are always 0 / stitched.TotalLength since the stitched path is built to
    /// start exactly at startPos's projection and end exactly at goalPos's projection.
    /// </summary>
    public TaxiwayPath BuildRoutedPath(Vector3 startPos, Vector3 goalPos,
                                       out float startArc, out float goalArc)
    {
        startArc = 0f; goalArc = 0f;

        var taxiways = new List<int>();
        for (int i = 0; i < _paths.Count; i++)
            if (_paths[i].IsTaxiway) taxiways.Add(i);
        if (taxiways.Count == 0) return null;

        int startIdx = NearestTaxiwayIndex(startPos, taxiways);
        int goalIdx  = NearestTaxiwayIndex(goalPos, taxiways);
        if (startIdx < 0 || goalIdx < 0) return null;

        if (startIdx == goalIdx)
        {
            var single = _paths[startIdx];
            var slice  = SlicePath(single,
                GetRelativeState(startPos, Vector3.forward, single).s,
                GetRelativeState(goalPos,  Vector3.forward, single).s);
            var stitchedSame = new TaxiwayPath { AerowayType = "taxiway" };
            stitchedSame.Waypoints.AddRange(slice);
            stitchedSame.Precompute();
            goalArc = stitchedSame.TotalLength;
            return stitchedSame;
        }

        // BFS over taxiway paths; edge (cur -> nb) exists when TryFindIntersection finds a
        // crossing between them within intersectionThreshold.
        var prevPath = new Dictionary<int, int>();
        var viaPoint = new Dictionary<int, Vector3>();
        var visited  = new HashSet<int> { startIdx };
        var queue    = new Queue<int>();
        queue.Enqueue(startIdx);

        while (queue.Count > 0 && !visited.Contains(goalIdx))
        {
            int cur = queue.Dequeue();
            foreach (int nb in taxiways)
            {
                if (visited.Contains(nb)) continue;
                if (!TryFindIntersection(_paths[cur], _paths[nb], out Vector3 ix)) continue;
                visited.Add(nb);
                prevPath[nb] = cur;
                viaPoint[nb] = ix;
                queue.Enqueue(nb);
            }
        }

        if (!visited.Contains(goalIdx))
        {
            Debug.LogWarning("[TaxiwayNetwork] BuildRoutedPath: no connected taxiway route from " +
                              "start to goal (paths never intersect within intersectionThreshold).");
            return null;
        }

        // Reconstruct route = [startIdx, ..., goalIdx] and the intersection point entering each
        // path after the first.
        var route    = new List<int> { goalIdx };
        var ixPoints = new List<Vector3>();
        int c = goalIdx;
        while (c != startIdx)
        {
            ixPoints.Add(viaPoint[c]);
            c = prevPath[c];
            route.Add(c);
        }
        route.Reverse();
        ixPoints.Reverse();

        // Stitch: for each path on the route, slice from its entry arc-length to its exit
        // arc-length (start marker / previous intersection → next intersection / goal marker).
        var combined = new List<Vector3>();
        for (int k = 0; k < route.Count; k++)
        {
            var path  = _paths[route[k]];
            float sFrom = (k == 0)
                ? GetRelativeState(startPos, Vector3.forward, path).s
                : GetRelativeState(ixPoints[k - 1], Vector3.forward, path).s;
            float sTo = (k < route.Count - 1)
                ? GetRelativeState(ixPoints[k], Vector3.forward, path).s
                : GetRelativeState(goalPos, Vector3.forward, path).s;

            var slice = SlicePath(path, sFrom, sTo);
            if (combined.Count > 0 && slice.Count > 0 &&
                Vector3.Distance(combined[combined.Count - 1], slice[0]) < 0.01f)
                slice.RemoveAt(0);
            combined.AddRange(slice);
        }

        var stitched = new TaxiwayPath { AerowayType = "taxiway" };
        stitched.Waypoints.AddRange(combined);
        if (stitched.Waypoints.Count < 2) return null;
        stitched.Precompute();
        goalArc = stitched.TotalLength;
        return stitched;
    }

    int NearestTaxiwayIndex(Vector3 p, List<int> taxiways)
    {
        int   best  = -1;
        float bestD = float.MaxValue;
        foreach (int i in taxiways)
        {
            float d = Mathf.Abs(GetRelativeState(p, Vector3.forward, _paths[i]).d);
            if (d < bestD) { bestD = d; best = i; }
        }
        return best;
    }

    /// <summary>
    /// Returns the ordered waypoints of `path` between arc-lengths sFrom and sTo (inclusive),
    /// oriented from sFrom to sTo (reversed if sFrom &gt; sTo, i.e. travelling backward along
    /// the source waypoint order).
    /// </summary>
    static List<Vector3> SlicePath(TaxiwayPath path, float sFrom, float sTo)
    {
        var wps    = path.Waypoints;
        var cum    = path.CumulativeLength;
        bool fwd   = sTo >= sFrom;
        float lo   = Mathf.Min(sFrom, sTo), hi = Mathf.Max(sFrom, sTo);

        Vector3 PointAtArc(float s)
        {
            s = Mathf.Clamp(s, 0f, path.TotalLength);
            for (int i = 0; i < cum.Length - 1; i++)
            {
                if (s <= cum[i + 1] || i == cum.Length - 2)
                {
                    float segLen = cum[i + 1] - cum[i];
                    float t = segLen > 1e-6f ? (s - cum[i]) / segLen : 0f;
                    return Vector3.Lerp(wps[i], wps[i + 1], Mathf.Clamp01(t));
                }
            }
            return wps[wps.Count - 1];
        }

        var result = new List<Vector3> { PointAtArc(lo) };
        for (int i = 0; i < wps.Count; i++)
            if (cum[i] > lo && cum[i] < hi) result.Add(wps[i]);
        result.Add(PointAtArc(hi));

        if (!fwd) result.Reverse();
        return result;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    TaxiwayPath BuildPath(JArray rawCoords)
    {
        if (rawCoords == null || rawCoords.Count < 2) return null;

        var path = new TaxiwayPath();
        foreach (var pt in rawCoords)
        {
            var arr = pt as JArray;
            if (arr == null || arr.Count < 2) continue;
            double lon = (double)arr[0];
            double lat = (double)arr[1];
            path.Waypoints.Add(LatLonToUnity(lat, lon));
        }

        if (path.Waypoints.Count < 2) return null;
        path.Precompute();
        return path;
    }

    Vector3 LatLonToUnity(double lat, double lon)
    {
        const double R      = 6378137.0;
        const double DEG2RAD = System.Math.PI / 180.0;
        double dLat  = (lat - originLat) * DEG2RAD;
        double dLon  = (lon - originLon) * DEG2RAD;
        double cosLat = System.Math.Cos(originLat * DEG2RAD);
        float  z = (float)(dLat * R);           // north → Unity +Z
        float  x = (float)(dLon * R * cosLat);  // east  → Unity +X
        return new Vector3(x, 0f, z);
    }

    // 3-D segment–segment closest-point (Ericson, Real-Time Collision Detection §5.1.9)
    static void SegmentsClosestPoints(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4,
                                      out Vector3 c1, out Vector3 c2)
    {
        Vector3 d1 = p2 - p1, d2 = p4 - p3, r = p1 - p3;
        float a = Vector3.Dot(d1, d1), e = Vector3.Dot(d2, d2), f = Vector3.Dot(d2, r);
        float s, t;

        if (a < 1e-6f && e < 1e-6f) { s = t = 0f; }
        else if (a < 1e-6f) { s = 0f; t = Mathf.Clamp01(f / e); }
        else
        {
            float c = Vector3.Dot(d1, r);
            if (e < 1e-6f)
            {
                t = 0f; s = Mathf.Clamp01(-c / a);
            }
            else
            {
                float b     = Vector3.Dot(d1, d2);
                float denom = a * e - b * b;
                s = denom > 1e-6f ? Mathf.Clamp01((b * f - c * e) / denom) : 0f;
                t = (f + b * s) / e;
                if      (t < 0f) { t = 0f; s = Mathf.Clamp01(-c / a); }
                else if (t > 1f) { t = 1f; s = Mathf.Clamp01((b - c) / a); }
            }
        }
        c1 = p1 + d1 * s;
        c2 = p3 + d2 * t;
    }

    // ── Runtime visualisation ─────────────────────────────────────────────────

    void DrawNetworkRuntime()
    {
        // One child GameObject + LineRenderer per path, parented to this object.
        for (int pi = 0; pi < _paths.Count; pi++)
        {
            var wps = _paths[pi].Waypoints;
            if (wps.Count < 2) continue;

            var go = new GameObject($"Path_{pi}");
            go.transform.SetParent(transform, worldPositionStays: false);

            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace   = true;
            lr.positionCount   = wps.Count;
            lr.startWidth      = lineWidth;
            lr.endWidth        = lineWidth;
            lr.material        = new Material(Shader.Find("Sprites/Default"));
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            lr.receiveShadows  = false;

            // Colour by aeroway type (standard airport-chart-ish convention):
            //   taxiway = yellow, runway = red, apron = cyan, unknown = grey.
            Color c;
            switch (_paths[pi].AerowayType)
            {
                case "taxiway": c = Color.yellow;                 break;
                case "runway":  c = Color.red;                    break;
                case "apron":   c = Color.cyan;                   break;
                default:        c = new Color(0.6f, 0.6f, 0.6f);  break;
            }
            lr.startColor = c;
            lr.endColor   = c;

            for (int i = 0; i < wps.Count; i++)
                lr.SetPosition(i, wps[i] + Vector3.up * lineHeight);
        }
    }

    // ── Gizmos ────────────────────────────────────────────────────────────────

    bool _editorLoadTried;   // avoid re-parsing every gizmo frame when the file is missing

    void OnDrawGizmos()
    {
        // Always drawn (not just when selected) so taxiways are visible while placing markers.
        // Edit mode: Awake() hasn't run, so lazily parse the GeoJSON once — makes the
        // network visible in the Scene view without entering Play mode.
        if (_paths.Count == 0 && !Application.isPlaying && !_editorLoadTried)
        {
            _editorLoadTried = true;
            LoadMapData();
        }
        if (_paths == null) return;
        for (int pi = 0; pi < _paths.Count; pi++)
        {
            // Colour by feature type so aprons (parking = occluder candidates) stand out:
            // taxiway = yellow, runway = red, apron = cyan, other = grey.
            switch (_paths[pi].AerowayType)
            {
                case "taxiway": Gizmos.color = Color.yellow;              break;
                case "runway":  Gizmos.color = Color.red;                 break;
                case "apron":   Gizmos.color = Color.cyan;                break;
                default:        Gizmos.color = new Color(.6f, .6f, .6f);  break;
            }
            var wps = _paths[pi].Waypoints;
            for (int i = 0; i < wps.Count - 1; i++)
            {
                Gizmos.DrawLine(wps[i], wps[i + 1]);
                Gizmos.DrawSphere(wps[i], 0.5f);
            }
            if (wps.Count > 0) Gizmos.DrawSphere(wps[wps.Count - 1], 0.5f);
        }
    }
}
