using System.Collections.Generic;
using UnityEngine;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;

/// <summary>
/// Draws the expanding keep-out circles the controller applies to SENSED MOVING obstacles
/// — the visible-agent counterpart of PhantomAgentVisualizer's blind-corner capsules.
///
/// The Python side (controller/dynamic_clusters.py) clusters the LiDAR returns, tracks the
/// clusters across scans, and classifies the ones whose estimated speed clears v_min as
/// moving. Each of those gets the same forward-reachable-set circle a hidden agent gets:
///
///     r_keep(t) = d_safe_hard + r_cluster + v_target * min(t + age, grow_horizon)
///
/// r_cluster is the mover's own measured extent and `age` is how stale the centroid is
/// (the cloud publishes at ~1 Hz, so the agent has been free to move since it was seen).
/// The cap exists because a visible mover is in EVERY solve, unlike an occasional corner.
///
/// THIS SCRIPT DECLARES NO MODEL CONSTANTS, on purpose. PhantomAgentVisualizer mirrors
/// v_target / d_safe_hard as Inspector fields and its docstring warns they must be kept
/// equal to controller/config.yaml by hand; when they drift, the Scene view confidently
/// draws a keep-out the controller never used. Here the three parameters ride in the
/// message ahead of the cluster rows, so what is drawn IS what the planner constrained.
///
/// Wire protocol — /dynamic_clusters, std_msgs/Float32MultiArray:
///
///     data[0..2]                    v_target, d_safe_hard, grow_horizon
///     data[3 + 4*i .. 3 + 4*i + 3]  c0, c1, r_cluster, age   for cluster i
///
/// Coordinates are the controller's world frame (a0, a1) = (Unity Z, Unity X), the same
/// convention LaserSensor3D packs its cloud in — so a0 -> Z and a1 -> X, no rotation.
///
/// The set published is POST-GATING: only the movers that actually entered the cost
/// (within query_r, nearest k_dyn). A detection the planner ignored does not appear, which
/// is the point — this shows the constraint, not the perception. An empty message clears
/// the overlay, so circles vanish as soon as the planner stops constraining them.
///
/// Attach anywhere in the scene (it needs no sensor reference — the geometry arrives in
/// world coordinates). Run the controller with --dynamic-obstacles --dynamic-viz.
/// </summary>
public class DynamicAgentVisualizer : MonoBehaviour
{
    [Header("Feed")]
    [Tooltip("Topic the controller publishes the keep-outs on (--dynamic-viz).")]
    public string topic = "/dynamic_clusters";
    [Tooltip("Clear the overlay if no message arrives for this long [s]. The controller "
           + "publishes every control step, so a gap means it stopped or died — better to "
           + "show nothing than a frozen circle that looks live.")]
    [Min(0.1f)] public float messageTimeout = 1.0f;

    [Header("Horizon")]
    [Tooltip("vehicle.dt — controller timestep [s]. Only sets the SPACING of the drawn "
           + "rings; the radii themselves come from the message.")]
    public float dt = 0.1f;
    [Tooltip("Rings across the horizon. Each is one t_k = (k+1)*dt*ringStride.")]
    [Min(1)] public int ringCount = 5;
    [Tooltip("Steps between rings — the rings span ringCount*ringStride*dt seconds.")]
    [Min(1)] public int ringStride = 10;

    [Header("Playback")]
    [Tooltip("ON: sweep the horizon, showing only the keep-out for the CURRENT t_k and "
           + "looping back to t_0 — the reachable set growing, as the controller sees it "
           + "step by step. OFF: draw every ring at once (their swept union).")]
    public bool animate = true;
    [Tooltip("Wall-clock seconds per horizon second. 1 = the keep-out grows in real time.")]
    [Min(0.01f)] public float playbackSpeed = 1.0f;
    [Tooltip("Hold on t_0 for this long [s] at the start of each loop, so the restart is "
           + "readable instead of a flicker.")]
    [Min(0f)] public float loopPause = 0.5f;

    [Header("Drawing")]
    [Tooltip("Mark the measured cluster extent (the t=0 disc, before any growth) so the "
           + "detection itself is visible separately from the reachable set around it.")]
    public bool showClusterExtent = true;
    [Tooltip("Segments per full circle.")]
    [Min(8)] public int circleSegments = 48;
    [Tooltip("Height of the drawing plane in Unity world Y [m]. The message carries only "
           + "the ground-plane centre, since the keep-out is a 2-D constraint.")]
    public float groundY = 0.05f;
    [Tooltip("Colour of the expanding keep-out rings. Deliberately distinct from the "
           + "occlusion capsules' orange: same model, different hazard source.")]
    public Color keepOutColor = new Color(0.1f, 0.85f, 1f, 1f);
    [Tooltip("Colour of the measured cluster extent.")]
    public Color extentColor = new Color(1f, 1f, 1f, 0.8f);

    // Latest keep-out set, in Unity world coordinates. Written on the ROS thread, read in
    // Update() — hence the lock: ROSConnection delivers on its own reader thread and a
    // half-written list would draw garbage.
    struct Mover
    {
        public Vector3 centre;      // Unity world, on the drawing plane
        public float rCluster;      // measured extent [m]
        public float rBase;         // d_safe_hard + rCluster + v_target*age  (radius at t=0)
    }

    readonly List<Mover> movers = new List<Mover>();
    readonly object movLock = new object();
    float vTarget;                  // from the message — never an Inspector constant
    float growHorizon;
    float lastMessageTime = -999f;

    // Phase of the sweep, in horizon-time seconds. Advanced by unscaled real time so the
    // animation keeps running while the sim is paused in the Editor.
    float sweepClock;

    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<Float32MultiArrayMsg>(topic, OnClusters);
    }

    void OnClusters(Float32MultiArrayMsg msg)
    {
        const int nParams = 3;
        const int row = 4;
        if (msg.data == null || msg.data.Length < nParams)
            return;

        lock (movLock)
        {
            vTarget = msg.data[0];
            float dSafeHard = msg.data[1];
            growHorizon = msg.data[2];

            movers.Clear();
            int nRows = (msg.data.Length - nParams) / row;
            for (int i = 0; i < nRows; i++)
            {
                int o = nParams + i * row;
                float a0 = msg.data[o];
                float a1 = msg.data[o + 1];
                float rCluster = msg.data[o + 2];
                float age = msg.data[o + 3];
                movers.Add(new Mover
                {
                    // Controller world (a0, a1) -> Unity (Z, X). Same axis convention
                    // LaserSensor3D uses when it packs the cloud, so no rotation is needed.
                    centre = new Vector3(a1, groundY, a0),
                    rCluster = rCluster,
                    // The age term is folded in here rather than re-derived per ring: it is
                    // travel that has ALREADY happened, so it shifts the whole family of
                    // radii up, it does not grow with t.
                    rBase = dSafeHard + rCluster + vTarget * age,
                });
            }
            lastMessageTime = Time.unscaledTime;
        }
    }

    void Update()
    {
        int activeRing = AdvanceSweep();

        lock (movLock)
        {
            // Stale feed: the controller publishes every control step, so silence means it
            // is gone. Drop the overlay rather than leave a circle that is no longer true.
            if (Time.unscaledTime - lastMessageTime > messageTimeout)
                movers.Clear();

            foreach (var m in movers)
            {
                var c = m.centre;
                c.y = groundY;

                if (showClusterExtent)
                    KeepOutGizmos.DrawCircle(c, m.rCluster, extentColor, circleSegments);

                if (animate)
                {
                    DrawRing(c, m.rBase, activeRing);
                }
                else
                {
                    for (int k = 0; k < ringCount; k++)
                        DrawRing(c, m.rBase, k);
                }
            }
        }
    }

    /// <summary>
    /// Steps the sweep clock and returns the ring index to draw this frame: loopPause
    /// seconds parked on t_0 followed by the horizon itself, replayed forever. Unscaled
    /// time keeps it moving while the sim is paused. Mirrors PhantomAgentVisualizer, so
    /// the two overlays animate in step when both are on.
    /// </summary>
    int AdvanceSweep()
    {
        if (!animate)
            return 0;

        float horizon = ringCount * ringStride * dt;
        float cycle = loopPause + horizon;

        sweepClock += Time.unscaledDeltaTime * playbackSpeed;
        if (sweepClock >= cycle)
            sweepClock -= cycle * Mathf.Floor(sweepClock / cycle);

        float tSweep = sweepClock - loopPause;
        int k = Mathf.FloorToInt(tSweep / (ringStride * dt));
        return Mathf.Clamp(k, 0, ringCount - 1);
    }

    void DrawRing(Vector3 centre, float rBase, int k)
    {
        float t = (k + 1) * ringStride * dt;
        // The SAME cap the cost applies (dynamic_clusters.grow_horizon). Without it the
        // Scene view would show the ego swerving around a circle far larger than the one it
        // was actually avoiding. 0 means uncapped, matching the config.
        if (growHorizon > 0f)
            t = Mathf.Min(t, growHorizon);

        // Fade with horizon time: the near-term reachable set is the confident one, the far
        // rings are increasingly speculative.
        float fade = Mathf.Max(0.45f, 1f - (float)k / ringCount);
        var col = keepOutColor;
        col.a *= fade;
        KeepOutGizmos.DrawCircle(centre, rBase + vTarget * t, col, circleSegments);
    }
}
