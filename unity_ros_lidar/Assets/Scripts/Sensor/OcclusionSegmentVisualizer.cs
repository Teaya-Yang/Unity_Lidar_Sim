using System.Collections.Generic;
using UnityEngine;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;

/// <summary>
/// Draws the occlusion boundary segments the PYTHON detector found on the current scan,
/// plus the expanding keep-out capsule each of them seeds.
///
/// WHY THIS EXISTS ALONGSIDE PhantomAgentVisualizer. That script draws boundaries Unity
/// finds itself, in LaserSensor3D's own edge pass, and mirrors v_target / d_safe_hard as
/// Inspector constants. Those are a DIFFERENT detector and a separate copy of the model,
/// so when the Python side of controller/occlusion_capsules.py is retuned the two views
/// disagree silently. This one shows exactly what segments_from_ordered_cloud returned —
/// corner first — with the model parameters carried in the message, so the overlay cannot
/// drift from what the controller is planning against.
///
/// Wire protocol — /occlusion_segments, std_msgs/Float32MultiArray:
///
///     data[0..2]                    v_target, d_safe_hard, t_grow_max
///     data[3 + 4*i .. 3 + 4*i + 3]  a0_near, a1_near, a0_far, a1_far   for segment i
///
/// The NEAR endpoint is the corner a hidden agent would round; the FAR endpoint is where
/// the escaping beam landed. Coordinates are the controller's world frame
/// (a0, a1) = (Unity Z, Unity X), the same convention LaserSensor3D packs its cloud in —
/// so a0 -> Z and a1 -> X, no rotation.
///
/// The set published is the RAW per-scan detection, BEFORE the corner tracker, the
/// forward wedge and the query_r / k_occ gates. It answers "what did the jump test see?",
/// not "what did the cost constrain?" — expect more segments here than the controller's
/// `used=` count, and expect them to flicker, since nothing has smoothed them yet.
///
/// An empty message clears the overlay, so segments vanish as soon as the detector stops
/// reporting them.
///
/// Attach anywhere in the scene (it needs no sensor reference — the geometry arrives in
/// world coordinates). Run the controller with --occlusion-aware --occlusion-viz.
/// </summary>
public class OcclusionSegmentVisualizer : MonoBehaviour
{
    [Header("Feed")]
    [Tooltip("Topic the controller publishes the boundaries on (--occlusion-viz).")]
    public string topic = "/occlusion_segments";
    [Tooltip("Clear the overlay if no message arrives for this long [s]. The controller "
           + "publishes every control step, so a gap means it stopped or died — better to "
           + "show nothing than a frozen segment that looks live.")]
    [Min(0.1f)] public float messageTimeout = 1.0f;

    [Header("Horizon")]
    [Tooltip("vehicle.dt — controller timestep [s]. Only sets the SPACING of the drawn "
           + "capsules; the radii themselves come from the message.")]
    public float dt = 0.1f;
    [Tooltip("Capsules across the horizon. Each is one t_k = (k+1)*dt*ringStride.")]
    [Min(1)] public int ringCount = 5;
    [Tooltip("Steps between capsules — they span ringCount*ringStride*dt seconds.")]
    [Min(1)] public int ringStride = 10;

    [Header("Playback")]
    [Tooltip("ON: sweep the horizon, showing only the keep-out for the CURRENT t_k and "
           + "looping back to t_0. OFF: draw every capsule at once (their swept union).")]
    public bool animate = true;
    [Tooltip("Wall-clock seconds per horizon second. 1 = the keep-out grows in real time.")]
    [Min(0.01f)] public float playbackSpeed = 1.0f;
    [Tooltip("Hold on t_0 for this long [s] at the start of each loop, so the restart is "
           + "readable instead of a flicker.")]
    [Min(0f)] public float loopPause = 0.5f;

    [Header("Drawing")]
    [Tooltip("Draw the bare detected segment (corner -> far endpoint) with no dilation, so "
           + "the detection itself is visible separately from the keep-out around it.")]
    public bool showSegments = true;
    [Tooltip("Draw the expanding keep-out capsules. Turn off to see only the raw "
           + "detections, which is the readable view when the segments are long.")]
    public bool showKeepOut = true;
    [Tooltip("Mark each corner with a cross of this size [m]. 0 = off. The corner is the "
           + "endpoint that matters — it anchors the whole keep-out.")]
    [Min(0f)] public float cornerMarkSize = 2.0f;
    [Tooltip("Clip drawn segments to this length [m] so one beam that escaped to max range "
           + "does not stretch a capsule across the whole map. 0 = draw them in full. This "
           + "is a VIEWER-side clip: it does not change what the controller constrains.")]
    [Min(0f)] public float maxDrawLength = 30.0f;
    [Tooltip("Segments per full circle on the capsule caps.")]
    [Min(8)] public int circleSegments = 48;
    [Tooltip("Height of the drawing plane in Unity world Y [m]. The message carries only "
           + "the ground-plane geometry, since the keep-out is a 2-D constraint.")]
    public float groundY = 0.05f;
    [Tooltip("Colour of the expanding keep-out capsules.")]
    public Color keepOutColor = new Color(1f, 0.55f, 0.1f, 1f);
    [Tooltip("Colour of the bare detected segment and its corner mark.")]
    public Color segmentColor = new Color(1f, 1f, 1f, 0.9f);

    // Latest boundary set, in Unity world coordinates. Written on the ROS reader thread,
    // read in Update() — hence the lock, exactly as DynamicAgentVisualizer does it.
    struct Boundary
    {
        public Vector3 corner;      // Unity world, on the drawing plane
        public Vector3 far;         // ditto, already clipped for drawing
    }

    readonly List<Boundary> boundaries = new List<Boundary>();
    readonly object segLock = new object();
    float vTarget;                  // from the message — never an Inspector constant
    float dSafeHard;
    float tGrowMax;
    float lastMessageTime = -999f;

    // Phase of the sweep, in horizon-time seconds. Advanced by unscaled real time so the
    // animation keeps running while the sim is paused in the Editor.
    float sweepClock;

    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<Float32MultiArrayMsg>(topic, OnSegments);
    }

    void OnSegments(Float32MultiArrayMsg msg)
    {
        const int nParams = 3;
        const int row = 4;
        if (msg.data == null || msg.data.Length < nParams)
            return;

        lock (segLock)
        {
            vTarget = msg.data[0];
            dSafeHard = msg.data[1];
            tGrowMax = msg.data[2];

            boundaries.Clear();
            int nRows = (msg.data.Length - nParams) / row;
            for (int i = 0; i < nRows; i++)
            {
                int o = nParams + i * row;
                // Controller world (a0, a1) -> Unity (Z, X).
                var corner = new Vector3(msg.data[o + 1], groundY, msg.data[o]);
                var far = new Vector3(msg.data[o + 3], groundY, msg.data[o + 2]);

                // A beam that escaped to max range gives a segment kilometres long. The
                // controller may well constrain it in full; drawing it in full just fills
                // the Scene view, so the clip is applied here and nowhere else.
                if (maxDrawLength > 0f)
                {
                    var v = far - corner;
                    float len = v.magnitude;
                    if (len > maxDrawLength)
                        far = corner + v * (maxDrawLength / len);
                }

                boundaries.Add(new Boundary { corner = corner, far = far });
            }
            lastMessageTime = Time.unscaledTime;
        }
    }

    void Update()
    {
        int activeRing = AdvanceSweep();

        lock (segLock)
        {
            // Stale feed: the controller publishes every control step, so silence means it
            // is gone. Drop the overlay rather than leave a segment that is no longer true.
            if (Time.unscaledTime - lastMessageTime > messageTimeout)
                boundaries.Clear();

            foreach (var b in boundaries)
            {
                if (showSegments)
                {
                    Debug.DrawLine(b.corner, b.far, segmentColor, 0f, false);
                    if (cornerMarkSize > 0f)
                    {
                        float h = cornerMarkSize * 0.5f;
                        Debug.DrawLine(b.corner + Vector3.left * h, b.corner + Vector3.right * h,
                                       segmentColor, 0f, false);
                        Debug.DrawLine(b.corner + Vector3.back * h, b.corner + Vector3.forward * h,
                                       segmentColor, 0f, false);
                    }
                }

                if (!showKeepOut)
                    continue;

                if (animate)
                {
                    DrawKeepOut(b, activeRing);
                }
                else
                {
                    for (int k = 0; k < ringCount; k++)
                        DrawKeepOut(b, k);
                }
            }
        }
    }

    /// <summary>
    /// Steps the sweep clock and returns the capsule index to draw this frame: loopPause
    /// seconds parked on t_0 followed by the horizon itself, replayed forever. Mirrors
    /// DynamicAgentVisualizer, so the two overlays animate in step when both are on.
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

    void DrawKeepOut(Boundary b, int k)
    {
        float t = (k + 1) * ringStride * dt;
        // The SAME cap the stage cost applies (occlusion.t_grow_max). 0 means uncapped,
        // matching the config.
        if (tGrowMax > 0f)
            t = Mathf.Min(t, tGrowMax);

        // Fade with horizon time: the near-term reachable set is the confident one, the far
        // capsules are increasingly speculative.
        float fade = Mathf.Max(0.45f, 1f - (float)k / ringCount);
        var col = keepOutColor;
        col.a *= fade;
        KeepOutGizmos.DrawCapsule(b.corner, b.far, dSafeHard + vTarget * t, col, circleSegments);
    }
}
