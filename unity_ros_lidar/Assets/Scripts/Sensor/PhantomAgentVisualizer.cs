using UnityEngine;
using UnityEngine.Serialization;

/// <summary>
/// Draws the forward-reachable-set keep-out CAPSULES the occlusion-aware controllers
/// actually plan against (Firoozi et al.). A worst-case hidden agent is assumed to lurk
/// anywhere along a blind-corner boundary and advance at up to V_TARGET in an arbitrary
/// direction, so the forbidden region is that boundary SEGMENT dilated by
///
///     r_keep(t) = D_SAFE_HARD + V_TARGET * t
///
/// The dilation of a segment is a capsule: a disc at each endpoint plus the rectangle
/// spanning them. Its outline is drawn here as two straight flanks closed by a
/// semicircular cap at each end — the direct analogue of capsule_polygon() in
/// controller/occlusion_capsules.py, and the level set of the point-to-SEGMENT distance
/// both controllers evaluate.
///
/// ORIENTATION. The capsule axis runs from the corner roughly ALONG THE SIGHTLINE,
/// radially away from the sensor and into the occluded region — it lies across the mouth
/// of the shadow, NOT along the occluder's face. It is offset from the exact
/// sensor->corner ray by about one beam spacing, because the two endpoints are adjacent
/// beams.
///
/// The keep-out is time-varying, so it is ANIMATED: each frame shows one horizon step t_k
/// for every boundary at once, and the sweep walks t_0 -> t_N and loops. That reads as the
/// reachable set expanding in time, which is what the controller sees step by step. Set
/// animate = false to fall back to the old view, where all rings are drawn simultaneously
/// and the picture is really their swept union.
///
/// A boundary whose far endpoint was never reached (the beam escaped to max range) is
/// clipped by LaserSensor3D.maxSegLenMeters, so in open scenes almost every capsule is
/// exactly that long.
///
/// Set useCapsules = false to collapse each boundary onto its corner, which reproduces
/// the older expanding-CIRCLE view. That mirrors `occlusion.use_capsules` in
/// controller/config.yaml — keep the two equal or the Scene view will disagree with what
/// the controllers constrain.
///
/// The constants below MUST track controller/config.yaml, which is the single source of
/// truth for both MPPI and MPC. This script is a viewer only — it never feeds the
/// controller, so a mismatch here shows a misleading capsule rather than changing
/// behaviour.
/// </summary>
[RequireComponent(typeof(PointCloudPublisher))]
public class PhantomAgentVisualizer : MonoBehaviour
{
    [Header("Hidden-agent model (mirror of controller/config.yaml)")]
    [Tooltip("occlusion.v_target — assumed max speed of a hidden agent emerging from occlusion [m/s].")]
    public float vTarget = 5.0f;
    [Tooltip("keepout.d_safe_hard — base (t=0) keep-out radius around an occlusion boundary [m].")]
    [FormerlySerializedAs("dSafeOcc")]
    public float dSafeHard = 30.0f;
    [Tooltip("occlusion.query_r — ignore boundaries further than this from the sensor [m].")]
    public float occQueryR = 60.0f;

    [Header("Keep-out shape")]
    [Tooltip("occlusion.use_capsules — ON: dilate the whole boundary segment (a capsule, the " +
             "hidden agent may lurk anywhere down the blind sightline). OFF: collapse to the " +
             "corner, giving the older expanding circle.")]
    public bool useCapsules = true;
    [Tooltip("Also draw the capsule AXIS (corner -> far endpoint), so the sightline " +
             "orientation is visible even when the keep-out is large.")]
    public bool showAxis = true;

    [Header("Horizon")]
    [Tooltip("vehicle.dt — controller timestep [s].")]
    public float dt = 0.1f;
    [Tooltip("Rings across the horizon. Each is one t_k = (k+1)*DT*ringStride.")]
    [Min(1)] public int ringCount = 5;
    [Tooltip("Steps between rings — the horizon spans ringCount*ringStride*DT seconds.")]
    [Min(1)] public int ringStride = 10;

    [Header("Playback")]
    [Tooltip("ON: sweep the horizon, showing only the keep-out for the CURRENT t_k and " +
             "looping back to t_0. OFF: draw every ring at once (the old stacked view).")]
    public bool animate = true;
    [Tooltip("Wall-clock seconds per horizon second. 1 = the keep-out grows in real time; " +
             "lower slows the sweep down.")]
    [Min(0.01f)] public float playbackSpeed = 1.0f;
    [Tooltip("Hold on t_0 for this long [s] at the start of each loop, so the restart is " +
             "readable instead of a flicker.")]
    [Min(0f)] public float loopPause = 0.5f;

    // Phase of the sweep, in horizon-time seconds. Advanced by unscaled real time so the
    // animation keeps running while the sim is paused in the Editor.
    float sweepClock;

    [Header("Drawing")]
    [Tooltip("Segments per full circle. Each capsule end-cap uses half of this.")]
    [Min(8)] public int circleSegments = 48;
    [Tooltip("Lift the outlines this far above the drawing plane [m].")]
    public float groundOffset = 0.05f;
    [Tooltip("Flatten everything onto one horizontal plane this far BELOW the sensor, instead " +
             "of leaving it at the boundary's own hit height (often partway up a wall).")]
    public bool projectToGroundPlane = true;
    [Tooltip("Sensor height above the driving plane [m]. Only used when projecting.")]
    public float sensorHeight = 2.0f;

    PointCloudPublisher publisher;

    void Awake()
    {
        publisher = GetComponent<PointCloudPublisher>();
    }

    void Update()
    {
        var sensor = publisher != null ? publisher.Sensor : null;
        if (sensor == null)
            return;

        // Ask the sensor to run its edge pass even when the debug lines are off.
        sensor.computeOcclusionEdges = true;

        var segments = sensor.OcclusionSegments;
        if (segments == null || segments.Count == 0)
            return;

        var origin = publisher.laser_sensor_link.transform.position;
        float queryRSq = occQueryR * occQueryR;
        float planeY = origin.y - sensorHeight + groundOffset;

        // One timestamp for the whole frame, so every boundary shows the SAME t_k — the
        // reachable set at one instant of the horizon, which is what the controller
        // constrains at that step. Drawing all k at once instead shows the swept union.
        int activeRing = AdvanceSweep();

        foreach (var seg in segments)
        {
            // Range-gate on the CORNER, matching how the controllers anchor a phantom.
            if ((seg.corner - origin).sqrMagnitude > queryRSq)
                continue;

            var a = seg.corner + Vector3.up * groundOffset;
            var b = (useCapsules ? seg.far : seg.corner) + Vector3.up * groundOffset;
            if (projectToGroundPlane)
            {
                a.y = planeY;
                b.y = planeY;
            }

            if (showAxis)
                Debug.DrawLine(a, b, new Color(1f, 1f, 1f, 0.6f), 0f, false);

            if (animate)
            {
                DrawRing(a, b, activeRing);
            }
            else
            {
                for (int k = 0; k < ringCount; k++)
                    DrawRing(a, b, k);
            }
        }
    }

    /// <summary>
    /// Steps the sweep clock and returns the ring index to draw this frame. The cycle is
    /// loopPause seconds parked on t_0 followed by the horizon itself, replayed forever.
    /// Unscaled time keeps it moving while the sim is paused.
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

    void DrawRing(Vector3 a, Vector3 b, int k)
    {
        float t = (k + 1) * ringStride * dt;
        // Fade with horizon time: the near-term reachable set is the confident one, the far
        // rings are increasingly speculative. Floored at 0.45 — below roughly that, a debug
        // line is too faint to pick out against the ground.
        float fade = Mathf.Max(0.45f, 1f - (float)k / ringCount);
        DrawCapsule(a, b, dSafeHard + vTarget * t, new Color(1f, 0.35f, 0f, fade));
    }

    void DrawCapsule(Vector3 a, Vector3 b, float radius, Color color)
    {
        // Geometry lives in KeepOutGizmos, shared with DynamicAgentVisualizer: the two
        // keep-outs are the same shape by construction on the Python side, so they must be
        // the same shape here too.
        KeepOutGizmos.DrawCapsule(a, b, radius, color, circleSegments);
    }
}
