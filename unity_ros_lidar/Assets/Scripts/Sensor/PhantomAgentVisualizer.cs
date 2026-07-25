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
    [Tooltip("Rings drawn across the horizon. Each is one t_k = (k+1)*DT*ringStride.")]
    [Min(1)] public int ringCount = 5;
    [Tooltip("Steps between drawn rings — the horizon spans ringCount*ringStride*DT seconds.")]
    [Min(1)] public int ringStride = 10;

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

            for (int k = 0; k < ringCount; k++)
            {
                float t = (k + 1) * ringStride * dt;
                // Fade with horizon time: the near-term reachable set is the confident one,
                // the far rings are increasingly speculative.
                // Floored at 0.45 — below roughly that, a debug line is too faint to pick out
                // against the ground.
                float fade = Mathf.Max(0.45f, 1f - (float)k / ringCount);
                DrawCapsule(a, b, dSafeHard + vTarget * t, new Color(1f, 0.35f, 0f, fade));
            }
        }
    }

    /// <summary>
    /// Outline of every point within `radius` of segment AB, on the ground plane: two
    /// straight flanks joined by a semicircular cap at each end. A degenerate segment
    /// (a == b) falls back to a full circle, which is exactly what the controllers plan
    /// against in circle mode — same code path, same geometry.
    /// </summary>
    void DrawCapsule(Vector3 a, Vector3 b, float radius, Color color)
    {
        var axis = b - a;
        axis.y = 0f;
        float len = axis.magnitude;

        int capSegments = Mathf.Max(4, circleSegments / 2);

        if (len < 1e-4f)
        {
            // Degenerate: full circle about a. Two back-to-back semicircles rather than a
            // separate routine, so the circle and capsule cases cannot drift apart.
            DrawSemicircle(a, radius, Vector3.right, Vector3.forward, capSegments, color);
            DrawSemicircle(a, radius, -Vector3.right, -Vector3.forward, capSegments, color);
            return;
        }

        var u = axis / len;                              // along the axis
        var n = new Vector3(-u.z, 0f, u.x);              // left normal, ground plane

        // The two flanks of the rectangle.
        Debug.DrawLine(a + n * radius, b + n * radius, color, 0f, false);
        Debug.DrawLine(a - n * radius, b - n * radius, color, 0f, false);

        // End caps: at b the arc sweeps from the left flank round the far end to the right
        // flank; at a it sweeps the other way. Together they close the outline.
        DrawSemicircle(b, radius, n, u, capSegments, color);
        DrawSemicircle(a, radius, -n, -u, capSegments, color);
    }

    /// <summary>
    /// Half-circle of `radius` about `centre`, starting along `from`, bulging toward
    /// `through`, ending along -`from`. Both directions must be unit and perpendicular.
    /// </summary>
    void DrawSemicircle(Vector3 centre, float radius, Vector3 from, Vector3 through,
                        int segments, Color color)
    {
        var prev = centre + from * radius;
        for (int s = 1; s <= segments; s++)
        {
            float ang = Mathf.PI * s / segments;
            var dir = from * Mathf.Cos(ang) + through * Mathf.Sin(ang);
            var next = centre + dir * radius;
            // Zero duration = this frame only. Update() redraws every frame, so any
            // lingering duration would stack stale outlines on top of the live ones.
            // depthTest:false — otherwise the ground plane occludes an outline lying on it
            // and nothing is visible.
            Debug.DrawLine(prev, next, color, 0f, false);
            prev = next;
        }
    }
}
