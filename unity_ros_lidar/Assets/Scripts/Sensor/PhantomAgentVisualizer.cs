using UnityEngine;

/// <summary>
/// Draws the forward-reachable-set keep-out circles the occlusion-aware controllers
/// actually plan against (Firoozi et al.). A worst-case hidden agent is assumed to sit
/// on each blind-corner mouth and advance at up to V_TARGET, so the forbidden radius
/// grows along the prediction horizon:
///
///     r_keep(t) = D_SAFE_OCC  + V_TARGET * t     (hard keep-out)
///     r_infl(t) = D_INFL_OCC  + V_TARGET * t     (soft influence ring)
///
/// Corner points come from LaserSensor3D's range-jump edge pass, the Unity-side analogue
/// of LidarCostmap.occlusion_points that the controllers consume.
///
/// The constants below MUST track taxi_controller.py, which is the single source of truth
/// for both MPPI and MPC. This script is a viewer only — it never feeds the controller,
/// so a mismatch here shows a misleading circle rather than changing behaviour.
/// </summary>
[RequireComponent(typeof(PointCloudPublisher))]
public class PhantomAgentVisualizer : MonoBehaviour
{
    [Header("Hidden-agent model (mirror of taxi_controller.py)")]
    [Tooltip("V_TARGET — assumed max speed of a hidden agent emerging from occlusion [m/s].")]
    public float vTarget = 3.0f;
    [Tooltip("D_SAFE_OCC — base (t=0) hard keep-out radius around an occlusion boundary [m].")]
    public float dSafeOcc = 16.0f;
    [Tooltip("D_INFL_OCC — base (t=0) soft-influence radius [m]. Must stay >= D_SAFE_OCC.")]
    public float dInflOcc = 24.0f;
    [Tooltip("OCC_QUERY_R — ignore corners further than this from the sensor [m].")]
    public float occQueryR = 60.0f;

    [Header("Horizon")]
    [Tooltip("DT — controller timestep [s].")]
    public float dt = 0.1f;
    [Tooltip("Rings drawn across the horizon. Each is one t_k = (k+1)*DT*ringStride.")]
    [Min(1)] public int ringCount = 5;
    [Tooltip("Steps between drawn rings — the horizon spans ringCount*ringStride*DT seconds.")]
    [Min(1)] public int ringStride = 10;

    [Header("Drawing")]
    [Tooltip("Segments per circle. Higher = smoother, more draw calls.")]
    [Min(8)] public int circleSegments = 48;
    [Tooltip("Draw the outermost soft-influence ring too.")]
    public bool showInfluenceRing = true;
    [Tooltip("Lift circles this far above the drawing plane [m].")]
    public float groundOffset = 0.05f;
    [Tooltip("Flatten all circles onto one horizontal plane this far BELOW the sensor, instead of leaving them at the corner's own hit height (which is often partway up a wall).")]
    public bool projectToGroundPlane = true;
    [Tooltip("Sensor height above the driving plane [m]. Only used when projecting.")]
    public float sensorHeight = 2.0f;

    PointCloudPublisher publisher;

    void Awake()
    {
        publisher = GetComponent<PointCloudPublisher>();
    }

    void OnValidate()
    {
        // The soft ring must enclose the hard one or the cost field is inverted.
        if (dInflOcc < dSafeOcc) dInflOcc = dSafeOcc;
    }

    void Update()
    {
        var sensor = publisher != null ? publisher.Sensor : null;
        if (sensor == null)
            return;

        // Ask the sensor to run its edge pass even when the debug lines are off.
        sensor.computeOcclusionEdges = true;

        var corners = sensor.OcclusionCorners;
        if (corners == null || corners.Count == 0)
            return;

        var origin = publisher.laser_sensor_link.transform.position;
        float queryRSq = occQueryR * occQueryR;

        foreach (var corner in corners)
        {
            if ((corner - origin).sqrMagnitude > queryRSq)
                continue;

            var centre = corner + Vector3.up * groundOffset;
            if (projectToGroundPlane)
                centre.y = origin.y - sensorHeight + groundOffset;

            for (int k = 0; k < ringCount; k++)
            {
                float t = (k + 1) * ringStride * dt;
                float grow = vTarget * t;
                // Fade with horizon time: the near-term reachable set is the confident one,
                // the far rings are increasingly speculative.
                // Floored at 0.45 — below roughly that, a debug line is too faint to pick out
                // against the ground.
                float fade = Mathf.Max(0.45f, 1f - (float)k / ringCount);

                DrawCircle(centre, dSafeOcc + grow, new Color(1f, 0.35f, 0f, fade));
                if (showInfluenceRing && k == ringCount - 1)
                    DrawCircle(centre, dInflOcc + grow, new Color(1f, 0.85f, 0.2f, 0.35f));
            }
        }
    }

    void DrawCircle(Vector3 centre, float radius, Color color)
    {
        float step = 2f * Mathf.PI / circleSegments;
        var prev = centre + new Vector3(radius, 0f, 0f);
        for (int s = 1; s <= circleSegments; s++)
        {
            float a = s * step;
            var next = centre + new Vector3(Mathf.Cos(a) * radius, 0f, Mathf.Sin(a) * radius);
            // Zero duration = this frame only. Update() redraws every frame, so any
            // lingering duration would stack stale circles on top of the live ones.
            // depthTest:false — otherwise the ground plane occludes a circle lying on it
            // and nothing is visible.
            Debug.DrawLine(prev, next, color, 0f, false);
            prev = next;
        }
    }
}
