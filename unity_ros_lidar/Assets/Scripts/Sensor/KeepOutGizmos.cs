using UnityEngine;

/// <summary>
/// Debug-line outlines for the keep-out regions the controllers plan against.
///
/// ONE definition of the geometry, shared by every viewer that draws a keep-out
/// (PhantomAgentVisualizer for occlusion boundaries, DynamicAgentVisualizer for sensed
/// moving clusters). The Python side already does this — both keep-outs go through the
/// single occlusion_stage_cost / capsule_polygon pair in controller/ — so duplicating the
/// outline code on the Unity side would be the one place the two hazards could drift
/// apart visually while being identical in the cost.
///
/// Everything is drawn with Debug.DrawLine at zero duration: Scene view only, redrawn
/// every frame by the caller's Update(), and depthTest disabled so an outline lying on the
/// ground plane is not swallowed by it.
/// </summary>
public static class KeepOutGizmos
{
    /// <summary>
    /// Outline of every point within `radius` of segment AB, on the ground plane: two
    /// straight flanks joined by a semicircular cap at each end. A degenerate segment
    /// (a == b) falls back to a full circle, which is exactly what the controllers plan
    /// against for a point-anchored keep-out — same code path, same geometry.
    /// </summary>
    public static void DrawCapsule(Vector3 a, Vector3 b, float radius, Color color,
                                   int circleSegments = 48)
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

    /// <summary>Full circle on the ground plane — the degenerate capsule, named.</summary>
    public static void DrawCircle(Vector3 centre, float radius, Color color,
                                  int circleSegments = 48)
    {
        DrawCapsule(centre, centre, radius, color, circleSegments);
    }

    /// <summary>
    /// Half-circle of `radius` about `centre`, starting along `from`, bulging toward
    /// `through`, ending along -`from`. Both directions must be unit and perpendicular.
    /// </summary>
    public static void DrawSemicircle(Vector3 centre, float radius, Vector3 from,
                                      Vector3 through, int segments, Color color)
    {
        var prev = centre + from * radius;
        for (int s = 1; s <= segments; s++)
        {
            float ang = Mathf.PI * s / segments;
            var dir = from * Mathf.Cos(ang) + through * Mathf.Sin(ang);
            var next = centre + dir * radius;
            // Zero duration = this frame only. The caller redraws every frame, so any
            // lingering duration would stack stale outlines on top of the live ones.
            // depthTest:false — otherwise the ground plane occludes an outline lying on it
            // and nothing is visible.
            Debug.DrawLine(prev, next, color, 0f, false);
            prev = next;
        }
    }
}
