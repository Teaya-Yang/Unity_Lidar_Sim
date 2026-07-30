using System.Text;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

// Publishes the footprints of the static obstacles listed in the Inspector, so the
// Python-side trajectory plot can draw the real walls/buildings instead of inferring
// them from the LiDAR. Plot-only: no planner reads this topic.
//
// Payload: JSON on std_msgs/String — {"boxes":[{cx,cy,sx,sy,yaw}, ...]}, a 2-D oriented
// footprint per obstacle in the PLANNER's world axes: x = Unity z, y = Unity x,
// yaw = +eulerY — the same axes TaxiAgent.CollectObservations reports the ego and goal in
// (ego0 = position.z, ego1 = position.x, theta = +eulerAngles.y).
//
// NOTE: this is NOT the ROS convention GroundTruthPublisher uses (y = -Unity x,
// yaw = -eulerY). Negating x here puts every obstacle on the wrong side of the map,
// mirrored about the taxiway, so do not "fix" it to match that file.
//
// The scene geometry never moves, so this is a slow heartbeat rather than a stream —
// it only exists so a subscriber that connects late still gets the map.
public class StaticObstaclePublisher : MonoBehaviour
{
    [Header("ROS")]
    public string topic = "/static_obstacles";
    public double republishPeriod = 1.0;

    [Header("Obstacles")]
    [Tooltip("The static obstacles to publish — drag in the walls / buildings / parked " +
             "aircraft you want drawn. Each object's own Collider (or Renderer, if it has " +
             "no Collider) defines the footprint; child colliders are included, so " +
             "dragging in a parent publishes one box per part.")]
    public GameObject[] obstacles;

    ROSConnection m_Ros;
    double m_LastPublish;

    void Start()
    {
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterPublisher<StringMsg>(topic);
        m_LastPublish = -1e9;

        if (obstacles == null || obstacles.Length == 0)
            Debug.LogWarning("[StaticObstaclePublisher] No obstacles assigned — the plot will " +
                             "have no static geometry.", this);
    }

    void Update()
    {
        if (Time.timeAsDouble < m_LastPublish + republishPeriod) return;
        m_LastPublish = Time.timeAsDouble;
        m_Ros.Publish(topic, new StringMsg(BuildJson()));
    }

    string BuildJson()
    {
        var sb = new StringBuilder(2048);
        sb.Append("{\"boxes\":[");
        int written = 0;

        if (obstacles != null)
        {
            foreach (GameObject go in obstacles)
            {
                if (go == null) continue;

                Collider[] cols = go.GetComponentsInChildren<Collider>();
                if (cols.Length > 0)
                {
                    foreach (Collider c in cols)
                    {
                        if (c.isTrigger) continue;
                        AppendBox(sb, ref written, c.transform, LocalBounds(c));
                    }
                }
                else
                {
                    // No collider: the LiDAR cannot see it either, but it may still be
                    // scene context worth drawing, so fall back to the render bounds.
                    foreach (Renderer r in go.GetComponentsInChildren<Renderer>())
                        AppendBox(sb, ref written, r.transform,
                                  r.localBounds);
                }
            }
        }

        sb.Append("]}");
        return sb.ToString();
    }

    // Local bounds + the transform become an ORIENTED footprint. An axis-aligned world
    // AABB would inflate every rotated wall into a square and lose the very boundary
    // the plot is meant to show, so the yaw is carried in the message instead.
    static void AppendBox(StringBuilder sb, ref int written, Transform t, Bounds local)
    {
        Vector3 s = t.lossyScale;
        Vector3 center = t.TransformPoint(local.center);

        float cx = center.z;                        // planner x  = Unity Z
        float cy = center.x;                        // planner y  = Unity X
        float sx = Mathf.Abs(local.size.z * s.z);   // extent along planner x
        float sy = Mathf.Abs(local.size.x * s.x);   // extent along planner y
        float yaw = t.eulerAngles.y * Mathf.Deg2Rad;

        if (written++ > 0) sb.Append(",");
        // Closing brace appended separately: `{yaw:F4}}}` makes the compiler read the `}}`
        // as an escaped brace INSIDE the format clause, emitting the literal "F4}" and
        // dropping the value. Same trap documented in GroundTruthPublisher.AppendBbox.
        sb.Append($"{{\"cx\":{cx:F2},\"cy\":{cy:F2},\"sx\":{sx:F2},\"sy\":{sy:F2},\"yaw\":{yaw:F4}");
        sb.Append("}");
    }

    static Bounds LocalBounds(Collider c)
    {
        switch (c)
        {
            case BoxCollider bc:     return new Bounds(bc.center, bc.size);
            case MeshCollider mc when mc.sharedMesh != null:
                                     return mc.sharedMesh.bounds;
            case CapsuleCollider cc: return new Bounds(cc.center,
                                         new Vector3(2f * cc.radius, cc.height, 2f * cc.radius));
            case SphereCollider sc:  return new Bounds(sc.center, 2f * sc.radius * Vector3.one);
            default:
                Bounds wb = c.bounds;   // unknown type: express the world AABB locally
                return new Bounds(c.transform.InverseTransformPoint(wb.center),
                                  c.transform.InverseTransformVector(wb.size));
        }
    }
}
