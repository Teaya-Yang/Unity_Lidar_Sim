using System.Text;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

// Publishes the footprints of the MOVING objects listed in the Inspector, so the Python-
// side trajectory plot can draw where they truly were at the solve it visualises, instead
// of only the LiDAR-clustered estimates. Plot-only: no planner reads this topic.
//
// Same payload and axes as StaticObstaclePublisher (planner world frame: x = Unity z,
// y = Unity x, yaw = +eulerY) — NOT the ROS convention of GroundTruthPublisher. The one
// difference is the rate: these move, so this streams instead of a heartbeat.
public class DynamicObstaclePublisher : MonoBehaviour
{
    [Header("ROS")]
    public string topic = "/dynamic_obstacles";
    public double rateHz = 20.0;

    [Header("Obstacles")]
    [Tooltip("Scene objects that move and are never re-created — drag them in directly. " +
             "Each object's own Collider (or Renderer, if it has no Collider) defines the " +
             "footprint; child colliders are included.")]
    public GameObject[] obstacles;

    [Tooltip("Roots whose ACTIVE descendants are published every frame. Use this for the " +
             "scenario agents: TaxiScenarioManager pools clones of incursionPrefab under " +
             "its own transform at runtime, so they cannot be dragged into the list above " +
             "— drag in the TaxiScenarioManager object instead. Re-scanned each publish, " +
             "so agents that spawn or are pooled in/out mid-episode are picked up.")]
    public Transform[] agentRoots;

    [Tooltip("Publish ONE footprint per object — the hull of all its non-trigger colliders " +
             "in the object's own frame. Off, each sub-collider becomes its own box, which " +
             "draws a vehicle as a scatter of slivers (wheels, mirrors) rather than a body.")]
    public bool mergeSubColliders = true;

    ROSConnection m_Ros;
    double m_LastPublish;
    int m_Written;        // footprints in the last message — for the one-time sanity log
    int m_AgentsFound;    // how many of them came from agentRoots
    bool m_Logged;

    void Start()
    {
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterPublisher<StringMsg>(topic);
        m_LastPublish = -1e9;

        if ((obstacles == null || obstacles.Length == 0) &&
            (agentRoots == null || agentRoots.Length == 0))
            Debug.LogWarning("[DynamicObstaclePublisher] Nothing assigned — the plot will " +
                             "have no dynamic objects.", this);
    }

    void Update()
    {
        if (rateHz <= 0.0 || Time.timeAsDouble < m_LastPublish + 1.0 / rateHz) return;
        m_LastPublish = Time.timeAsDouble;
        string json = BuildJson();
        // Log the SPLIT, not just the total: a message made entirely of static-list
        // footprints looks healthy on the topic while containing no agent at all.
        if (!m_Logged && m_Written > 0)
        {
            m_Logged = true;
            Debug.Log($"[DynamicObstaclePublisher] {m_Written} footprint(s) on {topic}: " +
                      $"{m_AgentsFound} agent(s) under agentRoots, the rest from the " +
                      $"Obstacles list", this);
            if (agentRoots != null && agentRoots.Length > 0 && m_AgentsFound == 0)
                Debug.LogWarning("[DynamicObstaclePublisher] agentRoots is assigned but no " +
                                 "ACTIVE IncursionAgentController was found under it — the " +
                                 "plot will show only the static list. Drag in the object " +
                                 "that owns TaxiScenarioManager (the pooled clones are " +
                                 "parented to its transform).", this);
        }
        m_Ros.Publish(topic, new StringMsg(json));
    }

    string BuildJson()
    {
        var sb = new StringBuilder(1024);
        sb.Append("{\"boxes\":[");
        int written = 0;

        if (obstacles != null)
            foreach (GameObject go in obstacles)
                AppendObject(sb, ref written, go);

        // Runtime agents: the pooled clones live under the root and are activated /
        // deactivated per episode, so only the ACTIVE ones are published. Excluding
        // inactive components here is what keeps a pooled-out agent off the plot.
        int agents = 0;
        if (agentRoots != null)
        {
            foreach (Transform root in agentRoots)
            {
                if (root == null) continue;
                foreach (IncursionAgentController a in
                         root.GetComponentsInChildren<IncursionAgentController>(false))
                {
                    AppendObject(sb, ref written, a.gameObject);
                    agents++;
                }
            }
        }

        sb.Append("]}");
        m_Written = written;
        m_AgentsFound = agents;
        return sb.ToString();
    }

    void AppendObject(StringBuilder sb, ref int written, GameObject go)
    {
        if (go == null || !go.activeInHierarchy) return;

        Collider[] cols = go.GetComponentsInChildren<Collider>();
        if (cols.Length > 0)
        {
            if (mergeSubColliders)
            {
                Bounds merged;
                if (MergedLocalBounds(go.transform, cols, out merged))
                    AppendBox(sb, ref written, go.transform, merged);
            }
            else
            {
                foreach (Collider c in cols)
                {
                    if (c.isTrigger) continue;
                    AppendBox(sb, ref written, c.transform, LocalBounds(c));
                }
            }
        }
        else
        {
            // No collider: the LiDAR cannot see it either, but it may still be worth drawing.
            foreach (Renderer r in go.GetComponentsInChildren<Renderer>())
                AppendBox(sb, ref written, r.transform, r.localBounds);
        }
    }

    // Every non-trigger collider in the subtree, expressed in the OBJECT's own local frame
    // and encapsulated into one box. Done in the object frame, not world: a world AABB of a
    // vehicle driving diagonally is a square that breathes as it turns, whereas this stays
    // the vehicle's actual hull and rides along with its yaw.
    static bool MergedLocalBounds(Transform root, Collider[] cols, out Bounds merged)
    {
        merged = default;
        bool any = false;

        foreach (Collider c in cols)
        {
            if (c.isTrigger) continue;
            Bounds lb = LocalBounds(c);
            Vector3 e = lb.extents;

            for (int k = 0; k < 8; k++)
            {
                Vector3 corner = lb.center + new Vector3(
                    (k & 1) == 0 ? -e.x : e.x,
                    (k & 2) == 0 ? -e.y : e.y,
                    (k & 4) == 0 ? -e.z : e.z);
                Vector3 p = root.InverseTransformPoint(c.transform.TransformPoint(corner));
                if (!any) { merged = new Bounds(p, Vector3.zero); any = true; }
                else merged.Encapsulate(p);
            }
        }
        return any;
    }

    // Local bounds + the transform become an ORIENTED footprint: a world AABB would
    // inflate every turning vehicle into a square that grows and shrinks as it drives.
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
        // Closing brace appended separately — see the note in StaticObstaclePublisher.
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
                Bounds wb = c.bounds;
                return new Bounds(c.transform.InverseTransformPoint(wb.center),
                                  c.transform.InverseTransformVector(wb.size));
        }
    }
}
