using UnityEngine;

using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;

using Unity.Robotics.ROSTCPConnector;

public class PointCloudPublisher : MonoBehaviour
{
    public GameObject laser_sensor_link;
    public string point_cloud_topic = "/point_cloud";
    public string pose_topic = "/laser_scan_pose";

    public float RangeMetersMin = 0;
    public float RangeMetersMax = 1000;

    public float fov_horizontal = 360;
    public float fov_vertical = 45;

    public float angularResolution_vertical = 1;
    public float angularResolution_horizontal = 1;

    public bool publishMaxRangeOnNoHit = false;
    public string frameIdOverride = "";

    public double m_PublishRateHz = 10.0;

    [Header("Debug visualization")]
    [Tooltip("Draw scan rays in the Scene view (green = hit, red = no return).")]
    public bool drawDebugRays = false;
    [Tooltip("Draw every Nth ray in both azimuth and elevation. 1 = every ray (slow).")]
    [Min(1)] public int debugRayStride = 8;
    [Tooltip("Mark every point where a ray hit geometry (cyan = near, magenta = far). Shows the boundary as the LiDAR actually sees it.")]
    public bool drawHitPoints = false;
    [Tooltip("Size of each hit-point cross, in metres.")]
    public float hitPointSize = 0.15f;

    [Tooltip("Detect occlusion edges from range jumps between adjacent beams and draw the blind-corner mouths (yellow = shadow mouth, red = corner point).")]
    public bool drawOcclusionEdges = false;
    [Tooltip("Most oblique surface still treated as continuous. Lower = fewer edges detected.")]
    [Range(1f, 45f)] public float grazingDeg = 12f;
    [Tooltip("Absolute floor on a reportable range step, in metres.")]
    public float minJumpMeters = 0.5f;
    [Tooltip("Corners closer than this collapse into one, so a single vertical edge seeds one phantom rather than a stack.")]
    public float cornerMergeRadius = 1.5f;

    [Header("Occlusion forward wedge (mirror of OCC_FWD_HALF_ANGLE)")]
    [Tooltip("Draw the wedge within which a detected boundary may seed a keep-out (cyan edges + arc, green heading tick).")]
    public bool drawOcclusionFov = false;
    [Tooltip("Discard corners outside the wedge, as taxi_controller_mpc.py does. Off = detect all round (the wedge is then only drawn, not applied).")]
    public bool filterCornersToFov = true;
    [Tooltip("OCC_FWD_HALF_ANGLE [deg]. MUST match taxi_controller_mpc.py or the Scene view disagrees with what the MPC constrains. 180 = no filter, 90 = strictly ahead.")]
    [Range(0f, 180f)] public float occFwdHalfAngleDeg = 100f;
    [Tooltip("Radius the wedge arc is drawn at [m]. Match OCC_QUERY_R (60).")]
    public float fovDrawRadius = 60f;

    ROSConnection ros;
    LaserSensor3D laser_sensor_3d;

    /// <summary>The sensor instance, for debug viewers (PhantomAgentVisualizer). Null before Start().</summary>
    public LaserSensor3D Sensor => laser_sensor_3d;

    double m_LastPublishTimeSeconds;
    double PublishPeriodSeconds => 1.0 / m_PublishRateHz;

    bool ShouldPublishMessage =>
        Time.timeAsDouble >= m_LastPublishTimeSeconds + PublishPeriodSeconds;

    void Start()
    {
        if (m_PublishRateHz <= 0.0)
        {
            Debug.LogWarning($"{nameof(PointCloudPublisher)} publish rate must be > 0. Using 10 Hz.");
            m_PublishRateHz = 10.0;
        }

        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<PointCloud2Msg>(point_cloud_topic);
        ros.RegisterPublisher<PoseMsg>(pose_topic);

        laser_sensor_3d = new LaserSensor3D(
            laser_sensor_link,
            RangeMetersMin,
            RangeMetersMax,
            fov_horizontal,
            fov_vertical,
            angularResolution_vertical,
            angularResolution_horizontal,
            publishMaxRangeOnNoHit,
            frameIdOverride
        );

        // Publish immediately on the first eligible Update.
        laser_sensor_3d.drawDebugRays = drawDebugRays;
        laser_sensor_3d.debugRayStride = Mathf.Max(1, debugRayStride);
        laser_sensor_3d.debugRayDuration = (float)PublishPeriodSeconds;
        laser_sensor_3d.drawHitPoints = drawHitPoints;
        laser_sensor_3d.hitPointSize = hitPointSize;
        laser_sensor_3d.drawOcclusionEdges = drawOcclusionEdges;
        laser_sensor_3d.grazingDeg = grazingDeg;
        laser_sensor_3d.minJumpMeters = minJumpMeters;
        laser_sensor_3d.cornerMergeRadius = cornerMergeRadius;
        laser_sensor_3d.drawOcclusionFov = drawOcclusionFov;
        laser_sensor_3d.filterCornersToFov = filterCornersToFov;
        laser_sensor_3d.occFwdHalfAngleDeg = occFwdHalfAngleDeg;
        laser_sensor_3d.fovDrawRadius = fovDrawRadius;

        m_LastPublishTimeSeconds = Time.timeAsDouble - PublishPeriodSeconds;
    }

    void Update()
    {
        if (!ShouldPublishMessage)
            return;

        // Debug.Log("Publishing point cloud");

        if (laser_sensor_3d == null)
        {
            Debug.LogError("[PointCloudPublisher] laser_sensor_3d is null — Start() failed. Check laser_sensor_link assignment and Console for earlier errors.");
            enabled = false;
            return;
        }
        // Re-read so the toggle can be flipped in the Inspector while playing.
        laser_sensor_3d.drawDebugRays = drawDebugRays;
        laser_sensor_3d.debugRayStride = Mathf.Max(1, debugRayStride);
        laser_sensor_3d.drawHitPoints = drawHitPoints;
        laser_sensor_3d.hitPointSize = hitPointSize;
        laser_sensor_3d.drawOcclusionEdges = drawOcclusionEdges;
        laser_sensor_3d.grazingDeg = grazingDeg;
        laser_sensor_3d.minJumpMeters = minJumpMeters;
        laser_sensor_3d.cornerMergeRadius = cornerMergeRadius;
        laser_sensor_3d.drawOcclusionFov = drawOcclusionFov;
        laser_sensor_3d.filterCornersToFov = filterCornersToFov;
        laser_sensor_3d.occFwdHalfAngleDeg = occFwdHalfAngleDeg;
        laser_sensor_3d.fovDrawRadius = fovDrawRadius;

        PointCloud2Msg point_cloud_msg = laser_sensor_3d.getScanMsg();

        PoseMsg pose_msg = new PoseMsg
        {
            position = new PointMsg(
                laser_sensor_link.transform.position.x,
                laser_sensor_link.transform.position.y,
                laser_sensor_link.transform.position.z
            ),
            orientation = new QuaternionMsg(
                laser_sensor_link.transform.rotation.x,
                laser_sensor_link.transform.rotation.y,
                laser_sensor_link.transform.rotation.z,
                laser_sensor_link.transform.rotation.w
            ),
        };

        ros.Publish(point_cloud_topic, point_cloud_msg);
        ros.Publish(pose_topic, pose_msg);

        m_LastPublishTimeSeconds = Time.timeAsDouble;
    }
}