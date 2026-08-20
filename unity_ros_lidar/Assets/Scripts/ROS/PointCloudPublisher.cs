using UnityEngine;

using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;

using Unity.MLAgents;
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

    [Header("Backend")]
    [Tooltip("Raycast the scan on the GPU with the Robotec GPU Lidar plugin instead of " +
             "Physics.Raycast on the main thread. Both backends publish a byte-identical " +
             "PointCloud2, so the controller cannot tell them apart. Python overrides this " +
             "per-run via the 'use_gpu' environment parameter (launch arg use_gpu:=true).")]
    public bool useGpuLidar = false;
    [Tooltip("Ignore the 'use_gpu' side-channel parameter and always honour the checkbox above. " +
             "For working in the Editor without the controller attached.")]
    public bool ignoreSideChannel = false;

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
    GpuLaserSensor3D gpu_sensor_3d;

    /// <summary>
    /// The CPU sensor instance, for debug viewers (PhantomAgentVisualizer, the occlusion
    /// gizmos). NULL on the GPU backend — those viewers read LaserSensor3D's Unity-side
    /// occlusion pass, which has no GPU equivalent; the controller does its own occlusion
    /// detection from the cloud and is unaffected. Viewers must null-check.
    /// </summary>
    public LaserSensor3D Sensor => laser_sensor_3d;

    /// <summary>Which backend actually came up. May differ from useGpuLidar if RGL failed.</summary>
    public bool UsingGpu => gpu_sensor_3d != null;

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

        m_LastPublishTimeSeconds = Time.timeAsDouble - PublishPeriodSeconds;
    }

    // Backend selection is DEFERRED to the first Update, not done in Start(). Start() runs as
    // soon as Unity connects, which is before Python's first env.reset() — so the side-channel
    // value has not arrived yet and reading it here would always see the Inspector default.
    bool backendReady;

    void EnsureBackend()
    {
        backendReady = true;

        // The side channel wins over the Inspector when the controller is driving, so a run is
        // reproducible from the launch command alone. Read once: the RGL ray set is built from
        // the FOV at construction, so switching mid-run would mean tearing the sensor down and
        // is not worth it for a per-run choice.
        if (!ignoreSideChannel && Academy.IsInitialized)
        {
            float p = Academy.Instance.EnvironmentParameters.GetWithDefault(
                "use_gpu", useGpuLidar ? 1f : 0f);
            useGpuLidar = p > 0.5f;
        }

        if (useGpuLidar && StartGpuBackend())
        {
            Debug.Log("[PointCloudPublisher] LiDAR backend: GPU (RGL)  " +
                      $"{gpu_sensor_3d.NumMeasurementsPerScan_h}x{gpu_sensor_3d.NumMeasurementsPerScan_v} beams");
            return;
        }

        Debug.Log("[PointCloudPublisher] LiDAR backend: CPU (Physics.Raycast)");

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

    /// <summary>
    /// Bring up the RGL backend. Returns false (with the reason logged) if anything is
    /// missing, so Start() can fall back to the CPU sensor rather than publishing nothing —
    /// a silent dead topic is much harder to diagnose than a slow one.
    /// </summary>
    bool StartGpuBackend()
    {
        if (laser_sensor_link == null)
        {
            Debug.LogError("[PointCloudPublisher] use_gpu requested but laser_sensor_link is unassigned.");
            return false;
        }

        try
        {
            gpu_sensor_3d = new GpuLaserSensor3D(
                laser_sensor_link,
                RangeMetersMin, RangeMetersMax,
                fov_horizontal, fov_vertical,
                angularResolution_vertical, angularResolution_horizontal,
                publishMaxRangeOnNoHit, frameIdOverride,
                Mathf.Max(1, Mathf.RoundToInt((float)m_PublishRateHz)));
        }
        catch (System.Exception e)
        {
            // Typically a missing RGL SceneManager in the scene, or the native library failing
            // to load (no CUDA-capable GPU / driver). Either way the CPU path still works.
            Debug.LogError($"[PointCloudPublisher] RGL backend failed to start ({e.GetType().Name}: " +
                           $"{e.Message}) — falling back to the CPU sensor.");
            gpu_sensor_3d = null;
            return false;
        }

        // RGL captures on its own FixedUpdate cadence; publish off its callback rather than
        // polling in Update, so a scan is sent exactly once and never re-sent or missed.
        gpu_sensor_3d.AddOnNewData(PublishGpuScan);
        return true;
    }

    void PublishGpuScan()
    {
        var msg = gpu_sensor_3d.getScanMsg();
        if (msg == null) return;                 // no scan produced yet

        ros.Publish(point_cloud_topic, msg);
        ros.Publish(pose_topic, BuildPoseMsg());
        m_LastPublishTimeSeconds = Time.timeAsDouble;
    }

    /// <summary>Sensor-link pose. Shared: the controller world-places every LiDAR-frame
    /// detection with it, so both backends must publish the same thing alongside the cloud.</summary>
    PoseMsg BuildPoseMsg()
    {
        var t = laser_sensor_link.transform;
        return new PoseMsg
        {
            position = new PointMsg(t.position.x, t.position.y, t.position.z),
            orientation = new QuaternionMsg(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w),
        };
    }

    void OnDestroy()
    {
        if (gpu_sensor_3d != null)
        {
            gpu_sensor_3d.RemoveOnNewData(PublishGpuScan);
            gpu_sensor_3d.Dispose();
        }
    }

    void Update()
    {
        if (!backendReady)
            EnsureBackend();

        if (gpu_sensor_3d != null)
            return;                              // GPU path publishes from RGL's callback

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

        ros.Publish(point_cloud_topic, point_cloud_msg);
        ros.Publish(pose_topic, BuildPoseMsg());

        m_LastPublishTimeSeconds = Time.timeAsDouble;
    }
}