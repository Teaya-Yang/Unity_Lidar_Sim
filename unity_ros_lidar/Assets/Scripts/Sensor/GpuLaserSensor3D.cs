using System;
using UnityEngine;

using RGLUnityPlugin;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using Unity.Robotics.Core;

/// <summary>
/// GPU-raycast drop-in for <see cref="LaserSensor3D"/>, backed by the Robotec GPU Lidar
/// plugin. Produces a PointCloud2 that is BYTE-COMPATIBLE with the CPU sensor's, so the
/// controller cannot tell which one produced a scan.
///
/// The compatibility is the whole point and it is not free — four things have to match:
///
///  1. ORDERED, NOT COMPACTED. obstacle_circles.parse_ordered_cloud reshapes the cloud to
///     exactly (n_h, n_v) and returns None on any count mismatch, which would silently
///     disable ALL occlusion detection. RGL compacts hits by default, so this connects to
///     the raytrace output via ConnectToWorldFrame(..., compacted: false) — one point per
///     ray, in ray order, misses included and flagged by IS_HIT_I32.
///
///  2. RAY ORDER. RGL emits idx = laserId + hStep * nLasers, i.e. elevation-minor within
///     azimuth. LaserSensor3D's loop nests v inside h, giving the same idx = v + h * n_v.
///
///  3. BEAM ANGLES. Built here from the same fov/resolution arithmetic (including the
///     360-degree wrap de-duplication) rather than from an RGL model preset, because
///     occlusion_capsules.scan_shape() re-derives the grid from those numbers on the ROS
///     side and a preset's ray count would not agree with it.
///
///  4. POINT FRAME. LaserSensor3D emits hit MINUS sensor position with the axes swapped —
///     a translation only, NOT a rotation into the sensor frame (see the README note). RGL's
///     built-in lidar-frame output applies the full inverse pose, so it is deliberately not
///     used: the world-frame cloud is shifted by the sensor position on the CPU instead.
///     The per-point work left on the CPU is one subtract and a hit test; the raycasting,
///     which is what actually costs, is on the GPU.
/// </summary>
public class GpuLaserSensor3D
{
    public float RangeMetersMin;
    public float RangeMetersMax;
    public float fov_horizontal;
    public float fov_vertical;
    public float angularResolution_vertical;
    public float angularResolution_horizontal;
    public bool  publishMaxRangeOnNoHit;

    public string FrameId { get; private set; }
    public int NumMeasurementsPerScan_h { get; private set; }
    public int NumMeasurementsPerScan_v { get; private set; }

    readonly GameObject laser_sensor_link;
    readonly LidarSensor lidar;
    readonly RGLNodeSequence outputGraph;

    // XYZ (12 B) + IS_HIT as int32 (4 B). Same width as the published point, by luck rather
    // than design — the payload is rebuilt below either way.
    const int RglPointStep = 16;
    const int OutPointStep = 16;

    static readonly RGLField[] Fields = { RGLField.XYZ_VEC3_F32, RGLField.IS_HIT_I32 };

    byte[] rglData = Array.Empty<byte>();
    readonly byte[] raw_data;
    readonly uint   raw_data_len;
    readonly uint   numPoints;

    bool warnedCount;

    public GpuLaserSensor3D(
        GameObject _laser_sensor_link,
        float _RangeMetersMin,
        float _RangeMetersMax,
        float _fov_horizontal,
        float _fov_vertical,
        float _angularResolution_vertical,
        float _angularResolution_horizontal,
        bool _publishMaxRangeOnNoHit = false,
        string _frameIdOverride = "",
        int _captureHz = 10)
    {
        laser_sensor_link = _laser_sensor_link;

        RangeMetersMin = _RangeMetersMin;
        RangeMetersMax = _RangeMetersMax;
        fov_horizontal = _fov_horizontal <= 360 ? _fov_horizontal : 360;
        fov_vertical   = _fov_vertical   <= 360 ? _fov_vertical   : 360;
        angularResolution_vertical   = Mathf.Max(0.001f, _angularResolution_vertical);
        angularResolution_horizontal = Mathf.Max(0.001f, _angularResolution_horizontal);
        publishMaxRangeOnNoHit = _publishMaxRangeOnNoHit;

        var fallbackFrame = laser_sensor_link.name.Replace(" ", "_");
        FrameId = string.IsNullOrWhiteSpace(_frameIdOverride) ? fallbackFrame : _frameIdOverride;

        // Grid size, mirroring LaserSensor3D's constructor arithmetic exactly — including its
        // quirk of sizing the horizontal count with the HORIZONTAL resolution and the vertical
        // count with the vertical one, and dropping the duplicated beam at a full 360.
        float start_h = -fov_horizontal / 2f, end_h = fov_horizontal / 2f;
        float start_v = -fov_vertical   / 2f, end_v = fov_vertical   / 2f;

        NumMeasurementsPerScan_h = Mathf.FloorToInt((end_h - start_h) / angularResolution_horizontal) + 1;
        if (fov_horizontal == 360) NumMeasurementsPerScan_h -= 1;
        NumMeasurementsPerScan_v = Mathf.FloorToInt((end_v - start_v) / angularResolution_vertical) + 1;
        if (fov_vertical == 360) NumMeasurementsPerScan_v -= 1;

        numPoints    = (uint)(NumMeasurementsPerScan_h * NumMeasurementsPerScan_v);
        raw_data_len = OutPointStep * numPoints;
        raw_data     = new byte[raw_data_len];

        // RGL raytraces against ITS OWN scene mirror, which SceneManager maintains. Without one
        // LidarSensor.Start() destroys itself and the topic goes quiet, so create it here
        // rather than making the scene carry a prefab whose only job is to exist. AddComponent
        // runs Awake synchronously, which is what assigns the singleton.
        if (RGLUnityPlugin.SceneManager.Instance == null)
        {
            var host = new GameObject("RGLSceneManager");
            var sm = host.AddComponent<RGLUnityPlugin.SceneManager>();
            if (RGLUnityPlugin.SceneManager.Instance == null)
                throw new InvalidOperationException(
                    "failed to create an RGL SceneManager — the native library is probably not " +
                    "loadable (no CUDA-capable GPU or driver)");
            SetMeshSourceToColliders(sm);
        }

        // The stock RGL publisher would emit a second, COMPACTED cloud on the same topic and
        // the two would interleave, so stand it down if the scene still has one wired up.
        var stock = laser_sensor_link.GetComponent<RglTcpPointCloudPublisher>();
        if (stock != null && stock.enabled)
        {
            Debug.LogWarning("[GpuLaserSensor3D] disabling RglTcpPointCloudPublisher on " +
                             $"'{laser_sensor_link.name}': it publishes a compacted cloud on the " +
                             "same topic, which the controller's occlusion pass cannot parse.");
            stock.enabled = false;
        }

        // RGL needs a LidarSensor component on the sensor link; add one if the scene has not.
        lidar = laser_sensor_link.GetComponent<LidarSensor>();
        if (lidar == null) lidar = laser_sensor_link.AddComponent<LidarSensor>();

        lidar.AutomaticCaptureHz = _captureHz;
        // Noise off: the CPU sensor is noiseless, and the two backends must be comparable.
        lidar.applyDistanceGaussianNoise = false;
        lidar.applyAngularGaussianNoise  = false;
        lidar.doValidateConfigurationOnStartup = false;
        lidar.configuration = BuildConfiguration();

        outputGraph = new RGLNodeSequence().AddNodePointsFormat("TAXI_FORMAT", Fields);
        // compacted: false — see (1) in the class doc. This is the load-bearing argument.
        lidar.ConnectToWorldFrame(outputGraph, false);
    }

    /// <summary>
    /// Force SceneManager onto OnlyColliders. Two reasons, both load-bearing:
    ///
    ///  * CORRECTNESS. The default (RegularMeshesAndSkinnedMeshes) uploads RENDERER meshes,
    ///    and Unity meshes are not CPU-readable unless Read/Write is ticked at import — so
    ///    most uploads throw "isReadable is false" and RGL's scene mirror comes up nearly
    ///    empty. The scan then returns a handful of stray far hits and no ground, and the
    ///    controller builds zero obstacle circles.
    ///  * PARITY. LaserSensor3D raycasts with Physics.Raycast, which sees colliders. Anything
    ///    else would make the two backends disagree about what is in the world.
    ///
    /// meshSource is a private [SerializeField], so this goes through reflection. If the
    /// plugin ever renames it the set fails loudly rather than silently reverting to the
    /// broken default.
    /// </summary>
    static void SetMeshSourceToColliders(RGLUnityPlugin.SceneManager sm)
    {
        const string field = "meshSource";
        var f = typeof(RGLUnityPlugin.SceneManager).GetField(
            field, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (f == null)
        {
            Debug.LogError($"[GpuLaserSensor3D] RGL SceneManager has no '{field}' field any more — " +
                           "set Mesh Source = OnlyColliders by hand on the RGLSceneManager object, " +
                           "or the GPU scan will be empty (renderer meshes are not readable).");
            return;
        }

        f.SetValue(sm, RGLUnityPlugin.SceneManager.MeshSource.OnlyColliders);
        // OnValidate() is what turns meshSource into the strategy delegate; without it the
        // field is set but the already-chosen mesh strategy still stands.
        sm.SendMessage("OnValidate", SendMessageOptions.DontRequireReceiver);
        Debug.Log("[GpuLaserSensor3D] created an RGLSceneManager with Mesh Source = OnlyColliders " +
                  "(matches the CPU sensor, which raycasts colliders).");
    }

    /// <summary>Ray set matching LaserSensor3D's beams one for one.</summary>
    BaseLidarConfiguration BuildConfiguration()
    {
        // LaserSensor3D's direction is (cos(psi)sin(theta), -sin(psi), cos(psi)cos(theta)) with
        // psi stepping up from -fov_v/2 — so POSITIVE psi aims DOWN. RGL's LaserArray builds a
        // laser from an elevation where positive is UP, hence the negated elevation below.
        var lasers = new Laser[NumMeasurementsPerScan_v];
        for (int j = 0; j < NumMeasurementsPerScan_v; j++)
        {
            float psi = -fov_vertical / 2f + j * angularResolution_vertical;
            lasers[j] = new Laser
            {
                horizontalAngularOffsetDeg = 0f,
                verticalAngularOffsetDeg   = -psi,
                verticalLinearOffsetMm     = 0f,
                ringId                     = j,
                timeOffset                 = 0f,
                minRange                   = RangeMetersMin,
                maxRange                   = RangeMetersMax,
            };
        }

        return new LaserBasedRangeLidarConfiguration
        {
            laserArray = new LaserArray
            {
                centerOfMeasurementLinearOffsetMm = Vector3.zero,
                focalDistanceMm = 0f,
                lasers = lasers,
            },
            horizontalResolution = angularResolution_horizontal,
            minHAngle = -fov_horizontal / 2f,
            // HorizontalSteps rounds (max-min)/res, so the end angle is set from the beam COUNT
            // rather than from fov/2 — at a full 360 those differ by exactly the dropped beam.
            maxHAngle = -fov_horizontal / 2f + NumMeasurementsPerScan_h * angularResolution_horizontal,
            noiseParams = LidarNoiseParams.ZeroNoiseParams,
        };
    }

    /// <summary>Subscribe to RGL's per-capture callback (the publisher hangs its publish on it).</summary>
    public void AddOnNewData(LidarSensor.OnNewDataDelegate cb) { if (lidar != null) lidar.onNewData += cb; }
    public void RemoveOnNewData(LidarSensor.OnNewDataDelegate cb) { if (lidar != null) lidar.onNewData -= cb; }

    /// <summary>
    /// Repack the latest GPU scan into the CPU sensor's exact wire format. Call from the
    /// onNewData callback; returns null if RGL has not produced a usable scan yet.
    /// </summary>
    public PointCloud2Msg getScanMsg()
    {
        int count = outputGraph.GetResultDataRaw(ref rglData, RglPointStep);
        if (count <= 0) return null;

        if (count != numPoints)
        {
            // A mismatch means the ray set and the grid have drifted apart, which would make
            // parse_ordered_cloud reject every scan on the ROS side. Loud once, then clamp.
            if (!warnedCount)
            {
                Debug.LogError($"[GpuLaserSensor3D] RGL returned {count} points but the beam grid is " +
                               $"{NumMeasurementsPerScan_h}x{NumMeasurementsPerScan_v}={numPoints}. " +
                               "Occlusion detection on the ROS side needs these equal — check that " +
                               "nothing else reconfigured the LidarSensor.");
                warnedCount = true;
            }
            count = Mathf.Min(count, (int)numPoints);
        }

        Vector3 origin = laser_sensor_link.transform.position;

        for (int i = 0; i < count; i++)
        {
            int src = i * RglPointStep;
            int dst = i * OutPointStep;

            float wx = BitConverter.ToSingle(rglData, src);
            float wy = BitConverter.ToSingle(rglData, src + 4);
            float wz = BitConverter.ToSingle(rglData, src + 8);
            bool  hit = BitConverter.ToInt32(rglData, src + 12) != 0;

            if (hit)
            {
                // Unity -> ROS: x = dz, y = -dx, z = dy, about the sensor POSITION only.
                BitConverter.GetBytes(wz - origin.z).CopyTo(raw_data, dst);
                BitConverter.GetBytes(-(wx - origin.x)).CopyTo(raw_data, dst + 4);
                BitConverter.GetBytes(wy - origin.y).CopyTo(raw_data, dst + 8);
            }
            else if (publishMaxRangeOnNoHit)
            {
                // RGL leaves a miss at the ray origin, so the direction has to be rebuilt to
                // place a max-range point — same formula the CPU sensor rays are built from.
                int h = i / NumMeasurementsPerScan_v, v = i % NumMeasurementsPerScan_v;
                float theta = Mathf.Deg2Rad * (-fov_horizontal / 2f + h * angularResolution_horizontal);
                float psi   = Mathf.Deg2Rad * (-fov_vertical   / 2f + v * angularResolution_vertical);
                Vector3 d = laser_sensor_link.transform.rotation *
                            new Vector3(Mathf.Cos(psi) * Mathf.Sin(theta), -Mathf.Sin(psi),
                                        Mathf.Cos(psi) * Mathf.Cos(theta)) * RangeMetersMax;
                BitConverter.GetBytes(d.z).CopyTo(raw_data, dst);
                BitConverter.GetBytes(-d.x).CopyTo(raw_data, dst + 4);
                BitConverter.GetBytes(d.y).CopyTo(raw_data, dst + 8);
            }
            else
            {
                BitConverter.GetBytes(float.NaN).CopyTo(raw_data, dst);
                BitConverter.GetBytes(float.NaN).CopyTo(raw_data, dst + 4);
                BitConverter.GetBytes(float.NaN).CopyTo(raw_data, dst + 8);
            }
            BitConverter.GetBytes(0.0f).CopyTo(raw_data, dst + 12);   // intensity, as the CPU sensor
        }

        var stamp = new TimeStamp(Clock.time);
        return new PointCloud2Msg
        {
            header = new HeaderMsg
            {
                frame_id = FrameId,
                stamp = new TimeMsg(stamp.Seconds, stamp.NanoSeconds),
            },
            height = 1,
            width = numPoints,
            fields = new[]
            {
                new PointFieldMsg("x", 0, PointFieldMsg.FLOAT32, 1),
                new PointFieldMsg("y", 4, PointFieldMsg.FLOAT32, 1),
                new PointFieldMsg("z", 8, PointFieldMsg.FLOAT32, 1),
                // "i", not "intensity" — LaserSensor3D names it that way and the two clouds
                // must be indistinguishable downstream.
                new PointFieldMsg("i", 12, PointFieldMsg.FLOAT32, 1),
            },
            is_bigendian = false,
            point_step = OutPointStep,
            row_step = raw_data_len,
            data = (byte[])raw_data.Clone(),
            is_dense = false,
        };
    }

    public void Dispose()
    {
        outputGraph?.Clear();
    }
}
