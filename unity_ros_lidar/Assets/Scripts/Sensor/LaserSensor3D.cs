using System;
using System.Collections.Generic;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;
using Unity.Robotics.Core;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine.Serialization;

public class LaserSensor3D
{
    float RangeMetersMin;
    float RangeMetersMax;

    float fov_horizontal;
    float fov_vertical;

    float angularResolution_vertical;
    float angularResolution_horizontal;

    int NumMeasurementsPerScan_h;
    int NumMeasurementsPerScan_v;

    float[] scanAngleArray_h;
    float[] scanAngleArray_v;
    
    string FrameId;
    bool publishMaxRangeOnNoHit;

    uint numPoints;
    uint raw_data_len;
    byte[] raw_data;

    GameObject laser_sensor_link;

    // Scene-view ray visualization (see PointCloudPublisher inspector fields).
    public bool drawDebugRays = false;
    public int debugRayStride = 8;
    public float debugRayDuration = 0.1f;

    // Hit-point markers: drawn for every hit, independent of debugRayStride,
    // so the detected boundary outline stays complete.
    public bool drawHitPoints = false;
    public float hitPointSize = 0.15f;

    // Occlusion edges via range discontinuity between adjacent azimuth beams — the
    // same test as controller/occlusion_jump_detect.py, run here so the blind-corner
    // mouths can be drawn in the Scene view without a ROS round trip.
    public bool drawOcclusionEdges = false;
    public float grazingDeg = 12f;
    public float minJumpMeters = 0.5f;

    // Set by consumers that need OcclusionCorners without the debug lines (the phantom
    // visualizer). The edge pass runs when either this or drawOcclusionEdges is set.
    public bool computeOcclusionEdges = false;
    // drawOcclusionFov is included so the wedge can be inspected on its own, without
    // also turning on the edge lines.
    bool EdgePassActive => drawOcclusionEdges || computeOcclusionEdges || drawOcclusionFov;

    // Corners closer together than this collapse to one — the test fires once per
    // elevation row, so a single vertical edge would otherwise seed a stack of phantoms.
    public float cornerMergeRadius = 1.5f;

    // Forward wedge — mirror of OCC_FWD_HALF_ANGLE in taxi_controller_mpc.py. A phantom
    // emerging from a corner the ego has already driven past cannot be run into, so
    // corners outside the wedge are discarded rather than seeding capsules. Keep this
    // equal to the controller's value or the Scene view will disagree with what the MPC
    // actually constrains.
    // Corners nearer than this are discarded. Beams that clip the airframe or the ground
    // directly under the sensor produce a huge range jump at ~0 m; the resulting corner
    // sits ON the sensor, where its bearing is meaningless, so it passes the forward-wedge
    // test at ANY half-angle and then draws a segment pointing wherever the escaping beam
    // went — including straight behind. Mirrors LidarCostmap's min_range (1.0 m).
    public float minCornerRange = 1.0f;

    public bool filterCornersToFov = true;
    public float occFwdHalfAngleDeg = 100f;
    public bool drawOcclusionFov = false;
    public float fovDrawRadius = 60f;      // matches OCC_QUERY_R

    // Per-beam ranges and ray endpoints, retained across the scan so the edge pass
    // can compare neighbours after all raycasts are done. Indexed [i * NumV + j].
    float[] beamRange;
    Vector3[] beamEnd;

    // Corner points from the last scan's edge pass — the blind-corner mouths a hidden
    // agent is assumed to emerge from. Read by PhantomAgentVisualizer.
    public List<Vector3> OcclusionCorners { get; private set; } = new List<Vector3>();

    // float avg_time;
    // int total_counts;


    public LaserSensor3D(
        GameObject _laser_sensor_link,
        float _RangeMetersMin,
        float _RangeMetersMax,
        float _fov_horizontal,
        float _fov_vertical,
        float _angularResolution_vertical,
        float _angularResolution_horizontal,
        bool _publishMaxRangeOnNoHit = false,
        string _frameIdOverride = ""
    )
    {
        RangeMetersMin = _RangeMetersMin;
        RangeMetersMax = _RangeMetersMax;
        fov_horizontal = _fov_horizontal<=360?_fov_horizontal:360;
        fov_vertical = _fov_vertical<=360?_fov_vertical:360;;
        angularResolution_vertical = Mathf.Max(0.001f, _angularResolution_vertical);
        angularResolution_horizontal = Mathf.Max(0.001f, _angularResolution_horizontal);
        publishMaxRangeOnNoHit = _publishMaxRangeOnNoHit;

        laser_sensor_link = _laser_sensor_link;

        var fallbackFrame = laser_sensor_link.name.Replace(" ", "_");
        FrameId = string.IsNullOrWhiteSpace(_frameIdOverride) ? fallbackFrame : _frameIdOverride;

        float ScanAngleStart_h = -fov_horizontal/2;
        float ScanAngleEnd_h = fov_horizontal/2;
        float ScanAngleStart_v = -fov_vertical/2;
        float ScanAngleEnd_v = fov_vertical/2;

        NumMeasurementsPerScan_h = Mathf.FloorToInt((ScanAngleEnd_h - ScanAngleStart_h) / angularResolution_horizontal) + 1;
        NumMeasurementsPerScan_v = Mathf.FloorToInt((ScanAngleEnd_v - ScanAngleStart_v) / angularResolution_horizontal) + 1;

        if (fov_horizontal==360)
        {
            NumMeasurementsPerScan_h = NumMeasurementsPerScan_h -1;
        }
        if (fov_vertical==360)
        {
            NumMeasurementsPerScan_v = NumMeasurementsPerScan_v -1;
            
        }
        
        scanAngleArray_h = new float[NumMeasurementsPerScan_h];
        for (int i = 0; i < NumMeasurementsPerScan_h; i++)
        {
            scanAngleArray_h[i] = ScanAngleStart_h + i * angularResolution_horizontal;
        }

        NumMeasurementsPerScan_v = Mathf.FloorToInt((ScanAngleEnd_v - ScanAngleStart_v) / angularResolution_vertical) + 1;
        scanAngleArray_v = new float[NumMeasurementsPerScan_v];
        for (int i = 0; i < NumMeasurementsPerScan_v; i++)
        {
            scanAngleArray_v[i] = ScanAngleStart_v + i * angularResolution_vertical;
        }

        numPoints = (uint)(NumMeasurementsPerScan_h*NumMeasurementsPerScan_v);

        raw_data_len = 16*numPoints;
        raw_data = new byte[raw_data_len];

        beamRange = new float[numPoints];
        beamEnd = new Vector3[numPoints];

        // avg_time = 0;
        // total_counts = 0;
    }

    public PointCloud2Msg getScanMsg()
    {

        float startTime = Time.realtimeSinceStartup;

        Transform sensor_transform = laser_sensor_link.transform;

        int raw_data_indx = 0;
        for (int i = 0; i < NumMeasurementsPerScan_h; i++)
        {
            for (int j = 0; j < NumMeasurementsPerScan_v; j++)
            {
                // azimuth
                var theta = Mathf.Deg2Rad*scanAngleArray_h[i];
                // elevation
                var psi = Mathf.Deg2Rad*scanAngleArray_v[j];
                var local_dir_vec = new Vector3(Mathf.Cos(psi)*Mathf.Sin(theta), -Mathf.Sin(psi), Mathf.Cos(psi)*Mathf.Cos(theta));
                var directionVector = sensor_transform.rotation * local_dir_vec;

                var measurementStart = RangeMetersMin * directionVector + sensor_transform.position;
                var measurementRay = new Ray(measurementStart, directionVector);
                var foundValidMeasurement = Physics.Raycast(measurementRay, out var hit, RangeMetersMax);

                if (EdgePassActive)
                {
                    // A non-return is a beam that reached max range without hitting anything,
                    // which is a real edge when its neighbour hit something — record it as
                    // max range rather than discarding it.
                    beamRange[raw_data_indx] = foundValidMeasurement ? hit.distance : RangeMetersMax;
                    beamEnd[raw_data_indx] = foundValidMeasurement
                        ? hit.point
                        : measurementStart + directionVector * RangeMetersMax;
                }

                if (drawDebugRays && i % debugRayStride == 0 && j % debugRayStride == 0)
                {
                    // green = hit a collider, red = no return within max range
                    var rayEnd = foundValidMeasurement ? hit.point : measurementStart + directionVector * RangeMetersMax;
                    Debug.DrawLine(measurementStart, rayEnd,
                        foundValidMeasurement ? Color.green : new Color(1f, 0f, 0f, 0.25f),
                        debugRayDuration);
                }

                if (drawHitPoints && foundValidMeasurement)
                {
                    // Colour by range so near/far structure is readable: cyan (near) -> magenta (far).
                    var t = RangeMetersMax > 0f ? Mathf.Clamp01(hit.distance / RangeMetersMax) : 0f;
                    var c = Color.Lerp(Color.cyan, Color.magenta, t);

                    // Cross laid in the surface plane so it reads as a patch of the boundary,
                    // not a floating dot.
                    var n = hit.normal;
                    var u = Vector3.Normalize(Vector3.Cross(n, Mathf.Abs(n.y) < 0.9f ? Vector3.up : Vector3.right));
                    var v = Vector3.Cross(n, u);
                    var h = hitPointSize * 0.5f;
                    Debug.DrawLine(hit.point - u * h, hit.point + u * h, c, debugRayDuration);
                    Debug.DrawLine(hit.point - v * h, hit.point + v * h, c, debugRayDuration);
                }

                // Only record measurement if it's within the sensor's operating range
                if (foundValidMeasurement)
                {
                    BitConverter.GetBytes(hit.point.z-sensor_transform.position.z).CopyTo(raw_data, raw_data_indx * 16);
                    BitConverter.GetBytes(-(hit.point.x-sensor_transform.position.x)).CopyTo(raw_data, raw_data_indx * 16+4);
                    BitConverter.GetBytes(hit.point.y-sensor_transform.position.y).CopyTo(raw_data, raw_data_indx * 16+8);
                    BitConverter.GetBytes(0.0f).CopyTo(raw_data, raw_data_indx * 16 + 12);
                }
                else
                {
                    if (publishMaxRangeOnNoHit)
                    {
                        var maxPoint = directionVector * RangeMetersMax;
                        BitConverter.GetBytes(maxPoint.z).CopyTo(raw_data, raw_data_indx * 16);
                        BitConverter.GetBytes(-maxPoint.x).CopyTo(raw_data, raw_data_indx * 16 + 4);
                        BitConverter.GetBytes(maxPoint.y).CopyTo(raw_data, raw_data_indx * 16 + 8);
                    }
                    else
                    {
                        BitConverter.GetBytes(float.NaN).CopyTo(raw_data, raw_data_indx * 16);
                        BitConverter.GetBytes(float.NaN).CopyTo(raw_data, raw_data_indx * 16 + 4);
                        BitConverter.GetBytes(float.NaN).CopyTo(raw_data, raw_data_indx * 16 + 8);
                    }
                    BitConverter.GetBytes(0.0f).CopyTo(raw_data, raw_data_indx * 16 + 12);
                }
                ++raw_data_indx;
            }
        }

        if (EdgePassActive)
        {
            DrawOcclusionEdges(sensor_transform);
        }


        var timestamp = new TimeStamp(Clock.time);
        // (edge pass ran above, before the message is packed)
        
        var msg = new PointCloud2Msg
        {
            header = new HeaderMsg
            {
                frame_id = FrameId,
                stamp = new TimeMsg
                {
                    sec = timestamp.Seconds,
                    nanosec = timestamp.NanoSeconds,
                }
            },
            height = 1,
            width = numPoints,
            fields = new PointFieldMsg[]
            {
                new PointFieldMsg("x", 0, PointFieldMsg.FLOAT32, 1),
                new PointFieldMsg("y", 4, PointFieldMsg.FLOAT32, 1),
                new PointFieldMsg("z", 8, PointFieldMsg.FLOAT32, 1),
                new PointFieldMsg("i", 12, PointFieldMsg.FLOAT32, 1)
            },
            is_bigendian = false,
            point_step = 16,
            row_step = raw_data_len,
            data = (byte[])raw_data.Clone(),
            is_dense = false,
        };

        // float endTime = Time.realtimeSinceStartup;
        // float elapsedTime = 1000*(endTime - startTime);

        // float total_time = (total_counts*avg_time + elapsedTime);
        // total_counts = total_counts + 1;
        // avg_time = total_time/total_counts;

        // Debug.Log("getScanMsg() Execution Time (curr, avg): (" + elapsedTime.ToString("F4") +", "+ avg_time.ToString("F4")+ ") ms");

        return msg;
    }

    /// <summary>
    /// Flags range discontinuities between adjacent azimuth beams and draws the blind-corner
    /// mouths. Mirrors controller/occlusion_jump_detect.py: a continuous surface at range r
    /// spans r*dtheta/tan(grazing) between neighbouring beams, so a larger step is a genuine
    /// depth jump rather than a receding surface. Scaling the threshold with range is what
    /// keeps it valid at distance, where the same physical gap subtends fewer beams.
    /// </summary>
    void DrawOcclusionEdges(Transform sensor_transform)
    {
        OcclusionCorners.Clear();
        float mergeSq = cornerMergeRadius * cornerMergeRadius;

        float dtheta = Mathf.Deg2Rad * angularResolution_horizontal;
        float tanGrazing = Mathf.Tan(Mathf.Deg2Rad * Mathf.Max(1f, grazingDeg));

        // A 360-degree scan wraps, so the last azimuth column neighbours the first.
        bool wraps = fov_horizontal >= 360f;
        int lastPair = wraps ? NumMeasurementsPerScan_h : NumMeasurementsPerScan_h - 1;

        // Forward wedge, flattened to the ground plane — the controller's test is 2-D
        // in (a0, a1), so any sensor pitch must not tilt the wedge.
        var fwd = sensor_transform.forward;
        fwd.y = 0f;
        fwd = fwd.sqrMagnitude > 1e-9f ? fwd.normalized : Vector3.forward;
        float cosLim = Mathf.Cos(Mathf.Deg2Rad * Mathf.Clamp(occFwdHalfAngleDeg, 0f, 180f));

        if (drawOcclusionFov)
            DrawFovWedge(sensor_transform.position, fwd, cosLim);

        for (int i = 0; i < lastPair; i++)
        {
            int iNext = (i + 1) % NumMeasurementsPerScan_h;
            for (int j = 0; j < NumMeasurementsPerScan_v; j++)
            {
                float rA = beamRange[i * NumMeasurementsPerScan_v + j];
                float rB = beamRange[iNext * NumMeasurementsPerScan_v + j];

                float near = Mathf.Min(rA, rB);
                float delta = Mathf.Abs(rA - rB);
                float thresh = Mathf.Max(near * dtheta / tanGrazing, minJumpMeters);
                if (delta <= thresh)
                    continue;

                // Self-hits / ground clutter right under the sensor: the corner would sit at
                // the origin, where the forward-wedge test is degenerate.
                if (near < minCornerRange)
                    continue;

                var pA = beamEnd[i * NumMeasurementsPerScan_v + j];
                var pB = beamEnd[iNext * NumMeasurementsPerScan_v + j];
                // The nearer endpoint is the corner the hidden agent would round; the farther
                // is where the shadow's far wall begins.
                var corner = (rA < rB) ? pA : pB;
                var far = (rA < rB) ? pB : pA;

                // Is this corner inside the forward wedge the controller constrains?
                var toCorner = corner - sensor_transform.position;
                toCorner.y = 0f;
                float m = toCorner.magnitude;
                bool inFov = m < 1e-6f || Vector3.Dot(toCorner / m, fwd) >= cosLim;

                if (drawOcclusionEdges)
                {
                    // In-wedge corners are drawn live (yellow mouth, red corner); corners
                    // the controller discards are drawn dim grey so it is obvious they were
                    // detected but deliberately not constrained.
                    if (inFov)
                    {
                        Debug.DrawLine(corner, far, Color.yellow, debugRayDuration);
                        Debug.DrawLine(corner - Vector3.up * hitPointSize,
                                       corner + Vector3.up * hitPointSize,
                                       Color.red, debugRayDuration);
                    }
                    else
                    {
                        Debug.DrawLine(corner, far, new Color(0.45f, 0.45f, 0.45f, 0.5f),
                                       debugRayDuration);
                    }
                }

                if (filterCornersToFov && !inFov)
                    continue;

                // Merge against corners already recorded this scan, comparing on the
                // ground plane so the stack from one vertical edge collapses to a point.
                bool merged = false;
                for (int c = 0; c < OcclusionCorners.Count; c++)
                {
                    var d = OcclusionCorners[c] - corner;
                    d.y = 0f;
                    if (d.sqrMagnitude < mergeSq) { merged = true; break; }
                }
                if (!merged)
                    OcclusionCorners.Add(corner);
            }
        }
    }

    /// <summary>
    /// Outlines the forward wedge (+/- occFwdHalfAngleDeg about the ego heading) within which
    /// a detected occlusion boundary is allowed to seed a keep-out capsule. Cyan edges plus
    /// an arc at fovDrawRadius (= OCC_QUERY_R), so the drawn region is exactly the
    /// angle-and-range gate the MPC applies.
    /// </summary>
    void DrawFovWedge(Vector3 origin, Vector3 fwd, float cosLim)
    {
        float half = Mathf.Acos(Mathf.Clamp(cosLim, -1f, 1f));   // [rad]
        var col = new Color(0f, 0.9f, 1f, 0.9f);
        var up = Vector3.up;

        // Sensor height is unknown here, so draw in the sensor's own horizontal plane.
        Vector3 Dir(float ang)
        {
            var q = Quaternion.AngleAxis(ang * Mathf.Rad2Deg, up);
            return q * fwd;
        }

        var left = origin + Dir(half) * fovDrawRadius;
        var right = origin + Dir(-half) * fovDrawRadius;
        Debug.DrawLine(origin, left, col, debugRayDuration, false);
        Debug.DrawLine(origin, right, col, debugRayDuration, false);

        // Arc closing the wedge at the query radius.
        const int ARC = 32;
        var prev = right;
        for (int a = 1; a <= ARC; a++)
        {
            float ang = -half + 2f * half * a / ARC;
            var p = origin + Dir(ang) * fovDrawRadius;
            Debug.DrawLine(prev, p, col, debugRayDuration, false);
            prev = p;
        }

        // Heading tick, so the wedge's orientation is unambiguous when it exceeds 90 deg.
        Debug.DrawLine(origin, origin + fwd * (fovDrawRadius * 0.25f),
                       new Color(0f, 1f, 0.6f, 1f), debugRayDuration, false);
    }
}
