#!/usr/bin/env python
"""Locate a FAST-LIO drift by measuring the three things that cause it.

Run alongside fastlio.launch. Everything here is read-only.

1. POSE ERROR vs GROUND TRUTH
   perfect_drone's odom is truth. FAST-LIO reports in camera_init, whose origin
   is the drone's world start pose, so the comparison needs that offset added
   back. A CONSTANT error means the offset is wrong; a GROWING one is real drift.

2. TIMESTAMP ALIGNMENT
   A lidar-inertial filter integrates IMU between scans, so it needs IMU samples
   that bracket each scan. If the two streams are stamped from different clocks,
   or one lags, propagation happens over the wrong interval and the pose walks
   even when both signals are individually correct. This is invisible in RViz.

3. RATES
   FAST-LIO wants IMU well above lidar rate. Starved IMU means long unconstrained
   propagation gaps.
"""

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2


class FastlioDebug(object):
    def __init__(self):
        # Learned from the first ground-truth sample: camera_init's origin IS
        # the drone's start pose. Hardcoding it made this tool report a constant
        # 3 m error that was its own bug, not FAST-LIO's.
        p0 = rospy.get_param("~offset", None)
        self.offset = np.array(p0, dtype=float) if p0 else None
        self.period = float(rospy.get_param("~period", 2.0))

        self.gt = None
        self.est = None
        self.t_imu = []
        self.t_lidar = []
        self.acc = []
        self.gyr = []
        self.err0 = None
        self.last = rospy.Time.now().to_sec()

        rospy.Subscriber(rospy.get_param("~gt_topic", "/lidar_slam/odom"),
                         Odometry, self.cb_gt, queue_size=50)
        rospy.Subscriber(rospy.get_param("~est_topic", "/fastlio/odom"),
                         Odometry, self.cb_est, queue_size=50)
        rospy.Subscriber(rospy.get_param("~imu_topic", "/quad_0/imu"),
                         Imu, self.cb_imu, queue_size=200)
        rospy.Subscriber(rospy.get_param("~lidar_topic",
                                         "/quad0_pcl_render_node/sensor_cloud"),
                         PointCloud2, self.cb_lidar, queue_size=10)
        rospy.Timer(rospy.Duration(self.period), self.report)
        rospy.loginfo("[fastlio_debug] waiting for ground truth to learn the "
                      "camera_init origin")

    def cb_gt(self, msg):
        p = msg.pose.pose.position
        xyz = np.array([p.x, p.y, p.z])
        if self.offset is None:
            self.offset = xyz.copy()
            rospy.loginfo("[fastlio_debug] camera_init origin learned from the "
                          "first GT sample: %s", np.round(xyz, 3).tolist())
        self.gt = (msg.header.stamp.to_sec(), xyz)

    def cb_est(self, msg):
        p = msg.pose.pose.position
        self.est = (msg.header.stamp.to_sec(), np.array([p.x, p.y, p.z]))

    def cb_imu(self, msg):
        self.t_imu.append(msg.header.stamp.to_sec())
        a = msg.linear_acceleration
        w = msg.angular_velocity
        self.acc.append([a.x, a.y, a.z])
        self.gyr.append([w.x, w.y, w.z])

    def cb_lidar(self, msg):
        self.t_lidar.append(msg.header.stamp.to_sec())

    def report(self, _evt):
        now = rospy.Time.now().to_sec()
        dt = now - self.last
        self.last = now

        if self.gt is None or self.est is None or self.offset is None:
            rospy.logwarn("[fastlio_debug] waiting: gt=%s est=%s",
                          self.gt is not None, self.est is not None)
            self.t_imu, self.t_lidar, self.acc, self.gyr = [], [], [], []
            return

        # FAST-LIO reports in camera_init; lift into world before comparing.
        est_world = self.est[1] + self.offset
        err = est_world - self.gt[1]
        if self.err0 is None:
            self.err0 = err.copy()
        # Growth since the first sample separates real drift from a bad offset.
        growth = np.linalg.norm(err - self.err0)

        f_imu = len(self.t_imu) / max(dt, 1e-6)
        f_lid = len(self.t_lidar) / max(dt, 1e-6)
        # Newest IMU stamp minus newest lidar stamp: must be >= 0, or every scan
        # is being processed with IMU that predates it.
        skew = ((self.t_imu[-1] - self.t_lidar[-1])
                if (self.t_imu and self.t_lidar) else float("nan"))

        rospy.loginfo(
            "[fastlio_debug]\n"
            "    gt   =[%+.2f %+.2f %+.2f]   est(world)=[%+.2f %+.2f %+.2f]\n"
            "    error=[%+.2f %+.2f %+.2f]  |%.2f| m   growth-since-start %.2f m\n"
            "    rates: imu %.0f Hz  lidar %.1f Hz   imu-minus-lidar stamp %+.3f s\n"
            "    est(camera_init raw)=[%+.2f %+.2f %+.2f]  <-- what ROG-Map's\n"
            "        virtual_ground/ceil thresholds are compared against\n"
            "    stamp gt-vs-est %+.3f s",
            self.gt[1][0], self.gt[1][1], self.gt[1][2],
            est_world[0], est_world[1], est_world[2],
            err[0], err[1], err[2], float(np.linalg.norm(err)), growth,
            f_imu, f_lid, skew,
            self.est[1][0], self.est[1][1], self.est[1][2],
            self.gt[0] - self.est[0])

        # What the IMU is actually emitting. |acc| must sit at ~9.81 and the z
        # component tells us the gravity convention: upstream MARSIM's
        # quadrotor_dynamics_node emits R^T(a + [0,0,-9.8]), i.e. NEGATIVE at
        # rest, which is what FAST-LIO's mean_acc=(0,0,-1) init expects. A |acc|
        # far off 9.81, or a z of the wrong sign, means gravity is being
        # double-counted and the propagation integrates a ~g bias.
        if self.acc:
            am = np.mean(np.array(self.acc), axis=0)
            gm = np.mean(np.array(self.gyr), axis=0)
            rospy.loginfo(
                "    IMU: acc_mean=[%+.2f %+.2f %+.2f] |%.2f| (expect 9.81, "
                "z NEGATIVE for MARSIM)  gyr_mean=[%+.3f %+.3f %+.3f]",
                am[0], am[1], am[2], float(np.linalg.norm(am)),
                gm[0], gm[1], gm[2])
            if abs(np.linalg.norm(am) - 9.81) > 2.0:
                rospy.logwarn("[fastlio_debug] |acc|=%.2f is far from g: gravity "
                              "is not being represented correctly",
                              float(np.linalg.norm(am)))

        if f_lid > 0 and f_imu < 5 * f_lid:
            rospy.logwarn("[fastlio_debug] IMU only %.0fx lidar rate; long "
                          "unconstrained propagation gaps", f_imu / max(f_lid, 1e-6))
        if not np.isnan(skew) and skew < 0:
            rospy.logwarn("[fastlio_debug] IMU stamps BEHIND lidar by %.3f s -- "
                          "scans processed without bracketing IMU", -skew)
        self.t_imu, self.t_lidar, self.acc, self.gyr = [], [], [], []


if __name__ == "__main__":
    rospy.init_node("fastlio_debug")
    FastlioDebug()
    rospy.spin()
