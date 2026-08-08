#!/usr/bin/env python
"""Broadcast world -> camera_init from the drone's ACTUAL start pose.

FAST-LIO's origin is wherever it initialised, published as frame camera_init.
Placing its output in the world needs that start pose exactly; a wrong value is
a constant position error that looks like drift but never grows.

Hardcoding it in a launch file has already been wrong twice here, because
single_drone.xml's init_position params and the pose perfect_drone actually
starts at have disagreed. So take it from the running system instead:

  1. the /perfect_drone/init_position/{x,y,z} params, if present, else
  2. the first ground-truth odometry sample, which cannot disagree with reality.

(2) is the authority when both exist and differ, and the discrepancy is logged
rather than silently preferred one way or the other.
"""

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


class CameraInitTf(object):
    def __init__(self):
        self.parent = rospy.get_param("~parent_frame", "world")
        self.child = rospy.get_param("~child_frame", "camera_init")
        self.gt_topic = rospy.get_param("~gt_topic", "/lidar_slam/odom")

        self.param_xyz = None
        ns = rospy.get_param("~init_ns", "/perfect_drone/init_position")
        try:
            self.param_xyz = np.array([rospy.get_param(ns + "/x"),
                                       rospy.get_param(ns + "/y"),
                                       rospy.get_param(ns + "/z")], dtype=float)
            rospy.loginfo("[camera_init_tf] %s says start = %s",
                          ns, self.param_xyz.tolist())
        except KeyError:
            rospy.loginfo("[camera_init_tf] %s not on the param server", ns)

        self.br = tf2_ros.StaticTransformBroadcaster()
        self.sent = False
        self.sub = rospy.Subscriber(self.gt_topic, Odometry, self.cb_gt,
                                    queue_size=1)
        # If no ground truth ever arrives, fall back to the params so the TF
        # exists at all rather than leaving camera_init unconnected.
        rospy.Timer(rospy.Duration(3.0), self.fallback, oneshot=True)

    def send(self, xyz, source):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.parent
        t.child_frame_id = self.child
        t.transform.translation.x = float(xyz[0])
        t.transform.translation.y = float(xyz[1])
        t.transform.translation.z = float(xyz[2])
        # FAST-LIO gravity-aligns and starts at identity attitude, and MARSIM
        # launches with init_yaw 0, so this is a pure translation. A non-zero
        # start yaw would need the rotation here too.
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)
        self.sent = True
        rospy.loginfo("[camera_init_tf] %s -> %s at %s (from %s)",
                      self.parent, self.child, np.round(xyz, 3).tolist(), source)

    def cb_gt(self, msg):
        if self.sent:
            return
        p = msg.pose.pose.position
        xyz = np.array([p.x, p.y, p.z], dtype=float)
        if self.param_xyz is not None:
            d = float(np.linalg.norm(xyz - self.param_xyz))
            if d > 0.05:
                rospy.logwarn("[camera_init_tf] init_position param %s disagrees "
                              "with the actual start %s by %.2f m -- using the "
                              "actual pose", self.param_xyz.tolist(),
                              np.round(xyz, 3).tolist(), d)
        self.send(xyz, "first %s sample" % self.gt_topic)
        self.sub.unregister()

    def fallback(self, _evt):
        if self.sent:
            return
        if self.param_xyz is None:
            rospy.logerr("[camera_init_tf] no odom and no init_position param; "
                         "camera_init will not be connected to %s", self.parent)
            return
        rospy.logwarn("[camera_init_tf] no %s in 3 s, using the param value",
                      self.gt_topic)
        self.send(self.param_xyz, "param (odom never arrived)")


if __name__ == "__main__":
    rospy.init_node("camera_init_tf")
    CameraInitTf()
    rospy.spin()
