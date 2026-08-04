#!/usr/bin/env python
"""Occlusion-aware MPPI node. Closes the loop inside MARSIM -- no Unity, no ROS 2.

    MARSIM lidar ─► /cloud_registered ─┐
                                       ├─► ROG-Map (rm_node) ─► /rm_node/occlusion_frontier ─┐
    perfect_drone ─► /lidar_slam/odom ─┘                                                     │
           ▲                                                                                 │
           └──────────────── /planning/pos_cmd ◄──────────── THIS NODE ◄────────────────────┘

This replaces keyboard_control.py: same topic, same message, same integrate-a-
setpoint pattern (perfect_drone consumes position/velocity through differential
flatness), except the velocity comes from MPPI instead of the W/A/S/D keys.
"""

import os
import sys

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from quadrotor_msgs.msg import PositionCommand

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from occlusion_mppi.boundary import BoundarySet          # noqa: E402
from occlusion_mppi.dynamics import DoubleIntegrator     # noqa: E402
from occlusion_mppi.mppi import OcclusionMPPI, MPPIConfig  # noqa: E402


class MPPINode(object):
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.goal = np.array(rospy.get_param("~goal", [0.0, 12.0]), dtype=float)
        self.cruise_z = rospy.get_param("~cruise_z", 1.0)
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.goal_tol = rospy.get_param("~goal_tol", 0.5)
        # Only boundaries within this vertical band of the drone matter while
        # flying at fixed altitude; a boundary 4 m overhead cannot be hit
        # laterally and would inflate the keep-out for nothing.
        self.z_band = rospy.get_param("~z_band", 1.5)

        cfg = MPPIConfig(
            horizon=int(rospy.get_param("~horizon", 30)),
            samples=int(rospy.get_param("~samples", 1000)),
            dt=float(rospy.get_param("~dt", 0.1)),
            d_safe=float(rospy.get_param("~d_safe", 1.0)),
            v_target=float(rospy.get_param("~v_target", 1.5)),
            t_grow_max=float(rospy.get_param("~t_grow_max", 3.0)),
            w_soft=float(rospy.get_param("~w_soft", 50.0)),
            d_infl=float(rospy.get_param("~d_infl", 1.0)),
        )
        self.cfg = cfg

        plant = DoubleIntegrator(dim=2, dt=cfg.dt,
                                 v_max=float(rospy.get_param("~v_max", 2.0)),
                                 a_max=float(rospy.get_param("~a_max", 3.0)))
        self.plant = plant
        self.planner = OcclusionMPPI(plant, cfg, rng=np.random.default_rng(0))

        self.boundaries = BoundarySet(np.zeros((0, 3)))
        self.pos = None
        self.vel = np.zeros(2)
        # The setpoint is integrated, exactly like keyboard_control.py, rather than
        # being snapped to the measured pose each tick -- feeding the measured pose
        # back into the setpoint would close a second loop through the plant and
        # let tracking error accumulate into the command.
        self.sp = None

        self.pub_cmd = rospy.Publisher("/planning/pos_cmd", PositionCommand, queue_size=10)
        self.pub_viz = rospy.Publisher("~rollouts", MarkerArray, queue_size=1)

        rospy.Subscriber("/lidar_slam/odom", Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber(rospy.get_param("~boundary_topic",
                                         "/rm_node/occlusion_frontier"),
                         PointCloud2, self.cb_boundary, queue_size=1)

        rospy.loginfo("[mppi] goal=%s d_safe=%.1f v_target=%.1f t_grow_max=%.1f "
                      "=> keep-out grows %.1f -> %.1f m",
                      self.goal.tolist(), cfg.d_safe, cfg.v_target, cfg.t_grow_max,
                      cfg.d_safe, cfg.d_safe + cfg.v_target * cfg.t_grow_max)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.step)

    def cb_odom(self, msg):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.pos = np.array([p.x, p.y])
        self.vel = np.array([v.x, v.y])
        if self.sp is None:
            self.sp = np.array([p.x, p.y, self.cruise_z])

    def cb_boundary(self, msg):
        pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                            skip_nans=True)), dtype=float)
        ego_z = self.sp[2] if self.sp is not None else self.cruise_z
        self.boundaries = BoundarySet(pts.reshape(-1, 3), z_band=self.z_band,
                                      ego_z=ego_z, planar=True)

    def step(self, _evt):
        if self.pos is None or self.sp is None:
            return

        if np.linalg.norm(self.pos - self.goal) < self.goal_tol:
            self.publish_cmd(np.zeros(2))
            rospy.loginfo_throttle(5.0, "[mppi] goal reached")
            return

        action, info = self.planner.plan(self.pos, self.vel, self.goal, self.boundaries)

        # Integrate the setpoint with the planned acceleration.
        self.vel_cmd = np.clip(self.vel + action * self.cfg.dt,
                               -self.plant.v_max, self.plant.v_max)
        self.publish_cmd(self.vel_cmd)

        d_now = float(self.boundaries.distance(self.pos[None, :])[0])
        rospy.loginfo_throttle(
            2.0, "[mppi] pos=(%.1f,%.1f) |v|=%.2f  d_occ=%s  boundaries=%d  infeas=%.2f",
            self.pos[0], self.pos[1], float(np.linalg.norm(self.vel)),
            ("inf" if not np.isfinite(d_now) else "%.2f" % d_now),
            len(self.boundaries), info["frac_infeasible"])

        self.publish_rollouts(info)

    def publish_cmd(self, vel_cmd):
        dt = 1.0 / self.rate_hz
        self.sp[0] += vel_cmd[0] * dt
        self.sp[1] += vel_cmd[1] * dt
        self.sp[2] = self.cruise_z

        m = PositionCommand()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.frame_id
        m.position.x, m.position.y, m.position.z = self.sp
        m.velocity.x, m.velocity.y, m.velocity.z = vel_cmd[0], vel_cmd[1], 0.0
        self.pub_cmd.publish(m)

    def publish_rollouts(self, info, n_show=40):
        arr = MarkerArray()
        traj = info["traj"]
        idx = np.argsort(info["cost"])[:n_show]
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns, m.id, m.type, m.action = "rollouts", 0, Marker.LINE_LIST, Marker.ADD
        m.scale.x = 0.02
        m.color.a, m.color.r, m.color.g, m.color.b = 0.35, 0.2, 0.6, 1.0
        m.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point
        for i in idx:
            for k in range(traj.shape[1] - 1):
                a, b = traj[i, k, :2], traj[i, k + 1, :2]
                m.points.append(Point(a[0], a[1], self.cruise_z))
                m.points.append(Point(b[0], b[1], self.cruise_z))
        arr.markers.append(m)

        best = Marker()
        best.header = m.header
        best.ns, best.id, best.type, best.action = "best", 1, Marker.LINE_STRIP, Marker.ADD
        best.scale.x = 0.08
        best.color.a, best.color.r, best.color.g, best.color.b = 1.0, 0.1, 1.0, 0.2
        best.pose.orientation.w = 1.0
        for k in range(info["best"].shape[0]):
            p = info["best"][k, :2]
            best.points.append(Point(p[0], p[1], self.cruise_z))
        arr.markers.append(best)

        self.pub_viz.publish(arr)


if __name__ == "__main__":
    rospy.init_node("occlusion_mppi")
    MPPINode()
    rospy.spin()
