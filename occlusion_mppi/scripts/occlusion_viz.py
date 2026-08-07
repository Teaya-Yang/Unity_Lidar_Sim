#!/usr/bin/env python
"""Occlusion keep-out visualisation, decoupled from the planner.

Draws the same keep-out bubble, sightlines and status readout as mppi_node.py, but
takes the drone pose straight from odometry instead of from a plan. That makes the
occlusion geometry inspectable while flying under keyboard_control.py -- the
question "is the frontier sane here?" is separate from "is the planner sane?", and
answering it should not require running the planner.

    roslaunch occlusion_boundary wall_test.launch
    roslaunch occlusion_boundary occlusion_boundary_walltest.launch
    rosrun occlusion_mppi occlusion_viz.py
    rosrun rog_map_example keyboard_control.py
"""

import os
import sys

import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from occlusion_mppi.boundary import BoundarySet, OccupancySet  # noqa: E402
from occlusion_mppi.viz import (nearest_markers, ego_occupied_markers,  # noqa: E402
                                nearest_occupied_markers)  # noqa: E402


class OcclusionViz(object):
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.rate_hz = rospy.get_param("~rate", 5.0)

        # Kept identical to mppi_node.py's defaults so the picture drawn here is the
        # one the planner would act on. Override both together or not at all.
        self.d_safe = float(rospy.get_param("~d_safe", 1.0))
        self.v_target = float(rospy.get_param("~v_target", 1.5))
        self.t_grow_max = float(rospy.get_param("~t_grow_max", 3.0))
        self.z_band = float(rospy.get_param("~z_band", 1.5))
        # Mirrors mppi_node: 3 => 3D geometry, no z folding, no z_band.
        self.dim = int(rospy.get_param("~dim", 3))
        self.planar = self.dim == 2
        self.occ_res = float(rospy.get_param("~occ_resolution", 0.2))
        self.occ_z_band = float(rospy.get_param("~occ_z_band", 0.3))
        self.sightline_max = int(rospy.get_param("~sightline_max", 120))
        self.keepout_times = rospy.get_param(
            "~keepout_times", [0.0, 0.5 * self.t_grow_max, self.t_grow_max])

        # Set before the placeholders below: query_z reads self.z.
        self.pos = None
        self.z = 1.0

        # Must match the dimension the callbacks build at -- see mppi_node.
        self.boundaries = BoundarySet(np.zeros((0, 3)), planar=self.planar,
                                      query_z=self.z)
        self.occupancy = OccupancySet(np.zeros((0, 3)), resolution=self.occ_res,
                                      planar=self.planar)

        self.pub_keepout = rospy.Publisher("~keepout", MarkerArray, queue_size=1)
        self.pub_sight = rospy.Publisher("~sightlines", MarkerArray, queue_size=1)
        self.pub_status = rospy.Publisher("~status", Marker, queue_size=1)
        self.show_nearest = bool(rospy.get_param("~show_nearest", True))
        self.show_occupied = bool(rospy.get_param("~show_occupied", True))
        # Builds a KD-tree over inf_occ (tens of thousands of points) whenever the
        # cloud changes. Nothing in the planner needs it -- turn it off if the
        # solve rate suffers.
        self.show_nearest_occ = bool(rospy.get_param("~show_nearest_occupied", True))
        self.pub_nearest_occ = rospy.Publisher("~nearest_occupied", MarkerArray,
                                               queue_size=1)
        self.pub_occupied = rospy.Publisher("~ego_occupied", MarkerArray, queue_size=1)
        self.pub_nearest = rospy.Publisher("~nearest_boundary", MarkerArray,
                                           queue_size=1)

        rospy.Subscriber("/lidar_slam/odom", Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber(rospy.get_param("~boundary_topic",
                                         "/rm_node/occlusion_frontier"),
                         PointCloud2, self.cb_boundary, queue_size=1)
        rospy.Subscriber(rospy.get_param("~occupancy_topic",
                                         "/rm_node/rog_map/inf_occ"),
                         PointCloud2, self.cb_occupancy, queue_size=1)

        rospy.loginfo("[occ_viz] keep-out grows %.1f -> %.1f m over %.1f s",
                      self.d_safe, self.d_safe + self.v_target * self.t_grow_max,
                      self.t_grow_max)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.step)

    def ego(self):
        """Ego pose at the dimension the sets were built at.

        self.pos is kept 2D because most of the drawing works in xy, but the
        occupancy/boundary sets are 3D when dim=3 and a 2D query against them is
        not a smaller query -- it is a different key space that matches nothing.
        """
        if self.dim == 3:
            return np.array([self.pos[0], self.pos[1], self.z])
        return self.pos

    def cb_odom(self, msg):
        p = msg.pose.pose.position
        self.pos = np.array([p.x, p.y])
        self.z = p.z

    def cb_boundary(self, msg):
        pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                            skip_nans=True)), dtype=float)
        self.boundaries = BoundarySet(
            pts.reshape(-1, 3),
            z_band=self.z_band if self.dim == 2 else None,
            ego_z=self.z, planar=self.planar, query_z=self.z)

    def cb_occupancy(self, msg):
        pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                            skip_nans=True)), dtype=float)
        self.occupancy = OccupancySet(
            pts.reshape(-1, 3), resolution=self.occ_res, planar=self.planar,
            z_band=self.occ_z_band if self.dim == 2 else None, ego_z=self.z)

    def step(self, _evt):
        if self.pos is None:
            return
        self.publish_keepout()
        self.publish_sightlines()
        if self.show_nearest:
            d, i = self.boundaries.nearest(self.ego()[None, :])
            self.pub_nearest.publish(
                nearest_markers(self.boundaries, self.pos, self.z, self.d_safe,
                                self.frame_id, float(d[0]), int(i[0])))
        if self.show_occupied:
            arr, _ = ego_occupied_markers(self.occupancy, self.ego(), self.z,
                                          self.frame_id)
            self.pub_occupied.publish(arr)
        if self.show_nearest_occ:
            arr, _ = nearest_occupied_markers(self.occupancy, self.ego(), self.z,
                                              self.frame_id)
            self.pub_nearest_occ.publish(arr)
        self.publish_status()

    def publish_keepout(self):
        pts = self.boundaries.points
        if len(pts) > 400:
            pts = pts[::int(np.ceil(len(pts) / 400.0))]

        arr = MarkerArray()
        for i, t_k in enumerate(self.keepout_times):
            r_keep = self.d_safe + self.v_target * min(t_k, self.t_grow_max)
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = rospy.Time.now()
            m.ns, m.id = "keepout", i
            m.type, m.action = Marker.SPHERE_LIST, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 2.0 * r_keep
            frac = i / float(max(len(self.keepout_times) - 1, 1))
            m.color.a = 0.30 - 0.20 * frac
            m.color.r, m.color.g, m.color.b = 1.0, 0.55 * frac, 0.0
            for p in pts:
                m.points.append(Point(p[0], p[1],
                                      p[2] if self.dim == 3 else self.z))
            arr.markers.append(m)
        self.pub_keepout.publish(arr)

    def publish_sightlines(self):
        """One segment from the drone to each of a sample of boundary voxels."""
        pts = self.boundaries.points
        if len(pts) > self.sightline_max:
            pts = pts[::int(np.ceil(len(pts) / float(self.sightline_max)))]

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns, m.id = "sightlines", 0
        m.type, m.action = Marker.LINE_LIST, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.015
        m.color.a, m.color.r, m.color.g, m.color.b = 0.5, 0.2, 1.0, 0.4
        for q in pts:
            m.points.append(Point(self.pos[0], self.pos[1], self.z))
            m.points.append(Point(q[0], q[1], q[2] if self.dim == 3 else self.z))

        arr = MarkerArray()
        arr.markers.append(m)
        self.pub_sight.publish(arr)
        self.n_drawn = len(pts)

    def publish_status(self):
        d = float(self.boundaries.distance(self.ego()[None, :])[0])
        r0 = self.d_safe
        r1 = self.d_safe + self.v_target * self.t_grow_max
        breach = np.isfinite(d) and d < r0

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns, m.id, m.type, m.action = "status", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y = self.pos[0], self.pos[1]
        m.pose.position.z = self.z + 1.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.35
        m.color.a = 1.0
        if breach:
            m.color.r, m.color.g, m.color.b = 1.0, 0.7, 0.1
        else:
            m.color.r, m.color.g, m.color.b = 0.2, 1.0, 0.4
        m.text = ("d_occ %s   r_keep %.1f->%.1f m%s\n"
                  "bnd %d   occ %d   rays %d"
                  % (("inf" if not np.isfinite(d) else "%.2f m" % d),
                     r0, r1, "   BREACH" if breach else "",
                     len(self.boundaries), len(self.occupancy),
                     getattr(self, "n_drawn", 0)))
        self.pub_status.publish(m)


if __name__ == "__main__":
    rospy.init_node("occlusion_viz")
    OcclusionViz()
    rospy.spin()
