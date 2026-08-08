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

from occlusion_mppi.boundary import BoundarySet, OccupancySet  # noqa: E402
from occlusion_mppi.dynamics import DoubleIntegrator     # noqa: E402
from occlusion_mppi.mppi import OcclusionMPPI, MPPIConfig  # noqa: E402
from occlusion_mppi.viz import (nearest_markers, ego_occupied_markers,  # noqa: E402
                                nearest_occupied_markers)  # noqa: E402


def _opt_float(v):
    """None/'' -> None, so a bound can be switched off from a launch file."""
    if v is None or v == "" or (isinstance(v, str) and v.lower() == "none"):
        return None
    return float(v)


class MPPINode(object):
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.goal = np.array(rospy.get_param("~goal", [0.0, 12.0]), dtype=float)
        self.cruise_z = rospy.get_param("~cruise_z", 1.0)
        self.goal_z = rospy.get_param("~goal_z", None)
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.goal_tol = rospy.get_param("~goal_tol", 0.5)
        # Only boundaries within this vertical band of the drone matter while
        # flying at fixed altitude; a boundary 4 m overhead cannot be hit
        # laterally and would inflate the keep-out for nothing.
        # With planar=False this only trims the set; the 3D distance already
        # discounts height. Keep it wide enough not to clip real structure.
        self.z_band = rospy.get_param("~z_band", 1.5)
        # 3 => the ego climbs, so it can fly OVER an occluder. 2 keeps the old
        # fixed-altitude baseline. Everything downstream keys off this: the plant
        # dimension, whether the occupancy test folds z away, and whether the
        # z_band pre-filters are applied at all.
        self.dim = int(rospy.get_param("~dim", 3))
        # Planar geometry only ever makes sense for a 2D ego. In 3D a folded
        # occupancy column would make a wall solid at every altitude, so there
        # would be no way over it.
        self.planar = self.dim == 2
        # Must match rog_map/inflation_resolution of the cloud being subscribed:
        # too small and rollouts step between voxel centres without ever landing
        # in one, so the collision test silently never fires.
        self.occ_res = rospy.get_param("~occ_resolution", 0.2)
        # Much tighter than z_band: with planar=True the occupancy test folds z
        # away, so anything kept here makes its whole (x,y) column solid. At 1.5 m
        # the floor would be included and the entire map would read as occupied.
        # Drone half-height. Unused in 3D -- nothing is folded, so nothing needs
        # banding, and banding would delete the geometry the ego climbs over.
        self.occ_z_band = rospy.get_param("~occ_z_band", 0.3)

        cfg = MPPIConfig(
            horizon=int(rospy.get_param("~horizon", 30)),
            samples=int(rospy.get_param("~samples", 1000)),
            dt=float(rospy.get_param("~dt", 0.1)),
            d_safe=float(rospy.get_param("~d_safe", 1.0)),
            v_target=float(rospy.get_param("~v_target", 1.5)),
            t_grow_max=float(rospy.get_param("~t_grow_max", 3.0)),
            w_soft=float(rospy.get_param("~w_soft", 50.0)),
            d_infl=float(rospy.get_param("~d_infl", 1.0)),
            w_collision=float(rospy.get_param("~w_collision", 1.0e6)),
            # Settling authority. At the default 1.2 the damping is ~0.12*v^2 per
            # stage against a goal pull of 2*d, so rushing the goal always beats
            # slowing for it and the ego orbits instead of arriving.
            w_v=float(rospy.get_param("~w_v", 1.2)),
            use_occlusion=bool(rospy.get_param("~use_occlusion", True)),
            # inf_occ carries no floor (ROG-Map's virtual ground never reaches the
            # occupancy buffer), so without these the ego dives under an occluder
            # instead of climbing over it.
            z_min=_opt_float(rospy.get_param("~z_min", 0.5)),
            z_max=_opt_float(rospy.get_param("~z_max", 3.5)),
        )
        self.cfg = cfg

        plant = DoubleIntegrator(dim=self.dim, dt=cfg.dt,
                                 v_max=float(rospy.get_param("~v_max", 2.0)),
                                 a_max=float(rospy.get_param("~a_max", 3.0)))
        self.plant = plant
        self.planner = OcclusionMPPI(plant, cfg, rng=np.random.default_rng(0))

        # Accept [x,y] or [x,y,z] regardless of dim, and normalise to exactly dim
        # here. Everything downstream -- the goal cost and the goal-reached test
        # against self.pos -- assumes the two match, and a 3-element goal against
        # a 2D pose raises a broadcast error rather than anything readable.
        if len(self.goal) < self.dim:
            z = self.cruise_z if self.goal_z is None else float(self.goal_z)
            self.goal = np.append(self.goal, z)
        elif len(self.goal) > self.dim:
            rospy.logwarn("[mppi] goal has %d elements but dim=%d -- ignoring z=%.2f",
                          len(self.goal), self.dim, self.goal[2])
            self.goal = self.goal[:self.dim]
        if len(self.goal) != self.dim:
            raise ValueError("~goal must have 2 or %d elements, got %d"
                             % (self.dim, len(self.goal)))

        # Placeholders must match the dimension the callbacks will build at:
        # they are queried on every tick before the first cloud arrives, and a
        # planar placeholder under dim=3 is a different key space, not an empty one.
        self.boundaries = BoundarySet(np.zeros((0, 3)), planar=self.planar,
                                      query_z=self.cruise_z)
        self.occupancy = OccupancySet(np.zeros((0, 3)), resolution=self.occ_res,
                                      planar=self.planar)
        self.pos = None
        self.vel = np.zeros(self.dim)
        # The setpoint is integrated, exactly like keyboard_control.py, rather than
        # being snapped to the measured pose each tick -- feeding the measured pose
        # back into the setpoint would close a second loop through the plant and
        # let tracking error accumulate into the command.
        self.sp = None
        self.at_goal = False

        self.pub_cmd = rospy.Publisher("/planning/pos_cmd", PositionCommand, queue_size=10)
        self.pub_viz = rospy.Publisher("~rollouts", MarkerArray, queue_size=1)
        self.pub_keepout = rospy.Publisher("~keepout", MarkerArray, queue_size=1)
        self.pub_status = rospy.Publisher("~status", Marker, queue_size=1)
        self.pub_sight = rospy.Publisher("~sightlines", MarkerArray, queue_size=1)
        # The single distance the occlusion cost is actually reacting to, drawn as
        # the vector that produced it. Everything else in the keep-out viz is a
        # region; this is the one scalar, so when the drone stops for no visible
        # reason this is the marker that says which voxel did it.
        self.pub_nearest = rospy.Publisher("~nearest_boundary", MarkerArray,
                                           queue_size=1)
        # Latched: the goal never moves, and RViz is usually started last.
        self.pub_goal = rospy.Publisher("~goal", MarkerArray, queue_size=1,
                                        latch=True)
        self.publish_goal()
        # Scalar d_occ for time-series plotting (rqt_plot ~d_occ/data).
        from std_msgs.msg import Float32
        self._Float32 = Float32
        self.pub_docc = rospy.Publisher("~d_occ", Float32, queue_size=10)

        # One marched ray per drawn sightline, so this is the one viz that costs
        # real time. Keep it well under the boundary count.
        self.sightline_max = int(rospy.get_param("~sightline_max", 120))
        self.show_nearest = bool(rospy.get_param("~show_nearest", True))
        # One membership test per tick on the measured pose. Cheap, but it is a
        # diagnostic rather than something the controller needs.
        self.show_occupied = bool(rospy.get_param("~show_occupied", True))
        # Builds a KD-tree over inf_occ (tens of thousands of points) whenever the
        # cloud changes. Nothing in the planner needs it -- turn it off if the
        # solve rate suffers.
        self.show_nearest_occ = bool(rospy.get_param("~show_nearest_occupied", True))
        self.pub_nearest_occ = rospy.Publisher("~nearest_occupied", MarkerArray,
                                               queue_size=1)
        self.pub_occupied = rospy.Publisher("~ego_occupied", MarkerArray, queue_size=1)

        # Horizon times [s] at which to draw the keep-out. Anything past
        # t_grow_max is the same radius, so the last useful slice is the cap.
        self.keepout_times = rospy.get_param(
            "~keepout_times", [0.0, 0.5 * cfg.t_grow_max, cfg.t_grow_max])
        # One sphere per frontier voxel per slice; the frontier routinely runs to
        # 10k+ voxels, which RViz will not draw at interactive rates. Stride down.
        self.keepout_max_pts = int(rospy.get_param("~keepout_max_pts", 400))

        # /Odometry_imu (200 Hz, IMU-extrapolated), NOT /lidar_slam/odom: the
        # latter is stamped at scan-end and trails the true pose by ~0.35 m at
        # 1.2 m/s, which is a whole goal_tol. Planning on a pose that stale is
        # what makes the ego overshoot and orbit the goal instead of settling.
        rospy.Subscriber(rospy.get_param("~odom_topic", "/Odometry_imu"),
                         Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber(rospy.get_param("~boundary_topic",
                                         "/rm_node/occlusion_frontier"),
                         PointCloud2, self.cb_boundary, queue_size=1)
        rospy.Subscriber(rospy.get_param("~occupancy_topic",
                                         "/rm_node/rog_map/inf_occ"),
                         PointCloud2, self.cb_occupancy, queue_size=1)

        rospy.loginfo("[mppi] dim=%dD goal=%s", self.dim, self.goal.tolist())
        rospy.loginfo("[mppi] goal=%s occlusion=%s d_safe=%.1f v_target=%.1f "
                      "t_grow_max=%.1f => keep-out grows %.1f -> %.1f m",
                      self.goal.tolist(), cfg.use_occlusion, cfg.d_safe,
                      cfg.v_target, cfg.t_grow_max,
                      cfg.d_safe, cfg.d_safe + cfg.v_target * cfg.t_grow_max)
        if not cfg.use_occlusion:
            rospy.logwarn("[mppi] occlusion term DISABLED -- collision only (baseline)")

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.step)

    def ego_z(self):
        """Altitude to draw at: the live one in 3D, the fixed cruise one in 2D."""
        if self.dim == 3 and self.pos is not None:
            return float(self.pos[2])
        return float(self.cruise_z)

    def cb_odom(self, msg):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        if self.dim == 3:
            self.pos = np.array([p.x, p.y, p.z])
            self.vel = np.array([v.x, v.y, v.z])
        else:
            self.pos = np.array([p.x, p.y])
            self.vel = np.array([v.x, v.y])
        if self.sp is None:
            self.sp = np.array([p.x, p.y, p.z if self.dim == 3 else self.cruise_z])

    def cb_boundary(self, msg):
        pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                            skip_nans=True)), dtype=float)
        ego_z = self.sp[2] if self.sp is not None else self.cruise_z
        # z_band pre-filter is a 2D-cost device: it drops boundaries the planar
        # distance would misjudge. In 3D the distance already accounts for
        # height, and banding would hide exactly the top edge the ego flies over.
        self.boundaries = BoundarySet(
            pts.reshape(-1, 3),
            z_band=self.z_band if self.dim == 2 else None,
            ego_z=ego_z, planar=self.planar, query_z=ego_z)

    def cb_occupancy(self, msg):
        pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                            skip_nans=True)), dtype=float)
        # inf_occ flickers to zero points under load. An empty message is not
        # "the world became free": planning one blind cycle at 2 m/s walks the
        # ego straight through a wall the previous message contained. Keep the
        # last real map until a non-empty one arrives.
        if len(pts) == 0 and self.occupancy is not None and len(self.occupancy):
            return
        ego_z = self.sp[2] if self.sp is not None else self.cruise_z
        # planar/z_band are 2D-cost devices: folding z makes a wall solid at every
        # altitude, so in 3D there would be no way over it.
        self.occupancy = OccupancySet(
            pts.reshape(-1, 3), resolution=self.occ_res, planar=self.planar,
            z_band=self.occ_z_band if self.dim == 2 else None, ego_z=ego_z)

    def step(self, _evt):
        if self.pos is None or self.sp is None:
            return

        # Hysteresis: releasing at the same radius it latches at makes the ego
        # chatter in and out of "arrived" on estimator noise, which reads as
        # circling. Latch at goal_tol, release only at 1.5x.
        d_goal = float(np.linalg.norm(self.pos - self.goal))
        if d_goal < self.goal_tol:
            self.at_goal = True
        elif d_goal > 1.5 * self.goal_tol:
            self.at_goal = False
        if self.at_goal:
            # Park the setpoint ON the goal rather than wherever sp drifted to,
            # then hold it: publishing zero velocity alone leaves the PID
            # chasing a stale sp offset from the goal.
            self.sp[:self.dim] = self.goal[:self.dim]
            self.publish_cmd(np.zeros(self.dim))
            rospy.loginfo_throttle(5.0, "[mppi] goal reached (d=%.2f)", d_goal)
            return

        action, info = self.planner.plan(self.pos, self.vel, self.goal,
                                         self.boundaries, self.occupancy)

        # Every rollout colliding is degenerate: all costs equal, the softmax is
        # uniform and the mean action is the stale nominal -- which WALKS THE EGO
        # THROUGH THE WALL it is facing. Do not follow it: brake at full
        # authority until some rollout survives again.
        if info["frac_collide"] > 0.99:
            rospy.logwarn_throttle(
                2.0, "[mppi] ALL rollouts collide (occ=%d) -- braking instead "
                "of following the degenerate mean", len(self.occupancy))
            speed = float(np.linalg.norm(self.vel))
            action = (-self.plant.a_max * self.vel / speed if speed > 1e-3
                      else np.zeros(self.dim))

        # Integrate the setpoint with the planned acceleration.
        self.vel_cmd = np.clip(self.vel + action * self.cfg.dt,
                               -self.plant.v_max, self.plant.v_max)
        self.publish_cmd(self.vel_cmd)

        # nearest() rather than distance(): the index is what lets publish_nearest
        # draw the actual voxel, and it is free -- the KD-tree returns it anyway.
        d_arr, i_arr = self.boundaries.nearest(self.pos[None, :])
        d_now, i_now = float(d_arr[0]), int(i_arr[0])
        if np.isfinite(d_now):
            self.pub_docc.publish(self._Float32(data=d_now))
        rospy.loginfo_throttle(
            2.0, "[mppi] pos=(%.1f,%.1f) |v|=%.2f  d_occ=%s  boundaries=%d  occ=%d  "
            "infeas=%.2f  collide=%.2f  solve=%.1fms (%.1fHz, %.2fus/rollout-step)%s",
            self.pos[0], self.pos[1], float(np.linalg.norm(self.vel)),
            ("inf" if not np.isfinite(d_now) else "%.2f" % d_now),
            len(self.boundaries), len(self.occupancy),
            info["frac_infeasible"], info["frac_collide"],
            info["solve_s"] * 1e3, info["solve_hz"], info["us_per_rollout_step"],
            "  IN OCCUPIED CELL" if self.occupancy.inside(self.pos[None, :])[0] else "")

        # The timer fires at rate_hz; if a solve outlasts its own period the loop is
        # already late and the integrated setpoint no longer matches real dt.
        if info["solve_s"] > 1.0 / self.rate_hz:
            rospy.logwarn_throttle(
                2.0, "[mppi] solve %.0fms exceeds the %.0fms control period -- "
                "lower ~samples/horizon or ~rate",
                info["solve_s"] * 1e3, 1000.0 / self.rate_hz)

        self.publish_rollouts(info)
        if self.show_nearest:
            self.publish_nearest(d_now, i_now)
        if self.show_occupied:
            arr, _ = ego_occupied_markers(self.occupancy, self.pos, self.ego_z(),
                                          self.frame_id)
            self.pub_occupied.publish(arr)
        if self.show_nearest_occ:
            arr, _ = nearest_occupied_markers(self.occupancy, self.pos, self.ego_z(),
                                              self.frame_id)
            self.pub_nearest_occ.publish(arr)
        self.publish_keepout()
        self.publish_status(info, d_now)
        self.publish_sightlines()

    def publish_cmd(self, vel_cmd):
        dt = 1.0 / self.rate_hz
        self.sp[0] += vel_cmd[0] * dt
        self.sp[1] += vel_cmd[1] * dt
        if self.dim == 3:
            self.sp[2] += vel_cmd[2] * dt
            # Hard clamp: the ground is NOT in inf_occ (ROG-Map's virtual ground
            # is a z-test, never published), so the cost alone cannot stop a
            # dive. The commanded setpoint must never leave the altitude band.
            lo = self.cfg.z_min if self.cfg.z_min is not None else -np.inf
            hi = self.cfg.z_max if self.cfg.z_max is not None else np.inf
            z_free = self.sp[2]
            self.sp[2] = float(np.clip(self.sp[2], lo, hi))
            # At the bound, also stop commanding vertical speed -- a clamped
            # position with a live vz makes the PID overshoot past the floor.
            if self.sp[2] != z_free:
                vel_cmd = vel_cmd.copy()
                vel_cmd[2] = 0.0
        else:
            self.sp[2] = self.cruise_z
        vz = vel_cmd[2] if self.dim == 3 else 0.0

        m = PositionCommand()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.frame_id
        m.position.x, m.position.y, m.position.z = self.sp
        m.velocity.x, m.velocity.y, m.velocity.z = vel_cmd[0], vel_cmd[1], vz
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
        z0 = self.ego_z()
        for i in idx:
            for k in range(traj.shape[1] - 1):
                a, b = traj[i, k, :self.dim], traj[i, k + 1, :self.dim]
                m.points.append(Point(a[0], a[1], a[2] if self.dim == 3 else z0))
                m.points.append(Point(b[0], b[1], b[2] if self.dim == 3 else z0))
        arr.markers.append(m)

        best = Marker()
        best.header = m.header
        best.ns, best.id, best.type, best.action = "best", 1, Marker.LINE_STRIP, Marker.ADD
        best.scale.x = 0.08
        best.color.a, best.color.r, best.color.g, best.color.b = 1.0, 0.1, 1.0, 0.2
        best.pose.orientation.w = 1.0
        for k in range(info["best"].shape[0]):
            p = info["best"][k, :self.dim]
            best.points.append(Point(p[0], p[1], p[2] if self.dim == 3 else z0))
        arr.markers.append(best)

        self.pub_viz.publish(arr)

    def publish_nearest(self, d_now, i_now):
        """Draw the ego -> nearest-boundary vector. See occlusion_mppi.viz."""
        self.pub_nearest.publish(
            nearest_markers(self.boundaries, self.pos, self.ego_z(),
                            self.cfg.d_safe, self.frame_id, d_now, i_now))

    def publish_keepout(self):
        """The keep-out volume the occlusion term actually enforces, per horizon time.

        One SPHERE_LIST per time slice: same voxel centres, radius r_keep(t_k) from
        cost.py. This is the region rollouts are charged w_collision for entering, so
        a rollout crossing a drawn sphere is a bug in one of the two, not in RViz.
        """
        from geometry_msgs.msg import Point
        pts = self.boundaries.points
        if len(pts) > self.keepout_max_pts:
            pts = pts[::int(np.ceil(len(pts) / float(self.keepout_max_pts)))]

        arr = MarkerArray()
        for i, t_k in enumerate(self.keepout_times):
            t_eff = min(t_k, self.cfg.t_grow_max)
            r_keep = self.cfg.d_safe + self.cfg.v_target * t_eff

            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = rospy.Time.now()
            m.ns, m.id = "keepout", i
            m.type, m.action = Marker.SPHERE_LIST, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 2.0 * r_keep
            # Later slices are bigger and would bury the earlier ones; fade them out.
            frac = i / float(max(len(self.keepout_times) - 1, 1))
            m.color.a = 0.30 - 0.20 * frac
            m.color.r, m.color.g, m.color.b = 1.0, 0.55 * frac, 0.0
            # Drawn at the flight altitude, not the voxel's own z: the cost folds z
            # away (planar=True), so the enforced shape really is a column.
            for p in pts:
                m.points.append(Point(p[0], p[1],
                                      p[2] if self.dim == 3 else self.cruise_z))
            arr.markers.append(m)

        self.pub_keepout.publish(arr)

    def publish_goal(self):
        """Goal as a sphere sized to goal_tol, so 'inside the sphere' == arrived."""
        gx, gy = float(self.goal[0]), float(self.goal[1])
        gz = float(self.goal[2]) if self.dim == 3 else float(self.cruise_z)

        arr = MarkerArray()
        s = Marker()
        s.header.frame_id = self.frame_id
        s.header.stamp = rospy.Time.now()
        s.ns, s.id, s.type, s.action = "goal", 0, Marker.SPHERE, Marker.ADD
        s.pose.position.x, s.pose.position.y, s.pose.position.z = gx, gy, gz
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 2.0 * self.goal_tol
        s.color.r, s.color.g, s.color.b, s.color.a = 0.1, 0.9, 0.2, 0.45
        arr.markers.append(s)

        t = Marker()
        t.header.frame_id = self.frame_id
        t.header.stamp = s.header.stamp
        t.ns, t.id, t.type, t.action = "goal", 1, Marker.TEXT_VIEW_FACING, Marker.ADD
        t.pose.position.x, t.pose.position.y = gx, gy
        t.pose.position.z = gz + self.goal_tol + 0.4
        t.pose.orientation.w = 1.0
        t.scale.z = 0.45
        t.color.r, t.color.g, t.color.b, t.color.a = 0.1, 0.9, 0.2, 1.0
        t.text = "goal (%.1f, %.1f, %.1f)  tol %.1fm" % (gx, gy, gz, self.goal_tol)
        arr.markers.append(t)

        self.pub_goal.publish(arr)

    def publish_status(self, info, d_now):
        """Live readout of WHICH term is binding, drawn above the drone.

        The two failure modes look identical from the outside -- the drone stops --
        so the numbers that separate them are what this shows: d_occ vs r_keep says
        the occlusion term is biting, frac_collide says the occupancy term is.
        """
        c = self.cfg
        r0 = c.d_safe
        r1 = c.d_safe + c.v_target * c.t_grow_max
        breach = np.isfinite(d_now) and d_now < r0

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns, m.id, m.type, m.action = "status", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y = self.pos[0], self.pos[1]
        m.pose.position.z = self.ego_z() + 1.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.35
        m.color.a = 1.0
        if info["frac_infeasible"] > 0.99:
            m.color.r, m.color.g, m.color.b = 1.0, 0.2, 0.2
        elif breach:
            m.color.r, m.color.g, m.color.b = 1.0, 0.7, 0.1
        else:
            m.color.r, m.color.g, m.color.b = 0.2, 1.0, 0.4

        m.text = (
            "d_occ %s   r_keep %.1f->%.1f m%s\n"
            "infeas %.0f%%   collide %.0f%%\n"
            "bnd %d   occ %d   %.1f Hz"
            % (("inf" if not np.isfinite(d_now) else "%.2f m" % d_now),
               r0, r1, "   BREACH" if breach else "",
               100.0 * info["frac_infeasible"], 100.0 * info["frac_collide"],
               len(self.boundaries), len(self.occupancy), info["solve_hz"]))
        self.pub_status.publish(m)

    def publish_sightlines(self):
        """One segment from the drone to each of a sample of boundary voxels."""
        from geometry_msgs.msg import Point
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
        z0 = self.ego_z()
        for q in pts:
            m.points.append(Point(self.pos[0], self.pos[1], z0))
            m.points.append(Point(q[0], q[1], q[2] if self.dim == 3 else z0))

        arr = MarkerArray()
        arr.markers.append(m)
        self.pub_sight.publish(arr)


if __name__ == "__main__":
    rospy.init_node("occlusion_mppi")
    MPPINode()
    rospy.spin()
