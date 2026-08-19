"""
rviz_viz.py
===========
RViz2 visualisation feed for the MPPI controller.

Everything the planner already computes each control step, republished as
visualization_msgs/MarkerArray so RViz can draw it next to the Unity LiDAR cloud:

  /viz/rollouts              sampled MPPI rollouts (cheap→expensive colour ramp)
                             + the executed nominal plan, thick and white
  /viz/occlusion_boundaries  the detected blind-corner segments, their corners, and
                             the EXPANDING keep-out capsule at sampled horizon times
  /viz/dynamic_obstacles     sensed movers and their expanding keep-out circles
  /viz/ego                   ego pose arrow + goal marker

FRAME. Markers are published in `map`, whose axes are chosen to match the Unity
LiDAR cloud exactly (RglTcpPointCloudPublisher emits ROS x/y/z, sensor-relative,
world-aligned):

    RViz x = controller a0 (Unity Z)
    RViz y = -controller a1 (Unity X)
    RViz z = height

The cloud is sensor-relative, so map -> lidar_link is a pure TRANSLATION; this node
broadcasts it from /laser_scan_pose, which is what puts the cloud and the markers in
the same picture. Without that TF, RViz has no `map` frame at all.

A no-op when rclpy is unavailable, matching how ObstacleCircles degrades.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from occlusion_capsules import capsule_polygon

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Point, Pose, TransformStamped
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
    from tf2_ros import TransformBroadcaster
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False


def _pt(a0: float, a1: float, z: float = 0.0) -> "Point":
    """Controller world (a0, a1) -> RViz map point. The y flip is the cloud's."""
    return Point(x=float(a0), y=float(-a1), z=float(z))


def _rgba(r, g, b, a=1.0) -> "ColorRGBA":
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _polyline(pts, z=0.0):
    """(N,2) or (N,3) world polyline -> the LINE_STRIP point list.

    Tolerates a third column because OCC_PLAN carries [x, y, theta] (the heading the
    figures need to place the sensor). Only the first two are positions — a blind
    reshape(-1, 2) would fold the heading in as a coordinate and scramble the line.
    """
    a = np.asarray(pts, float)
    a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(-1, 2)
    return [_pt(q[0], q[1], z) for q in a[:, :2]]


# Colour ramp for the sampled rollouts: cheap (green) -> expensive (red).
def _cost_color(frac: float, alpha: float) -> "ColorRGBA":
    f = float(np.clip(frac, 0.0, 1.0))
    return _rgba(f, 1.0 - f, 0.15, alpha)


class RvizVisualizer:
    """Owns its own rclpy node; publish() is called once per control step."""

    FRAME = "map"
    SENSOR_FRAME = "lidar_link"

    def __init__(self, frame: str = FRAME, sensor_frame: str = SENSOR_FRAME,
                 pose_topic: str = "/laser_scan_pose", z: float = 0.0):
        self.frame = frame
        self.sensor_frame = sensor_frame
        self.pose_topic = pose_topic
        self.z = z                      # height the flat markers are drawn at
        self._node = None
        self._thread = None
        self._executor = None
        self._tf = None
        self._pubs = {}
        self._lock = threading.Lock()
        self._sensor_xyz = None         # (a0, a1, height) from /laser_scan_pose

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if not _HAS_RCLPY:
            print("[RvizVisualizer] rclpy/tf2_ros not importable — RViz feed disabled.")
            return False
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("mppi_rviz_viz")
        for name in ("rollouts", "occlusion_boundaries", "dynamic_obstacles", "ego"):
            self._pubs[name] = self._node.create_publisher(
                MarkerArray, f"/viz/{name}", 1)
        self._tf = TransformBroadcaster(self._node)

        outer = self

        def _pose_cb(msg):
            p = msg.position
            with outer._lock:
                # Unity pose -> controller world: position.z -> a0, .x -> a1, .y = height
                outer._sensor_xyz = (float(p.z), float(p.x), float(p.y))

        self._node.create_subscription(Pose, self.pose_topic, _pose_cb,
                                       qos_profile_sensor_data)
        # A DEDICATED executor, not rclpy.spin(): that attaches the node to the process-
        # wide GLOBAL executor, which ObstacleCircles is already spinning in its own
        # thread — two threads driving one executor raises "generator already executing".
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        print(f"[RvizVisualizer] publishing markers on /viz/* in frame '{self.frame}' "
              f"(set RViz Fixed Frame to '{self.frame}')")
        return True

    def _spin(self):
        """Ctrl-C tears the context down under the thread; that is a normal exit here,
        not a failure worth a traceback across the launch log."""
        try:
            self._executor.spin()
        except (ExternalShutdownException, KeyboardInterrupt, RuntimeError):
            pass

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=0.0)
        if self._node is not None:
            self._node.destroy_node()
        self._node = self._tf = self._executor = None
        self._pubs = {}

    # ── per-step publish ──────────────────────────────────────────────────────

    def publish(self, *, ego=None, goal_xy=None, plan=None, infeasible=False,
                rollouts=None, rollout_costs=None, occ_segs=None, occ_segs_all=None,
                dyn_set=None, dt=0.1, v_target=0.0, d_safe=0.0, t_grow_max=None,
                n_stages=5, n_rollouts=40):
        """Draw one control step.

        ego          [x, y, theta, v, ...] controller world state
        plan         (H,2) the executed nominal rollout (or the braking one)
        rollouts     (K,H,2) sampled rollout paths, rollout_costs (K,) their costs
        occ_segs     (M,2,2) boundaries the solve actually constrained against — only
                     these get a keep-out capsule, because only these were charged for
        occ_segs_all (M,2,2) every boundary the detector produced this scan, drawn dim:
                     the difference between the two IS the gating, and seeing a bright
                     detection with no capsule is how a mis-gated boundary shows up
        dyn_set      (K,4) [c0, c1, r_cluster, age] sensed movers
        """
        if self._node is None:
            return
        self._broadcast_tf()
        self._pubs["rollouts"].publish(
            self._rollout_markers(rollouts, rollout_costs, plan, infeasible,
                                  n_rollouts))
        self._pubs["occlusion_boundaries"].publish(
            self._occlusion_markers(occ_segs, occ_segs_all, plan, dt, v_target, d_safe,
                                    t_grow_max, n_stages))
        self._pubs["dynamic_obstacles"].publish(
            self._dynamic_markers(dyn_set, plan, dt, v_target, d_safe, n_stages))
        self._pubs["ego"].publish(self._ego_markers(ego, goal_xy))

    # ── marker builders ───────────────────────────────────────────────────────

    def _header(self, m: "Marker", ns: str, mid: int, mtype: int, scale: float):
        m.header.frame_id = self.frame
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = float(scale)
        m.pose.orientation.w = 1.0
        return m

    def _delete_all(self, ns: str) -> "Marker":
        """Clearing the namespace each step is what stops stale geometry from a
        previous solve accumulating in the view."""
        m = Marker()
        m.header.frame_id = self.frame
        m.ns, m.action = ns, Marker.DELETEALL
        return m

    def _rollout_markers(self, rollouts, costs, plan, infeasible, n_show):
        arr = MarkerArray()
        arr.markers.append(self._delete_all("rollouts"))

        if rollouts is not None and len(rollouts):
            r = np.asarray(rollouts, float)
            c = (np.asarray(costs, float) if costs is not None and len(costs) == len(r)
                 else np.zeros(len(r)))
            # Show the cheapest n_show samples: the expensive tail is a solid wall of
            # red lines that hides the plan without saying anything new.
            keep = np.argsort(c)[:int(n_show)]
            r, c = r[keep], c[keep]
            lo, hi = float(c.min()), float(c.max())
            span = max(hi - lo, 1e-9)

            m = self._header(Marker(), "rollouts", 0, Marker.LINE_LIST, 0.08)
            m.scale.x = 0.08
            for path, cost in zip(r, c):
                col = _cost_color((cost - lo) / span, 0.55)
                for i in range(len(path) - 1):
                    m.points.append(_pt(path[i, 0], path[i, 1], self.z))
                    m.points.append(_pt(path[i + 1, 0], path[i + 1, 1], self.z))
                    m.colors.append(col)
                    m.colors.append(col)
            arr.markers.append(m)

        if plan is not None and len(plan) > 1:
            m = self._header(Marker(), "plan", 1, Marker.LINE_STRIP, 0.35)
            m.scale.x = 0.35
            m.points = _polyline(plan, self.z + 0.05)
            m.color = (_rgba(1.0, 0.2, 0.2, 1.0) if infeasible
                       else _rgba(1.0, 1.0, 1.0, 1.0))
            arr.markers.append(m)

            m = self._header(Marker(), "plan", 2, Marker.SPHERE_LIST, 0.5)
            m.points = _polyline(plan, self.z + 0.05)
            m.color = _rgba(0.2, 0.6, 1.0, 0.9)
            arr.markers.append(m)
        return arr

    def _occlusion_markers(self, segs, segs_all, plan, dt, v_target, d_safe, t_grow_max,
                           n_stages):
        arr = MarkerArray()
        for ns in ("occ_seg", "occ_seg_all", "occ_corner", "occ_keepout"):
            arr.markers.append(self._delete_all(ns))
        s = (np.asarray(segs, float).reshape(-1, 2, 2)
             if segs is not None and len(segs) else np.empty((0, 2, 2)))
        s_all = (np.asarray(segs_all, float).reshape(-1, 2, 2)
                 if segs_all is not None and len(segs_all) else np.empty((0, 2, 2)))

        # Everything the detector found this scan, dim: the ones missing a keep-out
        # below were dropped by the gates.
        if len(s_all):
            m = self._header(Marker(), "occ_seg_all", 0, Marker.LINE_LIST, 0.12)
            m.scale.x = 0.12
            m.color = _rgba(0.6, 0.6, 0.25, 0.6)
            for seg in s_all:
                m.points.append(_pt(seg[0, 0], seg[0, 1], self.z))
                m.points.append(_pt(seg[1, 0], seg[1, 1], self.z))
            arr.markers.append(m)

        if not len(s):
            return arr

        # The boundary itself: corner (near endpoint) first, then the far end.
        m = self._header(Marker(), "occ_seg", 0, Marker.LINE_LIST, 0.25)
        m.scale.x = 0.25
        m.color = _rgba(1.0, 0.85, 0.0, 1.0)
        for seg in s:
            m.points.append(_pt(seg[0, 0], seg[0, 1], self.z))
            m.points.append(_pt(seg[1, 0], seg[1, 1], self.z))
        arr.markers.append(m)

        m = self._header(Marker(), "occ_corner", 0, Marker.SPHERE_LIST, 1.0)
        m.color = _rgba(1.0, 0.4, 0.0, 1.0)
        m.points = [_pt(seg[0, 0], seg[0, 1], self.z) for seg in s]
        arr.markers.append(m)

        # The keep-out the cost actually charged: radius d_safe + v_target·t_k, drawn
        # at a handful of horizon times so the growth is visible rather than implied.
        h = len(plan) if plan is not None else 0
        for i, t_k in enumerate(self._stage_times(h, dt, t_grow_max, n_stages)):
            r_k = d_safe + v_target * t_k
            m = self._header(Marker(), "occ_keepout", i, Marker.LINE_LIST, 0.12)
            m.scale.x = 0.12
            m.color = _cost_color(1.0 - i / max(n_stages - 1, 1), 0.85)
            for seg in s:
                self._append_outline(m, capsule_polygon(seg[0], seg[1], r_k))
            arr.markers.append(m)
        return arr

    def _dynamic_markers(self, dyn_set, plan, dt, v_target, d_safe, n_stages):
        arr = MarkerArray()
        arr.markers.append(self._delete_all("dyn_centre"))
        arr.markers.append(self._delete_all("dyn_keepout"))
        d = (np.asarray(dyn_set, float).reshape(-1, 4)
             if dyn_set is not None and len(dyn_set) else np.empty((0, 4)))
        if not len(d):
            return arr

        m = self._header(Marker(), "dyn_centre", 0, Marker.SPHERE_LIST, 1.2)
        m.color = _rgba(1.0, 0.3, 0.9, 1.0)
        m.points = [_pt(c0, c1, self.z) for c0, c1, _r, _a in d]
        arr.markers.append(m)

        h = len(plan) if plan is not None else 0
        for i, t_k in enumerate(self._stage_times(h, dt, None, n_stages)):
            m = self._header(Marker(), "dyn_keepout", i, Marker.LINE_LIST, 0.12)
            m.scale.x = 0.12
            m.color = _rgba(1.0, 0.3, 0.9, 0.8 - 0.1 * i)
            for c0, c1, r_c, _age in d:
                r_k = d_safe + float(r_c) + v_target * t_k
                self._append_outline(m, capsule_polygon((c0, c1), (c0, c1), r_k))
            arr.markers.append(m)
        return arr

    def _ego_markers(self, ego, goal_xy):
        arr = MarkerArray()
        arr.markers.append(self._delete_all("ego"))
        arr.markers.append(self._delete_all("goal"))
        if ego is not None:
            x, y, th = float(ego[0]), float(ego[1]), float(ego[2])
            m = self._header(Marker(), "ego", 0, Marker.ARROW, 0.6)
            m.scale.x, m.scale.y, m.scale.z = 0.8, 1.6, 1.6   # shaft d, head d, head l
            m.color = _rgba(0.1, 1.0, 0.4, 1.0)
            m.points = [_pt(x, y, self.z + 0.2),
                        _pt(x + 6.0 * np.cos(th), y + 6.0 * np.sin(th), self.z + 0.2)]
            arr.markers.append(m)
        if goal_xy is not None:
            m = self._header(Marker(), "goal", 0, Marker.SPHERE, 4.0)
            m.color = _rgba(1.0, 0.9, 0.1, 0.7)
            p = _pt(goal_xy[0], goal_xy[1], self.z)
            m.pose.position = p
            arr.markers.append(m)
        return arr

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _stage_times(horizon, dt, t_grow_max, n_stages):
        """Horizon times the keep-outs are drawn at, capped the way the cost caps its
        own growth (t_grow_max) so the drawn radius is the enforced one."""
        t_end = max(horizon, 1) * dt
        if t_grow_max is not None:
            t_end = min(t_end, float(t_grow_max))
        return np.linspace(0.0, t_end, max(int(n_stages), 1))

    @staticmethod
    def _append_outline(m, poly):
        """Closed (N,2) outline -> LINE_LIST segment pairs."""
        p = np.asarray(poly, float)
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            m.points.append(_pt(a[0], a[1]))
            m.points.append(_pt(b[0], b[1]))

    def _broadcast_tf(self):
        with self._lock:
            xyz = self._sensor_xyz
        if xyz is None or self._tf is None:
            return
        a0, a1, h = xyz
        t = TransformStamped()
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = self.frame
        t.child_frame_id = self.sensor_frame
        # Pure translation: the cloud is sensor-relative but world-ALIGNED.
        t.transform.translation.x = float(a0)
        t.transform.translation.y = float(-a1)
        t.transform.translation.z = float(h)
        t.transform.rotation.w = 1.0
        self._tf.sendTransform(t)
