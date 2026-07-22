"""
obstacle_circles.py
===================
Circle-based static-obstacle representation for the MPPI / NMPC planners — the
"direct circle-covering" model (robot-as-circle vs. obstacle-covering-circles),
replacing the persistent occupancy-grid costmap that used to live here.

Pipeline, per LiDAR scan (INSTANTANEOUS — no cross-step accumulation):

  1. DATA COLLECTION — the Unity LiDAR PointCloud2 gives the static environment
     (walls, parked aircraft, buildings). Obstacle returns are isolated by height.
  2. DOWN-SAMPLING — the raw obstacle points are collapsed onto a VOXEL GRID of
     side `voxel` m; each occupied voxel yields one representative point (its
     centre). This bounds the number of circles regardless of point density.
  3. OBSTACLE BOUNDING — a circle is centred on each down-sampled point. Its
     radius is HALF THE VOXEL DIAGONAL (voxel·√2/2), the smallest radius that is
     GUARANTEED to cover every raw point that fell in the voxel — so the visible
     portion of the obstacle between samples is fully enclosed with no gaps.
  4. ROBOT — modelled as a circle of radius r_robot (folded into the planners'
     D_SAFE_STATIC keep-out margin, so it is not re-encoded here).
  5. CONSTRAINT — collision avoidance is  ‖p_ego − cᵢ‖ ≥ r_robot + rᵢ  for every
     obstacle circle i. The MPC enforces this as an NLP keep-out (D_SAFE_STATIC +
     rᵢ); the MPPI applies the same margin to distance(), which returns clearance
     to the nearest circle SURFACE  minᵢ(‖p − cᵢ‖ − rᵢ).

Occlusion (range-jump blind-corner boundaries) is UNCHANGED in behaviour: it was
never a grid product — it reads the ORDERED cloud directly — so occlusion_segments()
is ported here verbatim (delegating to occlusion_capsules), letting the old
LidarCostmap be removed entirely.

FRAMES (controller (a0, a1) = (Unity Z, Unity X) axes), same as before:
  * cloud points are SENSOR-RELATIVE, WORLD-ALIGNED: ROS x = Δa0, −ROS y = Δa1,
    ROS z = height. World placement is a pure TRANSLATION by the sensor position.
  * /laser_scan_pose carries the sensor world position (position.z → a0, .x → a1).

Runs without ROS: if rclpy isn't importable, start() returns False and the
controllers degrade to no static-obstacle costs.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

import occlusion_capsules as occ_caps

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from geometry_msgs.msg import Pose
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False


class ObstacleCircles:
    """
    See module docstring. Public API used by the controllers:

        oc = ObstacleCircles(); oc.start()
        oc.configure_scan(fov_h, fov_v, res_h, res_v, max_range)   # for occlusion
        oc.update(ego_fwd)                       # once per control step
        C = oc.circles()                         # (M,3) world [a0, a1, radius], or None
        d = oc.distance(w0, w1)                   # (N,) clearance to nearest circle SURFACE
        segs = oc.occlusion_segments(...)         # (M,2,2) range-jump boundaries, or None
    """

    def __init__(self,
                 voxel: float = 2.0,
                 sensor_height: float = 3.45,
                 min_obstacle_height: float = 0.5, max_obstacle_height: float = 12.0,
                 min_range: float = 1.0, max_query_range: float = 150.0,
                 max_age: float = 0.5):
        self.voxel = float(voxel)
        # Radius that covers any point inside a square voxel from its centre.
        self.cover_r = 0.5 * self.voxel * np.sqrt(2.0)
        self.z_lo = -sensor_height + min_obstacle_height
        self.z_hi = -sensor_height + max_obstacle_height
        self.min_range = float(min_range)
        self.max_query_range = float(max_query_range)
        self.max_age = float(max_age)

        self._lock = threading.Lock()
        self._pts = None            # (N,3) ROS xyz, finite
        self._pose = None           # (a0, a1) sensor world position
        self._stamp = 0.0

        # Ordered-cloud path for range-jump occlusion SEGMENTS (occlusion_capsules).
        # Kept separate from self._pts because that one drops non-returns, which
        # destroys the beam adjacency the jump test needs.
        self._scan_shape = None     # (n_h, n_v) or None ⇒ segment path disabled
        self._scan_res_h = 1.0
        self._scan_max_range = 1000.0
        self._pts_ordered = None    # (n_h, n_v, 3) with NaN preserved

        self._circles = None        # (M,3) world [a0, a1, radius]
        self._ready = False
        self._node = self._thread = None

    @property
    def ready(self) -> bool:
        return self._ready

    # ── ROS side ──────────────────────────────────────────────────────────────

    def start(self, topic: str = "/point_cloud",
              pose_topic: str = "/laser_scan_pose") -> bool:
        if not _HAS_RCLPY:
            print("[ObstacleCircles] rclpy not importable — obstacle model disabled.")
            return False
        if not rclpy.ok():
            rclpy.init()
        outer = self

        class _Sub(Node):
            def __init__(self):
                super().__init__("obstacle_circles")
                # Sensor-data QoS (BEST_EFFORT/VOLATILE): the ros_tcp_endpoint bridge
                # publishes streamed sensor topics best-effort; a default (RELIABLE)
                # subscriber is INCOMPATIBLE with it and would silently receive nothing.
                qos = qos_profile_sensor_data
                self.create_subscription(PointCloud2, topic, self._cloud, qos)
                self.create_subscription(Pose, pose_topic, self._pose_cb, qos)
                self._got_cloud = self._got_pose = False

            def _cloud(self, msg):
                pts = outer._parse_cloud(msg)
                ordered = outer._parse_ordered(msg)
                if pts is not None:
                    with outer._lock:
                        outer._pts = pts
                        outer._pts_ordered = ordered
                        outer._stamp = time.monotonic()
                    if not self._got_cloud:
                        self._got_cloud = True
                        print(f"[ObstacleCircles] first /point_cloud message received "
                              f"({len(pts)} valid pts)")

            def _pose_cb(self, msg):
                p = msg.position
                with outer._lock:
                    outer._pose = (float(p.z), float(p.x))   # world (a0, a1)
                if not self._got_pose:
                    self._got_pose = True
                    print(f"[ObstacleCircles] first /laser_scan_pose message received "
                          f"(a0={p.z:.1f}, a1={p.x:.1f})")

        self._node = _Sub()
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()
        print(f"[ObstacleCircles] subscribed cloud='{topic}' pose='{pose_topic}'  "
              f"voxel={self.voxel:.1f} m  cover_r={self.cover_r:.2f} m")
        return True

    def configure_scan(self, fov_h: float, fov_v: float, res_h: float, res_v: float,
                       max_range: float) -> None:
        """Enable the range-jump occlusion-segment path by declaring the scan geometry.

        Must match the PointCloudPublisher Inspector fields — the flat cloud is
        reshaped back into its (azimuth x elevation) beam grid, so a mismatch makes
        adjacency meaningless. Mismatched scans are ignored rather than misread.
        """
        self._scan_shape = occ_caps.scan_shape(fov_h, fov_v, res_h, res_v)
        self._scan_res_h = float(res_h)
        self._scan_max_range = float(max_range)

    @staticmethod
    def _parse_cloud(msg) -> Optional[np.ndarray]:
        step = msg.point_step
        if step < 12:
            return None
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        n = len(raw) // step
        if n == 0:
            return None
        xyz = raw[:n * step].reshape(n, step)[:, :12].copy().view(np.float32).reshape(n, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        return xyz if len(xyz) else None

    def _parse_ordered(self, msg) -> Optional[np.ndarray]:
        if self._scan_shape is None:
            return None
        n_h, n_v = self._scan_shape
        return occ_caps.parse_ordered_cloud(msg, n_h, n_v)

    # ── Per-control-step build ────────────────────────────────────────────────

    def update(self, ego_fwd=None) -> bool:
        """Rebuild the covering circles from the latest scan. ego_fwd is accepted
        for API symmetry (the old costmap oriented an ROI wedge with it) but the
        circle model is direction-agnostic. Returns readiness."""
        with self._lock:
            pts, pose, stamp = self._pts, self._pose, self._stamp
        if pts is None or pose is None or (time.monotonic() - stamp) > self.max_age:
            self._ready = False
            return False
        self._circles = self._build_circles(pts, pose)
        self._ready = True
        return True

    def _build_circles(self, pts_ros, pose) -> Optional[np.ndarray]:
        ego0, ego1 = pose
        x, y, z = pts_ros[:, 0], pts_ros[:, 1], pts_ros[:, 2]

        # ROS → local (a0, a1); keep obstacle-height returns within the query window.
        a0, a1 = x, -y
        rng = np.hypot(a0, a1)
        keep = ((rng > self.min_range) & (rng < self.max_query_range)
                & (z > self.z_lo) & (z < self.z_hi))
        if not keep.any():
            return None

        # LOCAL → WORLD (pure translation; axes are world-aligned).
        w0 = a0[keep] + ego0
        w1 = a1[keep] + ego1

        # Voxel down-sample: one representative point per occupied voxel, placed at
        # the voxel CENTRE so the fixed cover_r radius provably encloses every raw
        # point that fell in the cell (max centre-to-corner distance = voxel·√2/2).
        vi = np.floor(w0 / self.voxel).astype(np.int64)
        vj = np.floor(w1 / self.voxel).astype(np.int64)
        _, uidx = np.unique(np.stack([vi, vj], axis=1), axis=0, return_index=True)
        c0 = (vi[uidx] + 0.5) * self.voxel
        c1 = (vj[uidx] + 0.5) * self.voxel
        r = np.full(len(c0), self.cover_r)
        return np.column_stack([c0, c1, r])

    # ── Planner-side lookups ──────────────────────────────────────────────────

    def circles(self) -> Optional[np.ndarray]:
        """(M,3) ABSOLUTE world [a0, a1, radius] covering circles, or None. The MPC
        feeds these as per-circle keep-out points; the MPPI reads them via distance()."""
        if not self._ready or self._circles is None or not len(self._circles):
            return None
        return self._circles

    def distance(self, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
        """(N,) ABSOLUTE world (a0,a1) → clearance to the nearest circle SURFACE [m]:
        minᵢ(‖p − cᵢ‖ − rᵢ). The planners then require this ≥ the robot margin
        D_SAFE_STATIC, which is exactly ‖p − cᵢ‖ ≥ D_SAFE_STATIC + rᵢ. Returns 1e6
        where there are no circles (⇒ no static penalty)."""
        w0 = np.asarray(w0, float)
        w1 = np.asarray(w1, float)
        if not self._ready or self._circles is None or not len(self._circles):
            return np.full(w0.shape, 1e6)
        c0 = self._circles[:, 0]
        c1 = self._circles[:, 1]
        r = self._circles[:, 2]
        dx = w0[:, None] - c0[None, :]
        dy = w1[:, None] - c1[None, :]
        surf = np.sqrt(dx * dx + dy * dy) - r[None, :]      # (N, M) clearance per circle
        return surf.min(axis=1)

    def occlusion_segments(self, grazing_deg: float = 12.0, min_jump: float = 0.5,
                           merge_radius: float = 1.5, max_seg_len: float = 30.0,
                           ego_fwd=None, fwd_half_angle_deg: float = 100.0,
                           min_corner_range: Optional[float] = None) -> Optional[np.ndarray]:
        """(M,2,2) world occlusion BOUNDARY segments from LiDAR range jumps, or None.

        Ported unchanged from the old costmap: reads the ORDERED cloud directly
        (not any grid), so occlusion-aware behaviour is bit-for-bit the same.
        Returns None when configure_scan() was never called or the scan is unusable.
        """
        with self._lock:
            xyz, pose = self._pts_ordered, self._pose
        if xyz is None or pose is None:
            return None
        return occ_caps.segments_from_ordered_cloud(
            xyz, self._scan_res_h, self._scan_max_range, pose,
            grazing_deg=grazing_deg, min_jump=min_jump,
            merge_radius=merge_radius, max_seg_len=max_seg_len,
            ego_fwd=ego_fwd, fwd_half_angle_deg=fwd_half_angle_deg,
            min_corner_range=(self.min_range if min_corner_range is None
                              else min_corner_range))

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
