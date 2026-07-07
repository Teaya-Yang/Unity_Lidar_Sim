"""
lidar_costmap.py
================
LiDAR-derived static-obstacle costmap for the MPPI planner.

Subscribes to the Unity-published PointCloud2 topic (via ROS-TCP-Endpoint), and
each control step turns the latest scan into a 2-D ego-centric DISTANCE FIELD:
for any (dz, dx) point near the ego, "how far is the nearest LiDAR-observed
static surface?". mppi() samples that field along every rollout and applies the
usual two-ring cost, so walls / parked aircraft / buildings — objects the
oracle observation never contained — get a uniform, shape-true margin.

Pipeline per update():
    latest cloud → drop NaNs (no-hit rays) → drop ground returns by height
    → convert ROS(x,y) → controller (Δz, Δx) axes → mask returns near KNOWN
    dynamic agents (they're handled by the oracle pipeline; don't double-count)
    → rasterise to an occupancy grid (80×80 m @ 0.5 m) → Euclidean distance
    transform (scipy if available, else a two-pass chamfer).

Frames: the Unity publisher packs points RELATIVE TO THE SENSOR POSITION with
world-aligned axes: ROS x = Unity Δz, ROS y = −Unity Δx, ROS z = Unity Δy
(height relative to the sensor). The controller's obstacle frame is
(axis0 = Unity Δz, axis1 = Unity Δx) — so (a0, a1) = (ros_x, −ros_y), and the
MPPI rollout displacements (fwd − s0_fwd, lat − s0_lat) are directly comparable.

The cloud is ego-relative at CAPTURE time; using it one control period later
mis-places the field by ≤ v·Δt (~0.8 m at 8 m/s). Size D_SAFE_STATIC to absorb
that. Clouds older than max_age are considered stale and the field reports
not-ready (the static cost is then skipped, never frozen).

Runs without ROS: if rclpy isn't importable, start() returns False and the
controller degrades to its previous behaviour.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

try:                                   # optional — controller must run without ROS
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False

try:                                   # optional — chamfer fallback below
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _chamfer_distance(occ: np.ndarray, res: float) -> np.ndarray:
    """
    Two-pass 3×3 chamfer distance transform (weights 1, √2) — a few-% approximation
    of the exact EDT, plenty for a 0.5 m grid. Used when scipy is unavailable.
    """
    INF = 1e6
    d = np.where(occ, 0.0, INF)
    n, m = d.shape
    w1, w2 = 1.0, np.sqrt(2.0)
    for i in range(n):                                   # forward pass
        for j in range(m):
            v = d[i, j]
            if i > 0:
                v = min(v, d[i - 1, j] + w1)
                if j > 0:     v = min(v, d[i - 1, j - 1] + w2)
                if j < m - 1: v = min(v, d[i - 1, j + 1] + w2)
            if j > 0:
                v = min(v, d[i, j - 1] + w1)
            d[i, j] = v
    for i in range(n - 1, -1, -1):                       # backward pass
        for j in range(m - 1, -1, -1):
            v = d[i, j]
            if i < n - 1:
                v = min(v, d[i + 1, j] + w1)
                if j > 0:     v = min(v, d[i + 1, j - 1] + w2)
                if j < m - 1: v = min(v, d[i + 1, j + 1] + w2)
            if j < m - 1:
                v = min(v, d[i, j + 1] + w1)
            d[i, j] = v
    return d * res


class LidarCostmap:
    """
    Usage (see taxi_controller.py --lidar-costmap):

        cm = LidarCostmap()
        cm.start()                        # rclpy subscriber on a daemon thread
        ...
        cm.update(dyn_positions)          # once per control step
        d = cm.distance(pts)              # (N,2) ego-relative → (N,) metres
    """

    def __init__(self,
                 size_m: float = 80.0,        # grid side length [m], ego-centred
                 res: float = 0.5,            # cell size [m]
                 sensor_height: float = 2.0,  # sensor above ground [m] (ground filter)
                 min_obstacle_height: float = 0.4,   # keep returns this far above ground [m]
                 max_obstacle_height: float = 12.0,  # ...and below this (bridges/noise)
                 min_range: float = 1.0,      # drop self-returns closer than this [m]
                 dyn_mask_radius: float = 6.0,# scrub returns near known dynamic agents [m]
                 max_age: float = 0.5):       # stale-cloud cutoff [s]
        self.res      = float(res)
        self.half     = float(size_m) / 2.0
        self.n        = int(round(size_m / res))
        self.z_lo     = -sensor_height + min_obstacle_height
        self.z_hi     = -sensor_height + max_obstacle_height
        self.min_range = float(min_range)
        self.dyn_mask_radius = float(dyn_mask_radius)
        self.max_age  = float(max_age)

        self._lock         = threading.Lock()
        self._latest_pts   = None      # (N,3) float32 ROS (x,y,z), NaNs already dropped
        self._latest_stamp = 0.0       # wall-clock arrival time
        self._dist         = None      # (n,n) distance field [m]
        self._ready        = False
        self._node         = None
        self._thread       = None

    # ── ROS side ──────────────────────────────────────────────────────────────

    def start(self, topic: str = "/point_cloud") -> bool:
        """Spin an rclpy subscriber on a daemon thread. False if ROS is unavailable."""
        if not _HAS_RCLPY:
            print("[LidarCostmap] rclpy not importable — costmap disabled "
                  "(source your ROS 2 setup to enable).")
            return False

        if not rclpy.ok():
            rclpy.init()
        outer = self

        class _Sub(Node):
            def __init__(self):
                super().__init__("mppi_lidar_costmap")
                self.create_subscription(PointCloud2, topic, self._cb, 5)

            def _cb(self, msg: PointCloud2):
                pts = outer._parse_cloud(msg)
                if pts is not None:
                    with outer._lock:
                        outer._latest_pts   = pts
                        outer._latest_stamp = time.monotonic()

        self._node = _Sub()
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,),
                                        daemon=True)
        self._thread.start()
        print(f"[LidarCostmap] subscribed to '{topic}'  grid={self.n}×{self.n} @ "
              f"{self.res} m  EDT={'scipy' if _HAS_SCIPY else 'chamfer fallback'}")
        return True

    @staticmethod
    def _parse_cloud(msg) -> Optional[np.ndarray]:
        """PointCloud2 → (N,3) float32 (x,y,z), finite points only."""
        step = msg.point_step
        if step < 12:
            return None
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        n = len(raw) // step
        if n == 0:
            return None
        raw = raw[:n * step].reshape(n, step)
        # x,y,z are FLOAT32 at offsets 0,4,8 (the Unity publisher's layout).
        xyz = raw[:, :12].copy().view(np.float32).reshape(n, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        return xyz if len(xyz) else None

    # ── Per-control-step build ────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready

    def update(self, dyn_positions=None) -> bool:
        """
        Rebuild the distance field from the latest cloud. Call once per control
        step, BEFORE mppi(). dyn_positions: iterable of (a0, a1) ego-relative
        positions of KNOWN dynamic agents — their returns are scrubbed so the
        static field doesn't double-count obstacles the oracle pipeline handles.
        Returns (and stores) readiness; when False the field must not be used.
        """
        with self._lock:
            pts   = self._latest_pts
            stamp = self._latest_stamp
        if pts is None or (time.monotonic() - stamp) > self.max_age:
            self._ready = False
            return False
        self._ingest(pts, dyn_positions)
        return self._ready

    def _ingest(self, pts_ros: np.ndarray, dyn_positions=None) -> None:
        """Testable core: filter → controller frame → mask → rasterise → EDT."""
        x, y, z = pts_ros[:, 0], pts_ros[:, 1], pts_ros[:, 2]

        # Height gate (z is height relative to the SENSOR): drops the ground rings
        # and anything implausibly high; range gate drops self-returns.
        keep = (z > self.z_lo) & (z < self.z_hi)
        a0, a1 = x[keep], -y[keep]                     # ROS → controller (Δz, Δx)
        rng = np.hypot(a0, a1)
        in_grid = (rng > self.min_range) & (np.abs(a0) < self.half) & (np.abs(a1) < self.half)
        a0, a1 = a0[in_grid], a1[in_grid]

        # Scrub returns belonging to known dynamic agents (oracle handles those).
        if dyn_positions is not None and len(a0):
            for p in dyn_positions:
                d2 = (a0 - float(p[0])) ** 2 + (a1 - float(p[1])) ** 2
                m = d2 > self.dyn_mask_radius ** 2
                a0, a1 = a0[m], a1[m]

        occ = np.zeros((self.n, self.n), dtype=bool)
        if len(a0):
            i = ((a0 + self.half) / self.res).astype(int)
            j = ((a1 + self.half) / self.res).astype(int)
            np.clip(i, 0, self.n - 1, out=i)
            np.clip(j, 0, self.n - 1, out=j)
            occ[i, j] = True

        if not occ.any():
            # No static returns in range: field is "everything is far" — valid.
            self._dist = np.full((self.n, self.n), 1e6, dtype=np.float64)
        elif _HAS_SCIPY:
            self._dist = distance_transform_edt(~occ) * self.res
        else:
            self._dist = _chamfer_distance(occ, self.res)
        self._ready = True

    # ── Planner-side lookup ───────────────────────────────────────────────────

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """
        pts: (N,2) ego-relative (Δz, Δx) points (e.g. MPPI rollout displacements).
        Returns (N,) distance [m] to the nearest observed static surface.
        Outside the grid → +1e6 (unknown treated as free; the lane costs and the
        finite horizon keep rollouts from exploiting far-field ignorance).
        """
        if not self._ready or self._dist is None:
            return np.full(len(pts), 1e6)
        i = ((pts[:, 0] + self.half) / self.res).astype(int)
        j = ((pts[:, 1] + self.half) / self.res).astype(int)
        inside = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        out = np.full(len(pts), 1e6)
        ii, jj = i[inside], j[inside]
        out[inside] = self._dist[ii, jj]
        return out

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
