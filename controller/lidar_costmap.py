"""
lidar_costmap.py
================
LiDAR-derived world map for the MPPI planner, with two products:

  1. STATIC-OBSTACLE distance field — margin from walls / parked aircraft /
     buildings (objects the oracle observation never contained).
  2. VISIBILITY (active-perception) field — how much currently-OCCLUDED,
     path-relevant space stays hidden from each candidate ego position, so the
     planner can trade a wider arc for seeing into a blind corner.

Both derive from a PERSISTENT, world-fixed three-state occupancy grid
(OCCUPIED / FREE / UNKNOWN) accumulated across control steps. Memory matters:
without it the visibility term oscillates (peek → relax → re-occlude → peek),
because a memoryless map re-flags a just-cleared region as hidden the instant
the ego moves back. With three states + decay:

  * a scan HITS a cell            → OCCUPIED, timestamp now
  * a ray PASSES THROUGH a cell   → FREE,     timestamp now  (free-space carving)
  * never observed / behind an occluder → stays UNKNOWN
  * FREE/OCC cells older than a TTL → revert to UNKNOWN

So once the ego peeks and sees a blind region is empty, those cells become FREE
and leave the "hidden" set — the ego does NOT re-peek — until the observation
goes stale (free_ttl seconds later), at which point re-checking is warranted.

FRAMES (all in the controller's (a0, a1) = (Unity Z, Unity X) axes):
  * The Unity publisher packs cloud points SENSOR-RELATIVE with WORLD-ALIGNED
    axes: ROS x = Unity Δz = a0, −ROS y = Unity Δx = a1, ROS z = height. Because
    the axes are world-aligned, placing a scan in the world needs only the sensor
    TRANSLATION (no rotation).
  * The /laser_scan_pose PoseMsg carries the sensor world position in raw Unity
    coords: position.z → world a0, position.x → world a1.
  * Rollout displacements (fwd − s0_fwd, lat − s0_lat) are world Δ from the
    current ego, so distance()/hidden_fraction() take those directly.

Runs without ROS: if rclpy isn't importable, start() returns False and the
controller degrades to no LiDAR costs.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from geometry_msgs.msg import PoseStamped, Pose
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False

try:
    from scipy.ndimage import distance_transform_edt, binary_dilation
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Three-state cell values.
UNKNOWN = np.int8(0)
FREE    = np.int8(1)
OCC     = np.int8(2)

# Sightline frontier: an UNKNOWN cell counts as an occlusion shadow only if it lies within this
# many metres of an OCCUPIED cell. Excludes the open sensor-range rim (far from any object) so the
# sightline speed cap responds to real blind spots, not to the edge of the observed area. Larger =
# treats more of the deep shadow behind big occluders as a frontier (slows earlier/further out).
SHADOW_RADIUS = 4.0


def _chamfer_distance(occ: np.ndarray, res: float) -> np.ndarray:
    """Two-pass 3×3 chamfer EDT (weights 1, √2) — scipy fallback."""
    INF = 1e6
    d = np.where(occ, 0.0, INF)
    n, m = d.shape
    w2 = np.sqrt(2.0)
    for i in range(n):
        for j in range(m):
            v = d[i, j]
            if i > 0:
                v = min(v, d[i - 1, j] + 1.0)
                if j > 0:     v = min(v, d[i - 1, j - 1] + w2)
                if j < m - 1: v = min(v, d[i - 1, j + 1] + w2)
            if j > 0: v = min(v, d[i, j - 1] + 1.0)
            d[i, j] = v
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            v = d[i, j]
            if i < n - 1:
                v = min(v, d[i + 1, j] + 1.0)
                if j > 0:     v = min(v, d[i + 1, j - 1] + w2)
                if j < m - 1: v = min(v, d[i + 1, j + 1] + w2)
            if j < m - 1: v = min(v, d[i, j + 1] + 1.0)
            d[i, j] = v
    return d * res


class PersistentGrid:
    """
    Rolling, world-fixed three-state occupancy grid with free-space carving and
    time decay. The window (n×n cells) recentres on the ego each update; because
    the cloud axes are world-aligned, recentring is a pure integer-cell shift
    (np.roll + clear the newly-exposed border to UNKNOWN) — no rotation.
    """

    def __init__(self, size_m: float, res: float,
                 free_ttl: float, occ_ttl: float, carve_samples: int = 40):
        self.res  = float(res)
        self.n    = int(round(size_m / res))
        self.half = self.n * self.res / 2.0
        self.free_ttl = float(free_ttl)
        self.occ_ttl  = float(occ_ttl)
        self.carve_samples = int(carve_samples)

        self.state = np.full((self.n, self.n), UNKNOWN, dtype=np.int8)
        self.seen  = np.zeros((self.n, self.n), dtype=np.float64)   # last-observed time
        self._o0 = 0.0   # world coord (a0, a1) of grid cell [0,0]'s lower corner
        self._o1 = 0.0
        self._init = False

    # World (a0,a1) → integer cell, given current origin. Vectorised.
    def world_to_cell(self, w0, w1):
        i = np.floor((np.asarray(w0) - self._o0) / self.res).astype(int)
        j = np.floor((np.asarray(w1) - self._o1) / self.res).astype(int)
        return i, j

    def ego_cell(self, ego0: float, ego1: float):
        i, j = self.world_to_cell(np.array([ego0]), np.array([ego1]))
        return int(i[0]), int(j[0])

    def _recenter(self, ego0: float, ego1: float) -> None:
        """Shift the window so the ego is centred; clear exposed border to UNKNOWN."""
        new_o0 = ego0 - self.half
        new_o1 = ego1 - self.half
        if not self._init:
            self._o0, self._o1, self._init = new_o0, new_o1, True
            return
        si = int(round((new_o0 - self._o0) / self.res))
        sj = int(round((new_o1 - self._o1) / self.res))
        if si == 0 and sj == 0:
            return
        # Roll so world cells keep their content; rolled-in strips are new area.
        self.state = np.roll(self.state, (-si, -sj), axis=(0, 1))
        self.seen  = np.roll(self.seen,  (-si, -sj), axis=(0, 1))
        if si > 0:   self.state[-si:, :] = UNKNOWN; self.seen[-si:, :] = 0.0
        elif si < 0: self.state[:-si, :] = UNKNOWN; self.seen[:-si, :] = 0.0
        if sj > 0:   self.state[:, -sj:] = UNKNOWN; self.seen[:, -sj:] = 0.0
        elif sj < 0: self.state[:, :-sj] = UNKNOWN; self.seen[:, :-sj] = 0.0
        self._o0 += si * self.res
        self._o1 += sj * self.res

    def update(self, ego0: float, ego1: float,
               occ0: np.ndarray, occ1: np.ndarray,
               carve0: np.ndarray, carve1: np.ndarray, now: float) -> None:
        """
        Integrate one scan. ego0/ego1: sensor world (a0,a1).
        occ0/occ1  : world (a0,a1) of OBSTACLE-height returns → OCCUPIED cells.
        carve0/carve1: world (a0,a1) of ALL returns (incl. ground) → the ray to
        each marks the cells it passes through FREE. Using all returns (not just
        obstacle ones) is what lets looking at open ground register as FREE, so a
        peeked-and-empty region actually leaves the UNKNOWN set (no re-peek).
        """
        self._recenter(ego0, ego1)
        n = self.n

        # ── Free-space carving: sample along sensor → each return, mark FREE ───
        if len(carve0):
            fr = np.linspace(0.0, 1.0, self.carve_samples, endpoint=False)[1:]   # skip origin
            s0 = ego0 + np.outer(fr, (carve0 - ego0))       # (S, P)
            s1 = ego1 + np.outer(fr, (carve1 - ego1))
            ci, cj = self.world_to_cell(s0.ravel(), s1.ravel())
            m = (ci >= 0) & (ci < n) & (cj >= 0) & (cj < n)
            ci, cj = ci[m], cj[m]
            self.state[ci, cj] = FREE
            self.seen[ci, cj]  = now

        # ── Hits → OCCUPIED (after carving, so occupancy wins on the hit cell) ─
        if len(occ0):
            hi, hj = self.world_to_cell(occ0, occ1)
            m = (hi >= 0) & (hi < n) & (hj >= 0) & (hj < n)
            hi, hj = hi[m], hj[m]
            self.state[hi, hj] = OCC
            self.seen[hi, hj]  = now

        # ── Decay: stale observations revert to UNKNOWN ───────────────────────
        age = now - self.seen
        stale_free = (self.state == FREE) & (age > self.free_ttl)
        stale_occ  = (self.state == OCC)  & (age > self.occ_ttl)
        self.state[stale_free | stale_occ] = UNKNOWN


class LidarCostmap:
    """
    See module docstring. Public API used by taxi_controller.py:

        cm = LidarCostmap(); cm.start()
        cm.update(ego_fwd, dyn_positions)       # once per control step
        d   = cm.distance(w0, w1)               # (N,) static-surface distance [m], ABSOLUTE world
        occ = cm.occupancy(w0, w1)              # (N,) {occupied:1, free:0, unknown:-1}, ABS world
        hidden = cm.hidden_fraction(pts)        # (N,) occluded-ROI fraction [0..1], ego-relative Δ
    """

    def __init__(self,
                 size_m: float = 300.0, res: float = 0.5,
                 sensor_height: float = 3.45,
                 min_obstacle_height: float = 0.5, max_obstacle_height: float = 12.0,
                 min_range: float = 1.0, dyn_mask_radius: float = 6.0,
                 max_age: float = 0.5,
                 free_ttl: float = 4.0, occ_ttl: float = 12.0, carve_samples: int = 80,
                 # visibility params
                 roi_range: float = 70.0, roi_half_angle_deg: float = 70.0,
                 max_roi_cells: int = 48,
                 cand_reach: float = 70.0, cand_res: float = 3.0,
                 los_samples: int = 20, enable_visibility: bool = False):
        self.enable_visibility = bool(enable_visibility)
        self.res  = float(res)
        self.half = float(size_m) / 2.0
        self.z_lo = -sensor_height + min_obstacle_height
        self.z_hi = -sensor_height + max_obstacle_height
        self.min_range = float(min_range)
        self.dyn_mask_radius = float(dyn_mask_radius)
        self.max_age = float(max_age)

        self.grid = PersistentGrid(size_m, res, free_ttl, occ_ttl, carve_samples)

        self.roi_range = float(roi_range)
        self.roi_cos   = float(np.cos(np.radians(roi_half_angle_deg)))
        self.max_roi_cells = int(max_roi_cells)
        self.cand_reach = float(cand_reach)
        self.cand_res   = float(cand_res)
        self.los_samples = int(los_samples)

        self._lock = threading.Lock()
        self._pts = None            # (N,3) ROS xyz, finite
        self._pose = None           # (a0, a1) sensor world position
        self._stamp = 0.0
        self._dist = None           # (n,n) static distance field [m]
        self._dist_unknown = None   # (n,n) distance-to-nearest-UNKNOWN (frontier) field [m]
        self._ready = False
        self._node = self._thread = None
        # visibility lookup (ego-relative candidate offsets → hidden fraction)
        self._vis = None            # (C, C) hidden fraction, or None if no ROI
        self._vis_c = 0             # candidate grid side length
        self._vis_half = self.cand_reach
        self._peek_gain = 0.0       # hidden(here) − best reachable hidden (0 = peeking useless)

    @property
    def peek_gain(self) -> float:
        """How much a reachable position would reduce the hidden fraction vs staying put
        [0..1]. The controller relaxes lane-keeping when this exceeds a threshold, so the
        bounded visibility term can actually pull the ego off the centreline to investigate."""
        return self._peek_gain if (self._ready and self._vis is not None) else 0.0

    # ── ROS side ──────────────────────────────────────────────────────────────

    def start(self, topic: str = "/point_cloud",
              pose_topic: str = "/laser_scan_pose") -> bool:
        if not _HAS_RCLPY:
            print("[LidarCostmap] rclpy not importable — costmap disabled.")
            return False
        if not rclpy.ok():
            rclpy.init()
        outer = self

        class _Sub(Node):
            def __init__(self):
                super().__init__("mppi_lidar_costmap")
                # Sensor-data QoS (BEST_EFFORT/VOLATILE): the ros_tcp_endpoint bridge commonly
                # publishes streamed sensor topics best-effort. A default (RELIABLE) subscriber
                # is INCOMPATIBLE with a best-effort publisher per the DDS QoS spec — both sides
                # come up cleanly, no error either side, but zero messages ever cross. That
                # silent mismatch is the most common cause of "subscribed OK but ready=False".
                qos = qos_profile_sensor_data
                self.create_subscription(PointCloud2, topic, self._cloud, qos)
                # Unity publishes a bare Pose; accept PoseStamped too.
                self.create_subscription(Pose, pose_topic, self._pose_cb, qos)
                self._got_cloud = self._got_pose = False   # one-shot confirmation logs

            def _cloud(self, msg):
                pts = outer._parse_cloud(msg)
                if pts is not None:
                    with outer._lock:
                        outer._pts = pts
                        outer._stamp = time.monotonic()
                    if not self._got_cloud:
                        self._got_cloud = True
                        print(f"[LidarCostmap] first /point_cloud message received "
                              f"({len(pts)} valid pts)")

            def _pose_cb(self, msg):
                p = msg.position
                with outer._lock:
                    outer._pose = (float(p.z), float(p.x))   # world (a0, a1)
                if not self._got_pose:
                    self._got_pose = True
                    print(f"[LidarCostmap] first /laser_scan_pose message received "
                          f"(a0={p.z:.1f}, a1={p.x:.1f})")

        self._node = _Sub()
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()
        print(f"[LidarCostmap] subscribed cloud='{topic}' pose='{pose_topic}'  "
              f"persistent {self.grid.n}×{self.grid.n} @ {self.res} m  "
              f"free_ttl={self.grid.free_ttl}s  EDT={'scipy' if _HAS_SCIPY else 'chamfer'}")
        return True

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

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Per-control-step build ────────────────────────────────────────────────

    def update(self, ego_fwd=None, dyn_positions=None) -> bool:
        """
        Integrate the latest scan into the persistent map and rebuild the static
        distance + visibility fields. ego_fwd: (a0,a1) ego forward unit vector in
        world axes (for the ROI wedge); None → 360° ROI. dyn_positions: ego-
        relative (a0,a1) of known agents, scrubbed so their returns don't become
        static/unknown structure. Returns readiness.
        """
        with self._lock:
            pts, pose, stamp = self._pts, self._pose, self._stamp
        if pts is None or pose is None or (time.monotonic() - stamp) > self.max_age:
            self._ready = False
            return False
        self._ingest(pts, pose, ego_fwd, dyn_positions, now=stamp)
        return self._ready

    def _ingest(self, pts_ros, pose, ego_fwd, dyn_positions, now) -> None:
        ego0, ego1 = pose
        x, y, z = pts_ros[:, 0], pts_ros[:, 1], pts_ros[:, 2]

        # All in-window returns (local a0,a1) — these CARVE free space (incl. ground).
        a0, a1 = x, -y                                     # ROS → (a0, a1) local
        rng = np.hypot(a0, a1)
        inb = (rng > self.min_range) & (np.abs(a0) < self.half) & (np.abs(a1) < self.half)
        ca0, ca1 = a0[inb], a1[inb]                        # carve set

        # Obstacle-height subset of the carve set → OCCUPIED cells.
        obst = inb & (z > self.z_lo) & (z < self.z_hi)
        oa0, oa1 = a0[obst], a1[obst]

        # Scrub dynamic-agent returns from the OCC set (oracle handles those; don't
        # let a moving agent freeze into static structure). They still carve FREE.
        if dyn_positions is not None and len(oa0):
            for p in dyn_positions:
                k = (oa0 - float(p[0])) ** 2 + (oa1 - float(p[1])) ** 2 > self.dyn_mask_radius ** 2
                oa0, oa1 = oa0[k], oa1[k]

        # LOCAL → WORLD (pure translation; axes are world-aligned).
        self.grid.update(ego0, ego1,
                         oa0 + ego0, oa1 + ego1,
                         ca0 + ego0, ca1 + ego1, now)

        # Static distance field from the persistent OCCUPIED cells.
        occ = (self.grid.state == OCC)
        if not occ.any():
            self._dist = np.full((self.grid.n, self.grid.n), 1e6)
        elif _HAS_SCIPY:
            self._dist = distance_transform_edt(~occ) * self.res
        else:
            self._dist = _chamfer_distance(occ, self.res)

        # Frontier distance field: distance from every cell to the nearest OCCLUSION-SHADOW cell,
        # i.e. an UNKNOWN cell that is a genuine blind spot BEHIND an obstacle — NOT the open
        # sensor-range rim. Plain "nearest UNKNOWN" is wrong here: every un-observed cell (beyond
        # sensor range, the lateral/rear gaps, the map's outer border) is UNKNOWN, so in open
        # terrain the nearest UNKNOWN is just the edge of what's been seen (~sensor range), and the
        # sightline cap would fire everywhere. We instead keep only UNKNOWN cells within
        # SHADOW_RADIUS of an OCCUPIED cell (an occluder's shadow); the open rim, far from any
        # object, is excluded, so d_vis is large in the clear and only shrinks near real occlusions.
        unknown = (self.grid.state == UNKNOWN)
        occ_for_shadow = (self.grid.state == OCC)
        if not occ_for_shadow.any():
            shadow = np.zeros_like(unknown)
        elif _HAS_SCIPY:
            r_cells = max(1, int(round(SHADOW_RADIUS / self.res)))
            shadow = unknown & binary_dilation(occ_for_shadow, iterations=r_cells)
        else:
            shadow = unknown & occ_for_shadow  # no scipy: fall back to edge-adjacency only
        if not shadow.any():
            self._dist_unknown = np.full((self.grid.n, self.grid.n), 1e6)
        elif _HAS_SCIPY:
            self._dist_unknown = distance_transform_edt(~shadow) * self.res
        else:
            self._dist_unknown = _chamfer_distance(shadow, self.res)

        # Skip the O(candidates × ROI cells × LOS samples) visibility build entirely when
        # nothing consumes it — it ran unconditionally before, so tuning cand_reach silently
        # changed ingest() latency (and therefore how stale _dist/_dist_unknown are for the
        # ACTIVE static/sightline costs) even with the hidden-fraction term disabled.
        if self.enable_visibility:
            print(f"[DEBUG occ] occ_cells={int(occ.sum())} unknown_cells={int((self.grid.state == UNKNOWN).sum())}")
            self._build_visibility(ego0, ego1, ego_fwd, occ)
        self._ready = True

    # ── Visibility (active perception) ────────────────────────────────────────

    def _build_visibility(self, ego0, ego1, ego_fwd, occ) -> None:
        """
        ROI = UNKNOWN cells in the forward wedge within roi_range (occluded /
        never-seen, path-relevant space). For each candidate ego offset, compute
        the fraction of ROI cells whose line of sight is blocked by an OCCUPIED
        cell → the 'hidden' field the MPPI cost samples. None when ROI is empty
        (→ visibility term contributes nothing, behaviour stays nominal).
        """
        self._peek_gain = 0.0        # no worthwhile peek unless we build a real ROI below
        g = self.grid
        ei, ej = g.ego_cell(ego0, ego1)

        # Candidate ROI cells: UNKNOWN, within range, ahead (wedge).
        unk_i, unk_j = np.where(g.state == UNKNOWN)
        if len(unk_i) == 0:
            self._vis = None
            return
        d0 = (unk_i - ei) * self.res            # offset from ego (a0, a1) [m]
        d1 = (unk_j - ej) * self.res
        rng = np.hypot(d0, d1)
        sel = (rng > 1.0) & (rng < self.roi_range)
        if ego_fwd is not None:
            f = np.asarray(ego_fwd, float); f = f / (np.hypot(*f) or 1.0)
            cos = (d0 * f[0] + d1 * f[1]) / np.maximum(rng, 1e-6)
            sel &= cos > self.roi_cos
        ri, rj = unk_i[sel], unk_j[sel]
        rsel = rng[sel]
        if len(ri) == 0:
            self._vis = None
            return
        roi0 = g._o0 + (ri + 0.5) * self.res     # candidate ROI world (a0, a1)
        roi1 = g._o1 + (rj + 0.5) * self.res

        # Keep ONLY cells that are actually SHADOWED from the current ego — i.e. the
        # LOS ego→cell is blocked by an OCC cell. This is the occlusion region an
        # emerging threat could hide in. It also discards the spurious near-ego
        # UNKNOWN cells left by finite-resolution free-carving (those are in open
        # line of sight, not shadow), which would otherwise dominate the nearest-N.
        blk = self._los_blocked(ego0, ego1, roi0, roi1, occ, g)      # (R,) bool
        roi0, roi1, rsel = roi0[blk], roi1[blk], rsel[blk]
        if len(roi0) == 0:
            self._vis = None
            return
        # Cap to the nearest shadowed cells to bound the candidate×ROI cost.
        if len(roi0) > self.max_roi_cells:
            order = np.argsort(rsel)[:self.max_roi_cells]
            roi0, roi1 = roi0[order], roi1[order]

        # Candidate ego positions: a (C×C) grid of offsets around the ego.
        offs = np.arange(-self.cand_reach, self.cand_reach + 1e-6, self.cand_res)
        C = len(offs)
        self._vis_c = C
        self._vis_half = self.cand_reach
        cand0 = ego0 + offs[:, None] * np.ones(C)[None, :]     # (C,C) world a0
        cand1 = ego1 + np.ones(C)[:, None] * offs[None, :]     # (C,C) world a1
        cand0 = cand0.ravel(); cand1 = cand1.ravel()           # (Q,)

        hidden = self._blocked_fraction(cand0, cand1, roi0, roi1, occ, g)
        self._vis = hidden.reshape(C, C)
        # How much a REACHABLE position beats staying put: hidden(here) − min hidden.
        # >0 means peeking is worthwhile → the controller relaxes lane-keeping so the
        # (bounded [0,1]) visibility term can actually pull the ego off the centreline.
        cc = C // 2                              # centre candidate = current ego (offset 0)
        self._peek_gain = float(self._vis[cc, cc] - self._vis.min())

    def _los_blocked(self, p0, p1, r0, r1, occ, g) -> np.ndarray:
        """Per-ROI-cell: is the segment from the single point (p0,p1) to (r0,r1)
        blocked by an OCC cell? Returns (R,) bool."""
        S = self.los_samples
        fr = np.linspace(0.0, 1.0, S)
        s0 = p0 + fr[None, :] * (r0[:, None] - p0)          # (R, S)
        s1 = p1 + fr[None, :] * (r1[:, None] - p1)
        ci = np.floor((s0 - g._o0) / self.res).astype(int)
        cj = np.floor((s1 - g._o1) / self.res).astype(int)
        inb = (ci >= 0) & (ci < g.n) & (cj >= 0) & (cj < g.n)
        ci = np.clip(ci, 0, g.n - 1); cj = np.clip(cj, 0, g.n - 1)
        return (occ[ci, cj] & inb).any(axis=1)

    def _blocked_fraction(self, c0, c1, r0, r1, occ, g) -> np.ndarray:
        """
        For each candidate (c0,c1), fraction of ROI points (r0,r1) whose segment
        crosses an OCCUPIED cell. Vectorised over candidates × ROI × LOS samples.
        """
        Q, R, S = len(c0), len(r0), self.los_samples
        fr = np.linspace(0.0, 1.0, S)                          # include both ends
        # sample coords: (Q, R, S)
        s0 = c0[:, None, None] + fr[None, None, :] * (r0[None, :, None] - c0[:, None, None])
        s1 = c1[:, None, None] + fr[None, None, :] * (r1[None, :, None] - c1[:, None, None])
        ci = np.floor((s0 - g._o0) / self.res).astype(int)
        cj = np.floor((s1 - g._o1) / self.res).astype(int)
        inb = (ci >= 0) & (ci < g.n) & (cj >= 0) & (cj < g.n)
        ci = np.clip(ci, 0, g.n - 1); cj = np.clip(cj, 0, g.n - 1)
        hit = occ[ci, cj] & inb                                # (Q,R,S) occupied sample
        blocked = hit.any(axis=2)                              # (Q,R) LOS broken
        return blocked.mean(axis=1)                            # (Q,) fraction hidden

    # ── Planner-side lookups ──────────────────────────────────────────────────

    def distance(self, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
        """
        (N,) ABSOLUTE world (a0,a1) → (N,) distance to nearest static surface [m]. Uses the
        grid's own world_to_cell (its tracked origin), NOT an ego-relative-delta + assumed
        centre — the grid's centre is the last SENSOR pose, which is not generally the same
        point as the caller's ego reference (e.g. a laser_link mounted away from the vehicle
        pivot), so that shortcut silently samples the wrong cell.
        """
        w0 = np.asarray(w0); w1 = np.asarray(w1)
        if not self._ready or self._dist is None:
            return np.full(len(w0), 1e6)
        g = self.grid
        i, j = g.world_to_cell(w0, w1)
        inside = (i >= 0) & (i < g.n) & (j >= 0) & (j < g.n)
        out = np.full(len(w0), 1e6)
        out[inside] = self._dist[i[inside], j[inside]]
        return out

    def distance_to_unknown(self, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
        """
        (N,) ABSOLUTE world (a0,a1) → (N,) distance to the nearest UNKNOWN cell [m], i.e. the
        distance to the FREE↔UNKNOWN frontier (blind-corner edge) for a query point in free
        space. Used by the virtual-obstacle / forward-reachable-set cost: a phantom agent is
        assumed to sit on that frontier, so this is the distance to where the bubble originates.
        Not-ready / out-of-window points return 1e6 (no nearby frontier ⇒ no virtual penalty).
        """
        w0 = np.asarray(w0); w1 = np.asarray(w1)
        if not self._ready or self._dist_unknown is None:
            return np.full(len(w0), 1e6)
        g = self.grid
        i, j = g.world_to_cell(w0, w1)
        inside = (i >= 0) & (i < g.n) & (j >= 0) & (j < g.n)
        out = np.full(len(w0), 1e6)
        out[inside] = self._dist_unknown[i[inside], j[inside]]
        return out

    def occupancy(self, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
        """
        G(p_WB): (N,) ABSOLUTE world (a0,a1) → (N,) three-state occupancy in the PAPER's
        convention {occupied: 1, free: 0, unknown: -1}. Points outside the grid window (or
        before the map is ready) are treated as free (0) so they carry no collision penalty.
        Uses the grid's own world_to_cell — see distance() docstring for why NOT ego-relative.
        """
        w0 = np.asarray(w0); w1 = np.asarray(w1)
        out = np.zeros(len(w0), dtype=np.int8)      # default: free (0)
        if not self._ready:
            return out
        g = self.grid
        i, j = g.world_to_cell(w0, w1)
        inside = (i >= 0) & (i < g.n) & (j >= 0) & (j < g.n)
        st = g.state[i[inside], j[inside]]            # internal {UNKNOWN:0, FREE:1, OCC:2}
        paper = np.where(st == OCC, 1, np.where(st == FREE, 0, -1)).astype(np.int8)
        out[inside] = paper
        return out

    def not_free(self, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
        """
        Collision indicator 1{G(p_WB) != 0}: (N,) bool, True where the cell is occupied OR
        unknown (i.e. NOT known-free). Out-of-window / not-ready points are free ⇒ False.
        w0/w1: ABSOLUTE world (a0,a1) — see occupancy()/distance() docstrings.
        """
        return self.occupancy(w0, w1) != 0

    def hidden_fraction(self, pts: np.ndarray) -> np.ndarray:
        """
        (N,2) ego-relative (Δa0,Δa1) → (N,) fraction of the occluded ROI that
        stays hidden from that position [0..1]. 0 when there is no ROI, or for
        points outside the candidate window (no incentive there).
        """
        if not self._ready or self._vis is None:
            return np.zeros(len(pts))
        C = self._vis_c
        i = np.round((pts[:, 0] + self._vis_half) / self.cand_res).astype(int)
        j = np.round((pts[:, 1] + self._vis_half) / self.cand_res).astype(int)
        inside = (i >= 0) & (i < C) & (j >= 0) & (j < C)
        out = np.zeros(len(pts))
        out[inside] = self._vis[i[inside], j[inside]]
        return out

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
