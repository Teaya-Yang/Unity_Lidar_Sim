"""
dynamic_clusters.py
===================
Sensed dynamic obstacles: LiDAR point clusters given temporal identity, classified
static-vs-moving by their own displacement, and turned into EXPANDING keep-outs of
exactly the same shape as the occlusion ones.

This is the replacement for the removed oracle obstacle slots (exact positions and
velocities handed over by Unity's scenario manager). Everything here is derived from
the point cloud, so the planner never knows more than the sensor model provides.

Pipeline, per scan:

  1. CLUSTER — the height-filtered world points (the same ones obstacle_circles.py
     covers with circles) are binned onto a grid of side `cell` and grouped by
     8-connectivity. Each connected component is one candidate object: centroid,
     covering radius (max centroid-to-point distance, so the disc encloses the whole
     cluster), and point count.
  2. REJECT THE SCENERY — a component wider than `max_radius` is a wall / hangar /
     apron edge, not a vehicle. Aircraft-sized blobs survive; buildings do not. This
     is what keeps the tracker's association problem small and well-posed.
  3. TRACK — nearest-neighbour association to live tracks within `assoc_radius`, with
     an EMA on position AND on the finite-difference velocity. Unlike
     OcclusionCornerTracker (whose corners are static world features, so it has no
     motion model), here the motion IS the signal, so velocity is estimated.
  4. CLASSIFY — a track is DYNAMIC when its NET displacement over a window of
     `dyn_window` seconds exceeds both `v_min * window` and a fraction of the cluster's
     own extent.

     THE EXTENT TERM IS THE IMPORTANT ONE, and instantaneous speed alone is not enough.
     A parked vehicle's centroid is the centroid of its VISIBLE FACE, and that face
     changes as the ego drives past it: rear face, then rear quarter, then flank. The
     centroid migrates across the body and the finite-difference speed reads well above
     v_min for as long as the pass lasts, so a parked ambulance is reported as moving.
     What separates the two cases is that a static body's apparent centroid can only
     ever wander WITHIN ITS OWN FOOTPRINT, however fast it appears to move, whereas a
     genuine mover translates its whole footprint and leaves it. So the test is net
     displacement against the body's own size, not speed against a speed.

     The cost of that is a DETECTION FLOOR: the slowest reliably detectable speed is
     about extent_frac * r_cluster / dyn_window (~2 m/s for a 3.5 m body, ~3 m/s for a
     5 m one, at the defaults). Slower movers are classified static. This is a real
     limit of centroid-only tracking rather than a tuning failure — over one window a
     slow mover and a drifting visible face produce the same centroid signal, so no
     threshold on this observable separates them. Breaking the trade-off requires
     free-space reasoning (a cluster is dynamic if its PREVIOUS position is now
     observed THROUGH, i.e. a beam passes where it used to be), which needs the
     ordered cloud rather than just centroids.

The keep-out itself is NOT built here — the planner applies
occlusion_capsules.occlusion_stage_cost() to the tracks this module returns, with

    r_keep(t_k) = d_safe + r_cluster + v_target * (t_k + age)

i.e. the same forward-reachable-set circle the occlusion boundaries get, inflated by
the cluster's own extent and by the measurement age. Age matters: Unity publishes the
cloud at ~1 Hz, so a centroid can be a full second stale, and at v_target = 5 m/s
ignoring that understates the reachable set by ~5 m.

WHY THE CIRCLE IS ISOTROPIC AND NOT SWEPT ALONG THE ESTIMATED VELOCITY. The velocity
estimate is a two-scan finite difference at ~1 Hz off a centroid that moves whenever
the visible face of the object changes; it is good enough to answer "is this thing
moving?" and nowhere near good enough to say where it will be in 3 s. Expanding
isotropically assumes only a speed BOUND, which is the assumption the occlusion model
already makes and the one this data can support. Swap it for a swept capsule only
after the velocity estimate is filtered properly (a real CV/CA Kalman filter).

FRAMES: everything is controller world (a0, a1) = (Unity Z, Unity X), the same frame
obstacle_circles.py emits, so no conversion happens in this module.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_points(w0: np.ndarray, w1: np.ndarray, *,
                   cell: float = 2.0, min_points: int = 4,
                   max_radius: float = 25.0) -> Optional[np.ndarray]:
    """Grid connected-component clustering of world points.

    w0, w1     : (N,) world coordinates [m]
    cell       : grid side [m]. Two points land in the same cluster if their cells
                 touch (8-connectivity), so this doubles as the merge distance —
                 anything closer than ~cell*sqrt(2) is one object.
    min_points : drop components thinner than this (single stray returns)
    max_radius : drop components WIDER than this — scenery, not a vehicle

    Returns (M,4) [c0, c1, r_cluster, n_points], or None if nothing survives.
    r_cluster is the max centroid-to-point distance, so the disc of that radius
    around the centroid provably covers the cluster.
    """
    w0 = np.asarray(w0, float).ravel()
    w1 = np.asarray(w1, float).ravel()
    if w0.size == 0:
        return None

    ci = np.floor(w0 / cell).astype(np.int64)
    cj = np.floor(w1 / cell).astype(np.int64)

    # Unique occupied cells, and each point's index into that list.
    cells, inv = np.unique(np.stack([ci, cj], axis=1), axis=0, return_inverse=True)
    n_cells = len(cells)
    lookup = {(int(a), int(b)): idx for idx, (a, b) in enumerate(cells)}

    # Flood-fill over occupied cells (8-connected). n_cells is small — the points are
    # already voxel-scale sparse — so an explicit stack beats any labelling library.
    label = np.full(n_cells, -1, dtype=np.int64)
    n_lab = 0
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for seed in range(n_cells):
        if label[seed] >= 0:
            continue
        stack = [seed]
        label[seed] = n_lab
        while stack:
            cur = stack.pop()
            a, b = int(cells[cur, 0]), int(cells[cur, 1])
            for da, db in neigh:
                nb = lookup.get((a + da, b + db))
                if nb is not None and label[nb] < 0:
                    label[nb] = n_lab
                    stack.append(nb)
        n_lab += 1

    pt_label = label[inv]                       # (N,) cluster id per point
    out: List[Tuple[float, float, float, int]] = []
    for lab in range(n_lab):
        m = pt_label == lab
        n = int(m.sum())
        if n < min_points:
            continue
        px, py = w0[m], w1[m]
        c0, c1 = float(px.mean()), float(py.mean())
        r = float(np.max(np.hypot(px - c0, py - c1)))
        # Half a cell of slack: the cluster is a SAMPLING of the object's visible
        # face, and the true surface extends up to the grid resolution beyond it.
        r += 0.5 * cell
        if r > max_radius:
            continue                            # scenery (wall / building / apron edge)
        out.append((c0, c1, r, n))

    return np.array(out, dtype=float) if out else None


# ── Tracking + static/dynamic classification ─────────────────────────────────

class DynamicClusterTracker:
    """Temporal identity + velocity for LiDAR clusters; flags the moving ones.

    Deliberately a nearest-neighbour tracker with an EMA velocity, not a Kalman
    filter. The measurement is a centroid of a partially-observed surface at ~1 Hz;
    its error is dominated by which face of the object is visible, not by zero-mean
    sensor noise, so the covariance a KF would propagate would be fiction. What this
    DOES need over the occlusion tracker is a velocity estimate at all, since the
    static/dynamic decision is the whole point.

    assoc_radius must exceed the per-scan travel of the fastest agent you care about
    (v_target * scan_period, ~5 m at 1 Hz) or a moving cluster is re-created as a
    brand-new track every scan and never accumulates the hits to be called dynamic.
    """

    def __init__(self, assoc_radius: float = 8.0, alpha: float = 0.6,
                 vel_beta: float = 0.5, ttl: float = 3.0, min_hits: int = 2,
                 v_min: float = 1.0, min_dyn_hits: int = 2,
                 require_motion: bool = True, dyn_window: float = 2.0,
                 extent_frac: float = 1.2, resegment_ratio: float = 1.5):
        self.assoc_radius = float(assoc_radius)
        self.alpha = float(alpha)          # EMA weight on the new POSITION measurement.
                                           #   High (0.6) on purpose: a laggy position on a
                                           #   moving target is a keep-out centred behind it.
        self.vel_beta = float(vel_beta)    # EMA weight on the new VELOCITY measurement
        self.ttl = float(ttl)
        self.min_hits = int(min_hits)
        self.v_min = float(v_min)          # [m/s] floor on the net-displacement rate
        self.min_dyn_hits = int(min_dyn_hits)
        self.dyn_window = float(dyn_window)      # [s] window the NET displacement is measured over
        self.extent_frac = float(extent_frac)    # net displacement must also clear this
                                                 #   fraction of the cluster's own radius
        self.resegment_ratio = float(resegment_ratio)  # extent change that means the clusterer
                                                       #   re-segmented, not that the body moved
        self.require_motion = bool(require_motion)  # False ⇒ every compact cluster is
                                                    #   treated as a potential mover
        self._tracks: List[dict] = []
        self._next_id = 0

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0

    def update(self, clusters: Optional[np.ndarray], now: float) -> None:
        """Fold this scan's clusters into the tracks. clusters: (M,4) from cluster_points."""
        if clusters is not None and len(clusters):
            clusters = np.asarray(clusters, float).reshape(-1, 4)
            claimed = set()
            for c0, c1, r, n in clusters:
                meas = np.array([c0, c1])
                best, best_d = None, self.assoc_radius
                for ti, tr in enumerate(self._tracks):
                    if ti in claimed:
                        continue
                    d = float(np.hypot(*(tr["pos"] - meas)))
                    if d < best_d:
                        best, best_d = ti, d
                if best is None:
                    self._tracks.append({"id": self._next_id, "pos": meas.copy(),
                                         "meas": meas.copy(), "meas_t": now,
                                         "vel": np.zeros(2), "r": float(r),
                                         "anchor": meas.copy(), "anchor_t": now,
                                         "anchor_r": float(r),
                                         "last_seen": now, "hits": 1, "dyn_hits": 0})
                    self._next_id += 1
                    claimed.add(len(self._tracks) - 1)
                    continue

                tr = self._tracks[best]
                dt = now - tr["meas_t"]
                if dt > 1e-3:
                    # Finite-difference RAW measurement against RAW measurement. Not against
                    # the smoothed position: the EMA lags by ~(1-alpha)/alpha of a scan, and
                    # dividing that lag by dt adds a constant bias to the speed — measured at
                    # +50% on a 4 m/s mover, which would promote a slowly-drifting static
                    # centroid over v_min.
                    v_meas = (meas - tr["meas"]) / dt
                    b = self.vel_beta
                    tr["vel"] = (1.0 - b) * tr["vel"] + b * v_meas
                tr["meas"] = meas.copy()
                tr["meas_t"] = now
                a = self.alpha
                tr["pos"] = (1.0 - a) * tr["pos"] + a * meas
                tr["r"] = (1.0 - a) * tr["r"] + a * float(r)
                tr["last_seen"] = now
                tr["hits"] += 1
                self._classify(tr, now)
                claimed.add(best)

        self._tracks = [t for t in self._tracks if (now - t["last_seen"]) <= self.ttl]

    def _classify(self, tr: dict, now: float) -> None:
        """Static/dynamic decision for one track, on a closed dyn_window.

        Runs only when the window is full, then re-anchors — so the quantity tested is NET
        displacement over the window, not a sum of per-scan steps. That distinction is the
        whole point: centroid jitter cancels over a window while real translation
        accumulates, so a body that wobbles 1 m either way every scan nets ~0 and stays
        static however large the individual steps were.
        """
        win = now - tr["anchor_t"]
        if win < self.dyn_window:
            return

        # The clusterer re-segmented (a merge with a neighbour, or a split): the centroid
        # jumped because the SET OF POINTS changed, not because the body moved. Counting
        # that as displacement is how a parked vehicle next to a wall gets promoted.
        # Re-anchor on the new segmentation and judge nothing this window.
        r_prev = max(tr["anchor_r"], 1e-3)
        r_now = max(tr["r"], 1e-3)
        resegmented = (r_now / r_prev > self.resegment_ratio
                       or r_prev / r_now > self.resegment_ratio)

        if not resegmented:
            net = float(np.hypot(*(tr["pos"] - tr["anchor"])))
            # Two thresholds, both of which must be cleared:
            #   v_min * win     — an absolute speed floor
            #   extent_frac * r — the body must have left its own footprint, which is the
            #                     part a static object physically cannot do no matter how
            #                     its visible face drifts
            thresh = max(self.v_min * win, self.extent_frac * tr["r"])
            moving = net > thresh
            # BOUNDED, and asymmetric in the SAFE direction: one moving window promotes
            # immediately (a mover must not wait to be believed), while demotion costs
            # min_dyn_hits still windows, so a vehicle pausing at a hold-short line is not
            # dropped and instantly re-acquired.
            #
            # The bound is the fix for the reported false positive: dyn_hits used to be
            # incremented once per scan with no cap, so a vehicle that drove for 20 s
            # banked ~20 hits and then, once parked, needed ~20 more scans to decay below
            # the threshold — it stayed flagged as dynamic for 21 s after stopping.
            if moving:
                tr["dyn_hits"] = self.min_dyn_hits
            else:
                tr["dyn_hits"] = max(0, tr["dyn_hits"] - 1)

        tr["anchor"] = tr["pos"].copy()
        tr["anchor_t"] = now
        tr["anchor_r"] = tr["r"]

    def _confirmed(self, tr: dict) -> bool:
        """Is this track usable as a dynamic obstacle? ONE definition, so dynamic(), ids()
        and speeds() cannot disagree about which rows they are describing.

        dyn_hits is a COUNTDOWN, not a tally: _classify sets it to min_dyn_hits the moment
        a window shows motion and decrements it on each still window, so any value above
        zero means "seen moving within the last min_dyn_hits windows".
        """
        if tr["hits"] < self.min_hits:
            return False
        return tr["dyn_hits"] > 0 if self.require_motion else True

    def dynamic(self, now: float) -> Optional[np.ndarray]:
        """(K,4) [c0, c1, r_cluster, age_s] for confirmed MOVING tracks, or None.

        age is seconds since the track was last measured — the planner adds
        v_target*age to the keep-out radius, because the agent has been free to move
        for that long since the centroid was observed.
        """
        out = []
        for t in self._tracks:
            if not self._confirmed(t):
                continue
            out.append((t["pos"][0], t["pos"][1], t["r"], max(0.0, now - t["last_seen"])))
        return np.array(out, dtype=float) if out else None

    def ids(self) -> List[int]:
        """Track ids aligned with the rows dynamic() returns (same filter, same order)."""
        return [t["id"] for t in self._tracks if self._confirmed(t)]

    def speeds(self) -> List[float]:
        """Estimated speeds [m/s] aligned with dynamic()'s rows — debug/plot only."""
        return [float(np.hypot(*t["vel"])) for t in self._tracks if self._confirmed(t)]

    @property
    def n_tracks(self) -> int:
        return len(self._tracks)


# ── Planner-side geometry ─────────────────────────────────────────────────────

class DynamicClusterPublisher:
    """Publishes the keep-outs the planner is constraining against, for the Unity viewer.

    Topic  : /dynamic_clusters   (std_msgs/Float32MultiArray)
    Payload: [v_target, d_safe_hard, grow_horizon] ++ [c0, c1, r_cluster, age] * K

    THE MODEL PARAMETERS TRAVEL WITH THE DATA, deliberately. The existing
    PhantomAgentVisualizer re-declares v_target / d_safe_hard as Inspector fields and its
    docstring warns they MUST be kept equal to config.yaml by hand — a silent mismatch
    there draws a keep-out the controller never used. Sending them in the message removes
    that failure mode entirely: Unity draws the radius the planner actually applied,
    whatever the YAML says.

    Coordinates are controller world (a0, a1). Unity converts: a0 -> Z, a1 -> X.

    A no-op when rclpy is unavailable, matching how ObstacleCircles degrades.
    """

    TOPIC = "/dynamic_clusters"
    N_PARAMS = 3        # leading floats before the cluster rows
    ROW = 4             # floats per cluster row

    def __init__(self, topic: str = TOPIC):
        self.topic = topic
        self._node = None
        self._pub = None

    def start(self) -> bool:
        if not _HAS_RCLPY:
            print("[DynamicClusterPublisher] rclpy not importable — Unity viewer feed disabled.")
            return False
        if not rclpy.ok():
            rclpy.init()
        # No spin thread: a publisher needs none, and ObstacleCircles already owns the
        # executor for this process's subscriptions.
        self._node = Node("dynamic_clusters_viz")
        self._pub = self._node.create_publisher(Float32MultiArray, self.topic, 10)
        print(f"[DynamicClusterPublisher] publishing keep-outs on '{self.topic}'")
        return True

    def publish(self, dyn: Optional[np.ndarray], v_target: float, d_safe: float,
                grow_horizon: float) -> None:
        """Send the CURRENT keep-out set. Call every control step, including with an empty
        set — an empty message is what clears the Unity overlay when the movers are gone;
        skipping the publish would leave the last circles frozen on screen."""
        if self._pub is None:
            return
        rows = (np.zeros((0, self.ROW)) if dyn is None or not len(dyn)
                else np.asarray(dyn, float).reshape(-1, self.ROW))
        msg = Float32MultiArray()
        d0 = MultiArrayDimension(label="params", size=self.N_PARAMS, stride=self.N_PARAMS)
        d1 = MultiArrayDimension(label="clusters", size=len(rows),
                                 stride=self.ROW * max(1, len(rows)))
        msg.layout.dim = [d0, d1]
        msg.data = [float(v_target), float(d_safe), float(grow_horizon)] + \
                   [float(x) for x in rows.ravel()]
        self._pub.publish(msg)

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = self._pub = None


def select_nearest(dyn: Optional[np.ndarray], ego_xy, query_r: float,
                   k_max: int) -> Optional[np.ndarray]:
    """Gate the dynamic set the way the occlusion set is gated: drop anything beyond
    query_r of the ego (measured to the cluster SURFACE, not its centre — a large
    cluster whose centre is out of range can still have its edge on top of the ego),
    then keep the k_max nearest. Returns the surviving (K,4) rows or None."""
    if dyn is None or not len(dyn):
        return None
    dyn = np.asarray(dyn, float).reshape(-1, 4)
    d = np.hypot(dyn[:, 0] - ego_xy[0], dyn[:, 1] - ego_xy[1]) - dyn[:, 2]
    keep = d < query_r
    if not keep.any():
        return None
    sel = dyn[keep]
    order = np.argsort(d[keep])[:max(0, int(k_max))]
    return sel[order] if len(order) else None
