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
  3. TRACK — a constant-velocity KALMAN FILTER per object, state [p0, p1, v0, v1],
     measured by the cluster centroid. Association is nearest-neighbour in MAHALANOBIS
     distance against the PREDICTED position, gated by both a chi-square test and
     `assoc_radius`. Unlike OcclusionCornerTracker (whose corners are static world
     features, so it has no motion model), here the motion IS the signal, so velocity
     is estimated — and with a filter rather than an EMA the estimate carries a
     covariance, so a track that has been coasting unseen widens its own gate instead
     of being stolen by whatever centroid happens to be nearby.

     WHY A KF IS WORTH IT HERE DESPITE THE MEASUREMENT BEING UGLY. The centroid error
     is dominated by which face of the object is visible, not by zero-mean sensor
     noise, so the covariance is not a faithful uncertainty — it is a tuned gain
     schedule. That is still strictly better than the EMA it replaces, because:
       * the measurement noise SCALES WITH THE CLUSTER'S OWN EXTENT (R = (r_frac*r)^2),
         which is exactly how far the visible-face centroid can wander, so big blobs
         are trusted less than small ones instead of both getting alpha = 0.6;
       * missed scans are PREDICTED THROUGH (x <- Fx, P <- FPF' + Q) instead of frozen,
         so a track occluded for a second comes back where the object actually is;
       * the velocity comes out of the same recursion as the position rather than from
         a finite difference of two noisy centroids, so it is usable at all.
  4. CLASSIFY — FREE SPACE FIRST, centroid displacement only as a fallback.

     THE PRIMARY TEST IS "IS THE PLACE IT USED TO BE NOW EMPTY?" (free_space.py). Each
     track's past position is projected into the current scan's beam grid; if the beams
     reach PAST it, the body vacated that space and the track is dynamic — one scan, no
     window, no confirmation count. If a beam still terminates there, it is static, and
     that verdict VETOES the displacement test below. Only when free space is
     INCONCLUSIVE (the spot is occluded by something else, or outside the FOV, or the
     track is too young to have a probe) does the displacement test decide.

     That ordering is the fix for two failures that the displacement test cannot escape
     by tuning, because they are the same trade-off seen from both ends:
       * it FALSE-POSITIVES on scenery. A long wall re-segments into different fragments
         every scan; each is small (small extent threshold) with a centroid that slides
         metres along the wall. Measured: a terminal face took cl 3 -> 21, dyn -> 19,
         with reported speeds shadowing the EGO's own speed.
       * it DETECTS REAL MOVERS LATE — a closed 2 s window, scan-aligned, plus min_hits.
     Lowering extent_frac trades the second against the first. Free space is not on that
     curve at all: a static body's beams stop on it however far its visible-face centroid
     wanders, so the test does not care about segmentation stability or centroids.

     The fallback test: a track is DYNAMIC when its NET displacement over a window of
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
     threshold on this observable separates them. That is precisely why the free-space
     test above was added and put FIRST; this floor now only applies where free space
     could not resolve the track.

     Free space has a slow-end limit of its own, but a milder and differently-shaped
     one: the body must clear its own footprint before a beam can pass through where it
     was, i.e. roughly r_cluster / fs_probe_age. Crucially, raising fs_probe_age to lower
     that floor costs no false positives (a static body never vacates, however long you
     watch), whereas lowering extent_frac to lower the centroid floor buys false
     positives directly.

The keep-out itself is NOT built here — the planner applies
occlusion_capsules.occlusion_stage_cost() to the tracks this module returns, with

    r_keep(t_k) = d_safe + r_cluster + v_target * (t_k + age)

i.e. the same forward-reachable-set circle the occlusion boundaries get, inflated by
the cluster's own extent and by the measurement age. Age matters: Unity publishes the
cloud at ~1 Hz, so a centroid can be a full second stale, and at v_target = 5 m/s
ignoring that understates the reachable set by ~5 m.

WHY THE CIRCLE IS ISOTROPIC AND NOT SWEPT ALONG THE ESTIMATED VELOCITY. The velocity
estimate is now filtered, but it is still filtered off a centroid that moves whenever
the visible face of the object changes, at ~1 Hz; it is good enough to answer "is this
thing moving?" and to bridge a missed scan, and nowhere near good enough to say where
the object will be in 3 s. Expanding isotropically assumes only a speed BOUND, which is
the assumption the occlusion model already makes and the one this data can support.
A swept capsule additionally needs the HEADING to be trustworthy over the horizon; the
filter's own velocity covariance is the thing to check before believing it (see
`vel_sigma()`), not the fact that a KF is now in the loop.

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

    A constant-velocity Kalman filter per track (state [p0, p1, v0, v1]) with
    gated nearest-neighbour association against the PREDICTED position.

    The measurement is the centroid of a partially-observed surface at ~1 Hz, so its
    error is NOT zero-mean sensor noise — it is dominated by which face of the object
    is visible. The filter is therefore tuned as a gain schedule rather than believed
    as an uncertainty: R scales with the cluster's own extent (`r_frac`), which is the
    scale on which that face-drift actually happens, and Q is a white-noise-acceleration
    model whose `q_accel` is set from how hard the agents you simulate can manoeuvre.
    The parts that are genuinely principled — predicting through missed scans, and a
    velocity that comes out of the recursion instead of a two-scan difference — are the
    reason this beats the EMA it replaced.

    assoc_radius is a HARD outer gate on top of the chi-square test, and must exceed the
    per-scan travel of the fastest agent you care about (v_target * scan_period, ~5 m at
    1 Hz) or a moving cluster is re-created as a brand-new track every scan and never
    accumulates the hits to be called dynamic. The chi-square gate can only tighten it.
    """

    # 2-DOF chi-square at ~99%: the association gate on the innovation, in units of its
    # own covariance S. A track with a stale, uncertain prediction has a large S and so
    # a physically WIDE gate; a freshly-updated one is held to a tight one.
    GATE_CHI2 = 9.21

    def __init__(self, assoc_radius: float = 8.0, ttl: float = 3.0, min_hits: int = 2,
                 v_min: float = 1.0, min_dyn_hits: int = 2,
                 require_motion: bool = True, dyn_window: float = 2.0,
                 extent_frac: float = 1.2, resegment_ratio: float = 1.5,
                 q_accel: float = 2.0, r_frac: float = 0.5, r_min: float = 0.5,
                 sigma_v0: float = 5.0, extent_alpha: float = 0.5,
                 fs_probe_age: float = 2.5, fs_hold: float = 3.0,
                 fs_trust: float = 3.0):
        # ── Free-space (ray-through) evidence — see free_space.py ─────────────
        self.fs_probe_age = float(fs_probe_age)  # [s] how far back the probed position may be.
                                                 #   The OLDEST position inside this window is
                                                 #   probed, because the older it is the further
                                                 #   the body has moved off it and the sooner the
                                                 #   spot reads empty. This is what sets the SLOW
                                                 #   end of detection: a body of radius r must
                                                 #   clear its own footprint, i.e. ~r/fs_probe_age
                                                 #   m/s. Unlike the centroid test's floor, though,
                                                 #   raising this costs no false positives.
        self.fs_hold = float(fs_hold)      # [s] how long a MOVED verdict keeps a track dynamic
                                           #   without fresh confirmation. Covers the object
                                           #   being briefly occluded mid-manoeuvre.
        self.fs_trust = float(fs_trust)    # [s] how long a CONCLUSIVE verdict overrides the
                                           #   centroid test. Inside this, free space decides and
                                           #   displacement is ignored — that is the whole point:
                                           #   the displacement test is what false-positives on
                                           #   re-segmenting scenery, so an OCCUPIED verdict must
                                           #   be able to veto it, not merely fail to agree.
        self.assoc_radius = float(assoc_radius)
        self.q_accel = float(q_accel)      # [m/s^2] 1-sigma process accel. Sets how fast the
                                           #   filter is allowed to change its mind: too low and
                                           #   the keep-out lags a manoeuvring agent, too high
                                           #   and the centroid's face-drift is tracked as motion.
        self.r_frac = float(r_frac)        # measurement sigma as a fraction of the cluster
                                           #   radius — the scale the visible-face centroid
                                           #   wanders on, so big blobs are trusted less
        self.r_min = float(r_min)          # [m] floor on that sigma, so a tiny cluster is not
                                           #   trusted absolutely (R -> 0 makes K -> 1)
        self.sigma_v0 = float(sigma_v0)    # [m/s] initial velocity sigma on a new track. Set it
                                           #   near the top speed you expect: it is what lets the
                                           #   filter accept the first real motion instead of
                                           #   damping it toward the zero it was born with.
        self.extent_alpha = float(extent_alpha)  # EMA on the cluster RADIUS. Stays an EMA: it is
                                                 #   a shape statistic, not part of the motion
                                                 #   state, and nothing propagates it.
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
        self._now = 0.0        # time of the latest predict/update; _confirmed reads it so
                               # every accessor judges the same instant

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0
        self._now = 0.0

    # ── Kalman filter primitives ─────────────────────────────────────────────
    #
    # State x = [p0, p1, v0, v1], measurement z = centroid. The two axes are modelled
    # independently and identically (no preferred direction on an apron), which keeps
    # every matrix 4x4 and every inverse a 2x2 done in closed form.

    def _predict(self, tr: dict, now: float) -> None:
        """Advance a track's state to `now`. Idempotent in the sense that it always
        propagates from tr['t'] and then sets tr['t'] = now, so calling it once per
        update() per track — associated or not — is what keeps coasting tracks honest."""
        dt = now - tr["t"]
        if dt <= 1e-6:
            return
        x, P = tr["x"], tr["P"]
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # Continuous white-noise acceleration, discretised. The velocity block grows
        # linearly in dt and the position block cubically, which is what makes a track
        # unseen for a second widen its own association gate rather than drift silently.
        q = self.q_accel ** 2
        Q = np.zeros((4, 4))
        for i in (0, 1):
            j = i + 2
            Q[i, i] = q * dt ** 3 / 3.0
            Q[i, j] = Q[j, i] = q * dt ** 2 / 2.0
            Q[j, j] = q * dt
        tr["x"] = F @ x
        tr["P"] = F @ P @ F.T + Q
        tr["t"] = now

    def _r_meas(self, r_cluster: float) -> float:
        """Measurement variance for a centroid of a cluster of this radius."""
        return max(self.r_min, self.r_frac * float(r_cluster)) ** 2

    def _gate(self, tr: dict, meas: np.ndarray, r_var: float):
        """(mahalanobis^2, euclidean) of a measurement against a track's prediction, or
        None if it fails either gate. Both gates matter: the chi-square one is the
        statistically right test but is only as good as the tuned covariance, so
        assoc_radius stays as a hard physical backstop against a diverged P swallowing
        a neighbouring object."""
        innov = meas - tr["x"][:2]
        d = float(np.hypot(*innov))
        if d > self.assoc_radius:
            return None
        S = tr["P"][:2, :2] + r_var * np.eye(2)
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if det <= 1e-12:
            return None
        Sinv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        m2 = float(innov @ Sinv @ innov)
        return (m2, d) if m2 <= self.GATE_CHI2 else None

    def _correct(self, tr: dict, meas: np.ndarray, r_var: float) -> None:
        x, P = tr["x"], tr["P"]
        S = P[:2, :2] + r_var * np.eye(2)
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        Sinv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        K = P[:, :2] @ Sinv                                  # (4,2)
        tr["x"] = x + K @ (meas - x[:2])
        # Joseph form: it stays symmetric positive-definite under the repeated updates
        # and long coasts this tracker does, where the short form (I-KH)P silently loses
        # symmetry and the gate stops meaning anything.
        IKH = np.eye(4)
        IKH[:, :2] -= K
        R = r_var * np.eye(2)
        tr["P"] = IKH @ P @ IKH.T + K @ R @ K.T

    def _new_track(self, meas: np.ndarray, r: float, now: float) -> dict:
        P = np.diag([self._r_meas(r), self._r_meas(r),
                     self.sigma_v0 ** 2, self.sigma_v0 ** 2])
        return {"id": self._next_id,
                "x": np.array([meas[0], meas[1], 0.0, 0.0]), "P": P, "t": now,
                "r": float(r),
                "anchor": meas.copy(), "anchor_t": now, "anchor_r": float(r),
                "last_seen": now, "hits": 1, "dyn_hits": 0,
                # Free-space state. `hist` is the trail of past positions the ray-through
                # test probes; fs_t stamps the last CONCLUSIVE verdict (either way), which
                # is what gates whether free space or the centroid test decides.
                # Empty: the seed position has not been verified occupied yet. update()
                # appends it at the end of this scan only if the range image agrees.
                "hist": [],
                "fs_state": 0, "fs_t": -1e9, "fs_moved_t": -1e9}

    def predict_to(self, now: float) -> None:
        """Advance every track to `now` and expire the stale ones, WITHOUT fusing any
        measurement. Safe — and intended — to call at the full control rate.

        This is the half of the filter that is cheap and always correct to run: it is
        pure propagation of the model, so calling it 20 times between scans gives exactly
        the same state as calling it once (the dt's compose). Correction is NOT like that,
        which is why it lives in update() and must be driven once per scan.
        """
        self._now = now
        for tr in self._tracks:
            self._predict(tr, now)
        self._tracks = [t for t in self._tracks if (now - t["last_seen"]) <= self.ttl]

    def _free_space_pass(self, checker, now: float) -> None:
        """Ask the range image whether each track's PAST position is now empty.

        PROBES THE TRAIL, NOT THE CURRENT ESTIMATE. Probing where the filter thinks the
        object is now would be self-defeating — for a genuine mover that spot is occupied
        by the mover itself, so it would answer OCCUPIED forever. The question is only
        meaningful about a place the object has had TIME TO LEAVE, so the oldest position
        within fs_probe_age is used: the further back it is, the further the body has
        translated off it, and the cleaner the verdict.
        """
        if checker is None or not getattr(checker, "ready", False):
            return
        from free_space import MOVED, OCCUPIED, INCONCLUSIVE
        for tr in self._tracks:
            hist = tr["hist"]
            # Oldest sample still inside the window; None if the track is too young to
            # have a probe that means anything yet.
            probe = next(((t, p, r) for (t, p, r) in hist
                          if (now - t) >= 1e-3 and (now - t) <= self.fs_probe_age), None)
            if probe is None:
                continue
            v = checker.verdict(probe[1], probe[2])
            if v == INCONCLUSIVE:
                continue                    # unobserved: keep whatever we believed

            if v == OCCUPIED:
                # AN "OCCUPIED" READING IS ONLY INFORMATIVE IF THE BODY SHOULD HAVE
                # CLEARED THE PROBE BY NOW. Free space has a floor: an object must
                # translate past its own footprint before any beam can pass through
                # where it was, so a genuinely-moving but SLOW body still reads OCCUPIED.
                # Letting that veto the displacement test made slow movers detected LATER
                # than under the old detector (measured: a 3 m/s mover went from scan 2 to
                # scan 8) — the veto was suppressing the one test that could still see it.
                #
                # So veto only on a CONTRADICTION: the filter claims the body travelled
                # far enough to have vacated the probe, and the beams say it is still
                # there. That is the scenery signature exactly — a re-segmenting wall
                # fragment reports a large bogus displacement while never actually moving.
                # Below that distance the two tests do not disagree, so free space stays
                # silent and the displacement test decides on its own.
                disp = float(np.hypot(*(tr["x"][:2] - probe[1])))
                if disp <= probe[2] + getattr(checker, "slack", 2.0):
                    continue
            tr["fs_state"] = v
            tr["fs_t"] = now
            if v == MOVED:
                tr["fs_moved_t"] = now

    def update(self, clusters: Optional[np.ndarray], now: float, checker=None) -> None:
        """Fold ONE SCAN's clusters into the tracks. clusters: (M,4) from cluster_points.

        CALL THIS ONCE PER SCAN, NOT ONCE PER CONTROL STEP. The cloud arrives at ~1 Hz
        while the control loop runs at ~20 Hz, and ObstacleCircles.clusters() is memoised
        on the scan stamp — so a naive call per step re-fuses the SAME centroid ~20 times.
        A Kalman filter treats those as 20 independent observations: P collapses by ~20x,
        and every one of those corrections drags the state back toward a position that is
        by then stale, which pulls the velocity estimate toward zero exactly when the
        object is moving. The over-tight P then narrows the chi-square gate and the next
        genuine measurement is likelier to be rejected. Drive this off
        ObstacleCircles.stamp changing, and call predict_to() on the steps in between.

        PREDICT EVERY TRACK FIRST, associated or not — that is the step the EMA version
        could not do. A track missed this scan still advances along its estimated
        velocity with a growing P, so it is where the object is (not where it was last
        seen) and it is available for association again next scan.
        """
        # Free space FIRST, against the trail — before this scan's measurement is fused,
        # so the verdict is about the scan as it arrived rather than about a state the
        # same scan has already moved.
        self._free_space_pass(checker, now)

        self.predict_to(now)

        if clusters is not None and len(clusters):
            clusters = np.asarray(clusters, float).reshape(-1, 4)
            # Measurements are matched best-first: a greedy pass in order of how well the
            # cluster fits SOME track, rather than in arbitrary row order, so a confident
            # match cannot be pre-empted by an earlier marginal one claiming the track.
            cands = []
            for mi, (c0, c1, r, _n) in enumerate(clusters):
                meas = np.array([c0, c1])
                r_var = self._r_meas(r)
                for ti, tr in enumerate(self._tracks):
                    g = self._gate(tr, meas, r_var)
                    if g is not None:
                        cands.append((g[0], mi, ti))
            cands.sort(key=lambda c: c[0])

            matched_m, matched_t = {}, set()
            for _m2, mi, ti in cands:
                if mi in matched_m or ti in matched_t:
                    continue
                matched_m[mi] = ti
                matched_t.add(ti)

            for mi, (c0, c1, r, _n) in enumerate(clusters):
                meas = np.array([c0, c1])
                ti = matched_m.get(mi)
                if ti is None:
                    self._tracks.append(self._new_track(meas, float(r), now))
                    self._next_id += 1
                    continue
                tr = self._tracks[ti]
                self._correct(tr, meas, self._r_meas(r))
                a = self.extent_alpha
                tr["r"] = (1.0 - a) * tr["r"] + a * float(r)
                tr["last_seen"] = now
                tr["hits"] += 1
                self._classify(tr, now)

        # Extend each track's trail with where it now believes it is, and forget samples
        # older than the probe window (plus a scan of slack, so the oldest usable sample
        # is never dropped by the same call that would have probed it).
        #
        # ONLY VERIFIED-OCCUPIED POSITIONS BECOME PROBES. A probe is only meaningful if
        # the body was actually THERE when the sample was taken, and a centroid is not a
        # guarantee of that: when the clusterer splits an object (or merges two), the
        # centroid of the resulting component can land in the gap BETWEEN the faces —
        # empty space. Probing that later reads "vacated" and manufactures a MOVED
        # verdict on perfectly static scenery. Confirming the spot is occupied in the
        # scan the sample comes from removes that class of false positive entirely,
        # at the cost of one extra range-image query per track per scan.
        fs_ok = checker is not None and getattr(checker, "ready", False)
        for tr in self._tracks:
            if not fs_ok:
                tr["hist"].append((now, tr["x"][:2].copy(), tr["r"]))
            else:
                from free_space import OCCUPIED
                if checker.verdict(tr["x"][:2], tr["r"]) == OCCUPIED:
                    tr["hist"].append((now, tr["x"][:2].copy(), tr["r"]))
            cutoff = now - (self.fs_probe_age + 1.5)
            tr["hist"] = [h for h in tr["hist"] if h[0] >= cutoff]

        self._tracks = [t for t in self._tracks if (now - t["last_seen"]) <= self.ttl]

    def _classify(self, tr: dict, now: float) -> None:
        """Static/dynamic decision for one track, on a closed dyn_window.

        Runs only when the window is full, then re-anchors — so the quantity tested is NET
        displacement over the window, not a sum of per-scan steps. That distinction is the
        whole point: centroid jitter cancels over a window while real translation
        accumulates, so a body that wobbles 1 m either way every scan nets ~0 and stays
        static however large the individual steps were.

        DELIBERATELY NOT A TEST ON THE FILTER'S VELOCITY, even though there now is one.
        |v| > v_min is exactly the instantaneous-speed test this module's header explains
        does not work: a parked vehicle whose visible face slides along its body as the
        ego passes produces a real, sustained, filtered velocity. The filter smooths that
        signal; it does not remove it, because the drift is not zero-mean. Only the
        displacement-against-own-extent test separates the two cases, so the KF's job here
        is to make the POSITIONS the test runs on cleaner (and to survive missed scans),
        not to supply a new decision variable.
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
            net = float(np.hypot(*(tr["x"][:2] - tr["anchor"])))
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

        tr["anchor"] = tr["x"][:2].copy()
        tr["anchor_t"] = now
        tr["anchor_r"] = tr["r"]

    def _confirmed(self, tr: dict) -> bool:
        """Is this track usable as a dynamic obstacle? ONE definition, so dynamic(), ids()
        and speeds() cannot disagree about which rows they are describing.

        A THREE-LEVEL DECISION, free space first:

          1. A MOVED verdict within fs_hold  -> DYNAMIC. One ray through the space the
             body used to fill is proof it left; no window, no confirmation count, so
             this is the low-latency path.
          2. Any CONCLUSIVE verdict within fs_trust -> that verdict DECIDES, which means
             an OCCUPIED reading actively VETOES the centroid test. This is the direction
             that matters for false positives: a re-segmenting wall fragment produces a
             large bogus centroid displacement every scan, and the only thing that stops
             it becoming a keep-out is free space being allowed to overrule it.
          3. Otherwise (the object is unobserved, or too young to probe) fall back to the
             displacement test, which is all the information there is in that case.

        dyn_hits is a COUNTDOWN, not a tally: _classify sets it to min_dyn_hits the moment
        a window shows motion and decrements it on each still window, so any value above
        zero means "seen moving within the last min_dyn_hits windows".
        """
        if tr["hits"] < self.min_hits:
            return False
        if not self.require_motion:
            return True
        now = self._now
        if (now - tr["fs_moved_t"]) <= self.fs_hold:
            return True
        if (now - tr["fs_t"]) <= self.fs_trust:
            return False        # conclusive and NOT moved -> static, whatever the centroid did
        return tr["dyn_hits"] > 0

    def dynamic(self, now: float) -> Optional[np.ndarray]:
        """(K,4) [c0, c1, r_cluster, age_s] for confirmed MOVING tracks, or None.

        The position is the FILTERED ESTIMATE PREDICTED FORWARD TO `now`, so a track that
        missed the last scan is reported where its velocity says it should be rather than
        frozen at its last sighting.

        age is still seconds since the track was last MEASURED, and the planner still adds
        v_target*age to the keep-out radius. That is not double-counting the prediction:
        the extrapolation moves the circle's CENTRE along the best guess, while the age
        term widens it to cover the guess being wrong — which is exactly the failure mode
        of extrapolating a centroid whose velocity came from a 1 Hz sensor.
        """
        out = []
        for t in self._tracks:
            if not self._confirmed(t):
                continue
            p = self._pos_at(t, now)
            out.append((p[0], p[1], t["r"], max(0.0, now - t["last_seen"])))
        return np.array(out, dtype=float) if out else None

    def _pos_at(self, tr: dict, now: float) -> np.ndarray:
        """Track position extrapolated to `now` without mutating the filter."""
        dt = max(0.0, now - tr["t"])
        return tr["x"][:2] + tr["x"][2:] * dt

    def ids(self) -> List[int]:
        """Track ids aligned with the rows dynamic() returns (same filter, same order)."""
        return [t["id"] for t in self._tracks if self._confirmed(t)]

    def speeds(self) -> List[float]:
        """Filtered speeds [m/s] aligned with dynamic()'s rows — debug/plot only."""
        return [float(np.hypot(*t["x"][2:])) for t in self._tracks if self._confirmed(t)]

    def velocities(self) -> List[Tuple[float, float]]:
        """Filtered velocity vectors [m/s] aligned with dynamic()'s rows."""
        return [(float(t["x"][2]), float(t["x"][3]))
                for t in self._tracks if self._confirmed(t)]

    def vel_sigma(self) -> List[float]:
        """Per-track velocity 1-sigma [m/s] (the larger of the two axes), aligned with
        dynamic()'s rows. CHECK THIS BEFORE BELIEVING A HEADING: a swept-capsule keep-out
        is only defensible where sigma is small against the speed itself. Freshly-created
        tracks carry sigma_v0 here by construction and mean nothing yet."""
        return [float(np.sqrt(max(t["P"][2, 2], t["P"][3, 3])))
                for t in self._tracks if self._confirmed(t)]

    def fs_counts(self) -> Tuple[int, int, int]:
        """(moved, occupied, inconclusive) over ALL live tracks at the last scan — the
        health check for the free-space detector. `moved + occupied` near zero means the
        range image is not resolving anything (wrong scan geometry, or every track beyond
        the beams' reach), and the classification has silently fallen back to the centroid
        test everywhere."""
        now = self._now
        moved = occ = 0
        for t in self._tracks:
            if t["fs_t"] < 0 or (now - t["fs_t"]) > self.fs_trust:
                continue
            if t["fs_state"] > 0:
                moved += 1
            elif t["fs_state"] < 0:
                occ += 1
        return moved, occ, max(0, len(self._tracks) - moved - occ)

    def fs_labels(self) -> List[str]:
        """Per-track free-space label aligned with dynamic()'s rows: MOV / OCC / -- ,
        where -- means no conclusive verdict is in force and the centroid test decided."""
        now = self._now
        out = []
        for t in self._tracks:
            if not self._confirmed(t):
                continue
            if (now - t["fs_moved_t"]) <= self.fs_hold:
                out.append("MOV")
            elif t["fs_t"] >= 0 and (now - t["fs_t"]) <= self.fs_trust:
                out.append("OCC")
            else:
                out.append("--")
        return out

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
