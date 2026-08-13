import math
from typing import Optional, Tuple

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False


def scan_shape(fov_h: float, fov_v: float, res_h: float, res_v: float) -> Tuple[int, int]:
    """Beam-grid dimensions, mirroring LaserSensor3D's constructor arithmetic
    (including its 360-degree wrap-around de-duplication)."""
    n_h = int(math.floor(fov_h / res_h)) + 1
    n_v = int(math.floor(fov_v / res_v)) + 1
    if fov_h == 360:
        n_h -= 1
    return n_h, n_v


def parse_ordered_cloud(msg, n_h: int, n_v: int) -> Optional[np.ndarray]:
    """PointCloud2 -> (n_h, n_v, 3) sensor-frame xyz with non-returns left as NaN.

    Unlike ObstacleCircles._parse_cloud this preserves beam order and keeps invalid
    points, because a dropped beam destroys the adjacency the jump test relies on.
    Returns None if the point count contradicts (n_h, n_v).
    """
    step = msg.point_step
    if step < 12:
        return None
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n = len(raw) // step
    if n == 0 or n != n_h * n_v:
        return None
    xyz = raw[:n * step].reshape(n, step)[:, :12].copy().view(np.float32).reshape(n, 3)
    return xyz.reshape(n_h, n_v, 3)


def _wall_line_deviation(win: np.ndarray, far: np.ndarray):
    """Total-least-squares line through the wall points `win`, tested against `far`.

    win : (w, 2) near-side (wall) beam hit-points in the horizontal plane.
    far : (2,)   the candidate's FAR endpoint.

    Returns (dev, wall_rms):
      dev      — perpendicular distance of `far` from the fitted wall line, or None if
                 the wall points are too coincident to define a direction.
      wall_rms — RMS perpendicular spread of `win` about that line (how clean a line the
                 near side actually is).

    A grazing flat wall has a small wall_rms (the points ARE a line) and a small dev (the
    far endpoint continues that line); a genuine occlusion corner has a large dev because
    the escaped beam has left the wall. The line is fit by the principal eigenvector of
    the 2x2 point covariance (closed form), which — unlike a range extrapolation — is
    exact for a straight surface at any angle of incidence.
    """
    c = win.mean(axis=0)
    d = win - c
    cov = d.T @ d
    a_, b_, dd = cov[0, 0], cov[0, 1], cov[1, 1]
    tr = a_ + dd
    disc = max(tr * tr / 4.0 - (a_ * dd - b_ * b_), 0.0)
    l1 = tr / 2.0 + math.sqrt(disc)                       # larger eigenvalue
    if abs(b_) > 1e-12:
        ex, ey = b_, l1 - a_                              # eigenvector for l1
    else:
        ex, ey = (1.0, 0.0) if a_ >= dd else (0.0, 1.0)   # diagonal cov: axis-aligned
    nrm = math.hypot(ex, ey)
    if nrm < 1e-9:
        return None, None
    ex, ey = ex / nrm, ey / nrm
    wall_rms = float(np.sqrt(np.mean((d[:, 0] * ey - d[:, 1] * ex) ** 2)))
    dev = abs((far[0] - c[0]) * ey - (far[1] - c[1]) * ex)
    return dev, wall_rms


def _collapse_runs(mask: np.ndarray, valid: np.ndarray, score: np.ndarray,
                   wraps: bool) -> np.ndarray:
    """One representative index per contiguous run of mask==True.

    mask  : which beams detected a jump — defines the RUNS.
    valid : which of those are eligible to represent a run (near-sensor corners are not).
    score : per-beam strength; the highest-scoring VALID beam in each run wins.
    wraps : treat the array as circular, so a run across index 0 stays one run.

    Runs with no valid beam yield nothing.
    """
    n = len(mask)
    if n == 0 or not mask.any():
        return np.empty(0, dtype=int)

    idx = np.arange(n)
    if wraps and mask.all():
        # Fully circular run: no boundary to start from, so it is a single run.
        v = np.where(valid)[0]
        return np.array([v[np.argmax(score[v])]], dtype=int) if len(v) else np.empty(0, int)

    # Rotate so index 0 begins a run — otherwise a wrapping run is split into two.
    shift = 0
    if wraps:
        starts = np.where(mask & ~np.roll(mask, 1))[0]
        shift = int(starts[0]) if len(starts) else 0
    m, vv, sc = (np.roll(a, -shift) for a in (mask, valid, score))

    out = []
    edges = np.where(m[:-1] != m[1:])[0] + 1
    for run in np.split(idx, edges):
        if not m[run[0]]:
            continue
        cand = run[vv[run]]
        if len(cand):
            out.append(int(cand[np.argmax(sc[cand])]))
    return (np.asarray(out, dtype=int) + shift) % n if out else np.empty(0, dtype=int)


def segments_from_ordered_cloud(
    xyz: np.ndarray,
    res_h: float,
    max_range: float,
    ego_xy: Tuple[float, float],
    grazing_deg: float = 1.0,
    min_jump: float = 30.0,
    sigma: float = 0.0,
    wraps: bool = True,
    merge_radius: float = 10,
    max_seg_len: float = 30.0,
    elev_row: Optional[int] = None,
    ego_fwd: Optional[Tuple[float, float]] = None,
    fwd_half_angle_deg: float = 180.0,
    min_corner_range: float = 1.0,
    trend_window: int = 4,
    min_far_run: int = 10,
) -> Optional[np.ndarray]:
    """Occlusion boundary segments in WORLD (a0, a1) coordinates.

    xyz:      (n_h, n_v, 3) ordered sensor-frame cloud, ROS axes, NaN for non-returns.
    ego_xy:   sensor world position as (a0, a1) — the cloud is sensor-relative, so this
              is a pure translation (matching ObstacleCircles._build_circles's convention).
    elev_row: which elevation row to extract from; None picks the middle row, which is
              the one nearest the horizontal plane the vehicle drives in.
    ego_fwd:  (a0,a1) ego forward unit vector. When given, only boundaries whose CORNER
              lies within fwd_half_angle_deg of it are returned — a phantom emerging
              from a corner the ego has already driven past cannot be run into, and
              constraining it just brakes the ego for nothing. None ⇒ full 360°.
    min_corner_range: corners nearer than this are dropped. Beams clipping the airframe
              or the ground under the sensor jump hugely at ~0 m, and the resulting corner
              sits ON the sensor where its bearing is meaningless — so it passes the
              forward wedge at ANY half-angle. Mirrors ObstacleCircles's min_range.
    min_far_run: consecutive beams that must STAY on the far side for a jump to count as
              a corner. A fragmented wall (dropouts, or returns that slip between OCC
              fragments) produces isolated beams that fly past and then the surface
              resumes immediately — a real corner instead has the far range PERSIST.
              This is what removes the column of spurious corners along a wall face.
              0 or 1 disables the check.
    trend_window: beams fitted to the LOCAL WALL LINE on the near side of each candidate.
              A surface seen at grazing incidence produces a large per-beam range step, so
              |Δr| alone flags a wall viewed nearly edge-on along its whole length. But a
              flat wall's beam hit-points are COLLINEAR in Cartesian space at *any*
              incidence, while its range profile is convex — so a line fit, not a range
              extrapolation, is the incidence-invariant test. This fits a line to the
              trend_window near-side (wall) points and keeps the candidate only if the FAR
              endpoint sits well OFF that line: a true depth break flies off the wall, a
              grazing surface lands right on it and is rejected. Only rejects when the
              near-side points are themselves a clean line, so a thin real occluder whose
              window straddles background is never suppressed. 0 disables the test.
    fwd_half_angle_deg: half-width of that wedge. 180 is equivalent to no filter; 90
              keeps everything strictly ahead; the 100 default keeps a little past
              abeam so a corner is not dropped the instant it draws level.

    Returns (M, 2, 2): M segments, each [[a0_near, a1_near], [a0_far, a1_far]], with
    the NEAR endpoint being the corner point the hidden agent would round. None if no
    boundary is found.
    """
    n_h, n_v, _ = xyz.shape
    j = n_v // 2 if elev_row is None else int(np.clip(elev_row, 0, n_v - 1))

    #print(xyz[0].shape) # (360, 46, 3) shape of the point cloud 46 are the number of beams (circles)
    # ROS (x, y) -> world-aligned (a0, a1), the same mapping ObstacleCircles._build_circles uses.
    # Note: points are nan if no returned is got
    
    # Get the beam j and first point
    a0 = xyz[:, j, 0].astype(np.float64)
    # Get the beam j and the second point
    a1 = -xyz[:, j, 1].astype(np.float64)
    # Euclidean distance from the two points
    rng = np.hypot(a0, a1)

    # A non-return reached max range without hitting anything: a real edge when its
    # neighbour hit something, so substitute rather than discard.
    miss = ~np.isfinite(rng)
    if miss.any():
        # Rebuild the missing beams' direction from their azimuth index so the endpoint
        # is still well defined at max range.
        # n_h orizontal fov
        az = np.deg2rad(-0.5 * n_h * res_h + np.arange(n_h) * res_h)
        if ego_fwd is not None:
            _f = np.asarray(ego_fwd, float)
            if float(np.linalg.norm(_f)) > 1e-9:
                az = az + math.atan2(_f[1], _f[0])
        a0 = np.where(miss, np.cos(az) * max_range, a0)
        a1 = np.where(miss, np.sin(az) * max_range, a1)
        rng = np.where(miss, max_range, rng)

    idx = np.arange(n_h)
    nxt = (idx + 1) % n_h if wraps else idx[:-1] + 1
    cur = idx if wraps else idx[:-1]

    rA, rB = rng[cur], rng[nxt]
    near = np.where(rA <= rB, rA, rB)
    delta = np.abs(rA - rB)

    # A continuous surface at range r spans r*dtheta/tan(grazing) between adjacent
    # beams; a larger step is a genuine depth discontinuity. Scaling with range keeps
    # the test valid at distance, where the same physical gap subtends fewer beams.
    # thresh = np.maximum(near * math.radians(res_h) / math.tan(math.radians(grazing_deg))
    #                     + 3.0 * sigma, min_jump)

    thresh = min_jump
    # The absolute-step test flags every large Δr, INCLUDING a wall seen edge-on; the
    # Cartesian line-fit test below (in the per-candidate loop) then rejects the grazing
    # surfaces, which need the beam geometry that only survives to that loop.
    detect = delta > 100
    # if min_far_run and min_far_run > 1:
    #     # Past a genuine corner the escaping beams keep flying; past a dropout the surface
    #     # comes straight back. Require the far side to hold for min_far_run beams.
    #     persist = np.ones(len(cur), dtype=bool)
    #     far_side = np.maximum(rA, rB)
    #     for m in range(1, min_far_run):
    #         # Beams beyond the jump, on whichever side is the far one.
    #         ahead = np.where(rB >= rA, rng[(nxt + m) % n_h], rng[(cur - m) % n_h])
    #         persist &= ahead > (near + 0.5 * (far_side - near))
    #     detect = detect & persist

    # # Drop degenerate near-sensor corners before the wedge test, for the reason above.
    # valid = detect & (near > min_corner_range)
    # if not valid.any():
    #     return None

    # hit = _collapse_runs(detect, valid, -far_side, wraps)
    # if len(hit) == 0:
    #     return None

    # # Forward wedge, as cos of the half-angle so the test is a single dot product.
    # fwd = None
    # if ego_fwd is not None:
    #     f = np.asarray(ego_fwd, float)
    #     n = float(np.linalg.norm(f))
    #     if n > 1e-9:
    #         fwd = f / n
    #         cos_lim = math.cos(math.radians(float(np.clip(fwd_half_angle_deg, 0.0, 180.0))))

    # use_line = bool(trend_window) and trend_window >= 2
    # w = int(trend_window)
    # segs = []
    # for h in hit:
    #     i, k = cur[h], nxt[h]
    #     # Near endpoint = the corner; far endpoint = where the escaping beam landed.
    #     # `near_idx` is the corner beam; the wall extends AWAY from the gap from there,
    #     # so we walk indices backward when the near side is the earlier beam and forward
    #     # when it is the later one.
    #     if rng[i] <= rng[k]:
    #         p_near = np.array([a0[i], a1[i]])
    #         p_far = np.array([a0[k], a1[k]])
    #         near_idx, step = i, -1
    #     else:
    #         p_near = np.array([a0[k], a1[k]])
    #         p_far = np.array([a0[i], a1[i]])
    #         near_idx, step = k, +1

    #     # Grazing-surface rejection. Fit a line to the near-side wall points and drop the
    #     # candidate if the far endpoint continues that line (a wall seen edge-on) rather
    #     # than flying off it (a real depth break). Only trust the rejection when the wall
    #     # points are themselves a clean line — a window straddling background gives a
    #     # large wall_rms, so a thin genuine occluder is never suppressed.
    #     if use_line:
    #         offs = near_idx + step * np.arange(w)
    #         widx = offs % n_h if wraps else np.clip(offs, 0, n_h - 1)
    #         win = np.column_stack([a0[widx], a1[widx]])
    #         dev, wall_rms = _wall_line_deviation(win, p_far)
    #         if dev is not None and wall_rms <= thresh[h] and dev <= thresh[h]:
    #             continue

    #     # Drop boundaries behind the ego. Tested on the CORNER (the phantom's anchor)
    #     # in sensor-relative coords, which are already ego-centred.
    #     if fwd is not None:
    #         nrm = float(np.linalg.norm(p_near))
    #         if nrm > 1e-9 and float(p_near @ fwd) / nrm < cos_lim:
    #             continue

    #     # A beam that escaped to max range gives an unboundedly long segment; clip it
    #     # so one open sightline can't create a capsule spanning the whole map.
    #     v = p_far - p_near
    #     L = float(np.linalg.norm(v))
    #     if L > max_seg_len:
    #         p_far = p_near + v * (max_seg_len / L)
    #     segs.append((p_near, p_far))

    # # Every jump can be filtered out (all behind the forward wedge, say), leaving
    # # nothing to translate — np.array([]) would be 1-D and break the indexing below.
    # if not segs:
    #     return None

    # if not merge_radius:
    #     out = np.array([[s[0], s[1]] for s in segs])
    # else:
    #     # Collapse segments whose corners coincide — a single physical edge fires on
    #     # several adjacent beams and would otherwise seed a cluster of near-identical
    #     # capsules, all constraining the same geometry.
    #     # Of the duplicates the SHORTEST sightline wins, not the first in beam order:
    #     # a corner that fires on several beams lands its far endpoint anywhere from the
    #     # first surface behind the gap out to max_range, and the shortest one is where
    #     # the returns actually say the gap ends. Keeping it is the conservative choice —
    #     # a longer segment dilates into a capsule covering ground that is visibly free,
    #     # and beam order is arbitrary with respect to which of the two is which.
    #     kept = []
    #     for p_near, p_far in segs:
    #         L = float(np.hypot(*(p_far - p_near)))
    #         for qi, (q_near, _q_far, q_L) in enumerate(kept):
    #             if np.hypot(*(p_near - q_near)) < merge_radius:
    #                 if L < q_L:
    #                     kept[qi] = (p_near, p_far, L)
    #                 break
    #         else:
    #             kept.append((p_near, p_far, L))
    #     if not kept:
    #         return None
    #     out = np.array([[s[0], s[1]] for s in kept])

    # list of 360 per one beam detect, 
    h = np.where(detect)[0]
    if not len(h):  
        return None
    i, k = cur[h], nxt[h]
    #print(True if np.all(i, h) else False)
    #i contains the x,y in the current position; nxt[h] returns the x,y
    # take the nearest of the points which is the boundary
    # shape [len(i)]
    near_is_i = rng[i] <= rng[k]
    #near index
    n_idx = np.where(near_is_i, i, k)
    f_idx = np.where(near_is_i, k, i)
    out = np.stack([np.column_stack([a0[n_idx], a1[n_idx]]),
                    np.column_stack([a0[f_idx], a1[f_idx]])], axis=1)

    # Sensor-relative -> world (pure translation; axes are world-aligned).
    out[:, :, 0] += ego_xy[0]
    out[:, :, 1] += ego_xy[1]
    return out


def capsule_radius(d_base: float, v_target: float, t_k: float) -> float:
    """Forward-reachable-set radius around the boundary segment at horizon time t_k.

    A hidden agent anywhere on the segment moving at up to v_target in an arbitrary
    direction can be anywhere within v_target*t_k of it, so the reachable set is the
    segment dilated by that radius — a capsule. d_base is the ego's own safety margin
    (D_SAFE_HARD), added on top.
    """
    return d_base + v_target * t_k


def point_segment_distance_sym(px, py, ax, ay, bx, by, fmin, fmax, sqrt, eps: float = 1e-9):
    """Squared and absolute distance from (px,py) to segment AB, backend-agnostic.

    Pass the clamping/sqrt primitives of whichever backend is in play (ca.fmin/ca.fmax/
    ca.sqrt for the MPC's symbolics, np.minimum/np.maximum/np.sqrt for arrays), so one
    implementation serves both and they cannot drift apart.

    Distance to the SEGMENT (not the infinite line) is what makes the level sets
    capsules: within the span it measures to the rectangle's flank, beyond either end
    it measures to that endpoint, tracing the two end circles.

    Returns (d2, d).
    """
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    L2 = vx * vx + vy * vy + eps
    # Projection parameter clamped to [0,1] — the clamp is exactly what turns the
    # infinite-line distance into the capsule.
    t = fmax(0.0, fmin(1.0, (wx * vx + wy * vy) / L2))
    dx = wx - t * vx
    dy = wy - t * vy
    d2 = dx * dx + dy * dy
    return d2, sqrt(d2 + 1e-6)


def capsule_polygon(a: np.ndarray, b: np.ndarray, r: float, n_arc: int = 24) -> np.ndarray:
    """Closed outline of the capsule = all points within r of segment AB.

    Returns (2*n_arc + 1, 2) vertices for plotting: the two flanks of the rectangle
    joined by a semicircular arc at each end. Degenerate segments (a == b) fall back
    to a full circle, matching how the MPC treats point-sourced boundaries.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    v = b - a
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        th = np.linspace(0.0, 2.0 * np.pi, 2 * n_arc + 1)
        return a + r * np.column_stack([np.cos(th), np.sin(th)])

    u = v / L
    nrm = np.array([-u[1], u[0]])          # left normal
    base = math.atan2(nrm[1], nrm[0])
    # Arc around b sweeps from the left flank to the right flank, and vice versa at a,
    # so the two arcs close the rectangle into a single convex outline.
    arc_b = base + np.linspace(0.0, -np.pi, n_arc)
    arc_a = base + np.linspace(np.pi, 0.0, n_arc)
    pts = np.vstack([
        b + r * np.column_stack([np.cos(arc_b), np.sin(arc_b)]),
        a + r * np.column_stack([np.cos(arc_a), np.sin(arc_a)]),
    ])
    return np.vstack([pts, pts[:1]])       # close the ring


def point_segments_min_distance(px: np.ndarray, py: np.ndarray,
                                segs: np.ndarray) -> np.ndarray:
    """(K,) distance from each of K points to the NEAREST of M capsule axes.

    px, py : (K,) rollout coordinates.
    segs   : (M, 2, 2) boundary segments [[near, far], ...].

    This is the capsule distance written as one expression rather than three. The
    keep-out around a boundary is the union of a disc at each endpoint and the
    rectangle spanning them; the distance to that union is
        min( |P-A|, |P-B|, perpendicular distance to AB )
    and that minimum IS the point-to-SEGMENT distance: the clamp of the projection
    parameter to [0,1] in point_segment_distance_sym selects the rectangle when the
    foot of the perpendicular lands within the span and the nearer endpoint disc
    when it does not. Computing the three pieces separately would evaluate the same
    numbers by a longer route.

    Deliberately routed through point_segment_distance_sym with NumPy primitives —
    the exact function the MPC feeds CasADi primitives — so the sampling planner and
    the NLP cannot disagree about where a capsule is.
    """
    segs = np.asarray(segs, float).reshape(-1, 2, 2)
    a, b = segs[:, 0, :], segs[:, 1, :]
    _, d = point_segment_distance_sym(
        px[:, None], py[:, None],
        a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1],
        np.minimum, np.maximum, np.sqrt)
    return d.min(axis=1)


def point_segment_distance_np(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized NumPy point-to-segment distance. p:(N,2) a,b:(2,) -> (N,)."""
    v = b - a
    L2 = float(v @ v) + 1e-12
    t = np.clip(((p - a) @ v) / L2, 0.0, 1.0)
    proj = a + t[:, None] * v
    return np.linalg.norm(p - proj, axis=1)


class OcclusionSegmentPublisher:
    """Publishes THIS SCAN'S detected boundary segments for the Unity Scene view.

    Topic  : /occlusion_segments   (std_msgs/Float32MultiArray)
    Payload: [v_target, d_safe_hard, t_grow_max] ++ [a0_near, a1_near, a0_far, a1_far] * M

    Same contract as DynamicClusterPublisher, and for the same reason: the model
    parameters ride in the message so the overlay cannot drift from what the planner
    used, unlike PhantomAgentVisualizer which re-declares them in the Inspector. That
    matters more here — PhantomAgentVisualizer draws boundaries UNITY detected in its own
    edge pass, which is a different detector from segments_from_ordered_cloud; this feed
    shows the Python one, corner first.

    Coordinates are controller world (a0, a1). Unity converts: a0 -> Z, a1 -> X.

    A no-op when rclpy is unavailable, matching how ObstacleCircles degrades.
    """

    TOPIC = "/occlusion_segments"
    N_PARAMS = 3        # leading floats before the segment rows
    ROW = 4             # floats per segment row: corner (a0,a1) then far end (a0,a1)

    def __init__(self, topic: str = TOPIC):
        self.topic = topic
        self._node = None
        self._pub = None

    def start(self) -> bool:
        if not _HAS_RCLPY:
            print("[OcclusionSegmentPublisher] rclpy not importable — Unity viewer feed disabled.")
            return False
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("occlusion_segments_viz")
        self._pub = self._node.create_publisher(Float32MultiArray, self.topic, 10)
        print(f"[OcclusionSegmentPublisher] publishing boundaries on '{self.topic}'")
        return True

    def publish(self, segs: Optional[np.ndarray], v_target: float, d_safe: float,
                t_grow_max: float) -> None:
        """Send the CURRENT boundary set, (M,2,2) as segments_from_ordered_cloud returns.

        Call every control step, including with an empty set — an empty message is what
        clears the overlay when the boundaries are gone."""
        if self._pub is None:
            return
        s = (np.zeros((0, 2, 2)) if segs is None or not len(segs)
             else np.asarray(segs, float).reshape(-1, 2, 2))
        msg = Float32MultiArray()
        d0 = MultiArrayDimension(label="params", size=self.N_PARAMS, stride=self.N_PARAMS)
        d1 = MultiArrayDimension(label="segments", size=len(s),
                                 stride=self.ROW * max(1, len(s)))
        msg.layout.dim = [d0, d1]
        msg.data = [float(v_target), float(d_safe), float(t_grow_max)] + \
                   [float(x) for x in s.reshape(-1, self.ROW).ravel()]
        self._pub.publish(msg)

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = self._pub = None


class OcclusionCornerTracker:
    """Gives detected occlusion corners TEMPORAL IDENTITY across scans.

    segments_from_ordered_cloud() re-derives corners from scratch every scan, so the
    SAME physical corner lands on a slightly different beam each time and comes back
    displaced by up to the beam spacing (~r·Δθ, e.g. 0.5 m at 30 m with 1° resolution).
    Downstream that looks like a stream of brand-new boundaries: the plot accumulates a
    smear of near-duplicate keep-outs, and the MPC's constraint centre jitters every
    step, which shows up as steering chatter.

    This associates each detection to the nearest live track within assoc_radius and
    smooths it (exponential moving average) instead of creating a new one. A track that
    goes unseen for ttl seconds is dropped, so corners genuinely left behind do expire.

    Deliberately a nearest-neighbour tracker, not a Kalman filter: occlusion corners are
    static world features (it is the EGO that moves), so there is no motion model worth
    estimating — only measurement noise worth averaging down.
    """

    def __init__(self, assoc_radius: float = 3.0, alpha: float = 0.35,
                 ttl: float = 0.6, min_hits: int = 2):
        self.assoc_radius = float(assoc_radius)
        self.alpha = float(alpha)          # EMA weight on the NEW measurement
        self.ttl = float(ttl)
        self.min_hits = int(min_hits)      # suppress one-off flickers
        self._tracks = []                  # list of dicts: id, pos, far, last_seen, hits
        self._next_id = 0                  # stable identity, so consumers can key on it

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0

    def update(self, segs: Optional[np.ndarray], now: float) -> Optional[np.ndarray]:
        """Fold this scan's segments into the tracks; return the STABLE segments.

        segs: (M,2,2) as produced by segments_from_ordered_cloud, or None.
        Returns (K,2,2) smoothed segments for confirmed tracks, or None if none.
        Pair with ids() to key downstream state on a STABLE identity rather than on
        coordinates, which keep drifting slightly as the EMA settles.
        """
        if segs is not None and len(segs):
            segs = np.asarray(segs, float).reshape(-1, 2, 2)
            claimed = set()
            for seg in segs:
                corner, far = seg[0], seg[1]
                # Nearest unclaimed track within the association gate.
                best, best_d = None, self.assoc_radius
                for ti, tr in enumerate(self._tracks):
                    if ti in claimed:
                        continue
                    d = float(np.hypot(*(tr["pos"] - corner)))
                    if d < best_d:
                        best, best_d = ti, d
                if best is None:
                    self._tracks.append({"id": self._next_id,
                                         "pos": corner.copy(), "far": far.copy(),
                                         "last_seen": now, "hits": 1})
                    self._next_id += 1
                    claimed.add(len(self._tracks) - 1)
                else:
                    tr = self._tracks[best]
                    a = self.alpha
                    tr["pos"] = (1.0 - a) * tr["pos"] + a * corner
                    tr["far"] = (1.0 - a) * tr["far"] + a * far
                    tr["last_seen"] = now
                    tr["hits"] += 1
                    claimed.add(best)

        # Expire tracks not re-observed recently.
        self._tracks = [t for t in self._tracks if (now - t["last_seen"]) <= self.ttl]

        out = [np.stack([t["pos"], t["far"]]) for t in self._tracks
               if t["hits"] >= self.min_hits]
        return np.array(out) if out else None

    def ids(self):
        """Track ids aligned with the rows returned by the last update()."""
        return [t["id"] for t in self._tracks if t["hits"] >= self.min_hits]

    @property
    def n_tracks(self) -> int:
        return len(self._tracks)

def occlusion_stage_cost(d, v, t_k, v_target, d_safe, w_obs, fmax=None, sqrt=None, clip=None,
                         w_sight=None, a_brake=None, v_floor=None, cost_current = None, dyn = False, action = None,
                         t_grow_max=None, w_soft=None, d_infl=None):
    """Occlusion stage cost. Backend-agnostic: pass numpy or CasADi primitives.

    d        : distance from the (rollout/predicted) pose to the nearest occlusion boundary
    v        : speed at this stage [m/s] — used only by the RSS sightline term below
    t_k      : horizon time of this stage [s]
    v_target : assumed max speed of the hidden agent [m/s]
    d_safe   : base (t=0) keep-out radius [m]
    w_obs    : hard-constraint weight
    w_sight  : RSS sightline weight, or None to skip that term (a_brake/v_floor then unused)
    t_grow_max : cap [s] on the expansion time, or None for the (freezing) uncapped growth
    w_soft   : weight of the soft influence ring outside r_keep, or None/0 to skip it
    d_infl   : width [m] of that ring. Cost is w_soft at d = r_keep, 0 at r_keep + d_infl
    Returns the scalar/array stage cost contribution.
    """
    import numpy as _np
    fmax = _np.maximum if fmax is None else fmax
    sqrt = _np.sqrt if sqrt is None else sqrt

    t_eff  = t_k if t_grow_max is None else min(t_k, t_grow_max)
    r_grow = v_target * t_eff
    r_keep = r_grow + d_safe

    # Hard term: binary, so every breach costs the same regardless of depth — a constraint,
    cost = np.where(d < r_keep, w_obs, 0.0)

    #cost = w_obs * (fmax(0.0, r_keep - d)/d_safe) ** 2
    # Soft term: the gradient the hard term cannot give. A one-sided quadratic over an
    # influence ring of width d_infl OUTSIDE r_keep, normalised so w_soft is the cost of
    # sitting exactly on the keep-out boundary and 0 at r_keep + d_infl and beyond.
    #   - decreasing in d, so farther is genuinely cheaper (the opposite of `+ k*d`)
    #   - bounded and local, so it cannot outvote the goal far from any boundary (which an
    #     unbounded `-k*d` would, by paying the ego to run away forever)
    if w_soft and d_infl:
            margin = fmax(0.0, (r_keep + d_infl) - d)
            cost = cost + w_soft * margin * margin

    # if not dyn:
    #     print(f"TIMESTAMP_occlussion: {t_k} -> r_grow : {r_grow}, distance from occlussion : {d}, total distance to keep : {r_keep} , current min cost: {np.min(cost_current)}, current_actions: {action[np.argmin(cost_current)]}")
    # else:
    #     print(f"TIMESTAMP_dynamic: {t_k} -> r_grow : {r_grow}, distance from dyn : {d}, total distance to keep : {r_keep} , current min cost: {np.min(cost_current)}")

    return cost
