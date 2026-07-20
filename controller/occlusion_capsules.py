"""
Occlusion boundary SEGMENTS from LiDAR range jumps, and the forward-reachable-set
capsules they seed (Firoozi et al.).

The pipeline the paper describes, and what lives where:

  1. DETECTION (here, `segments_from_ordered_cloud`) — at the current timestep, find
     occlusion boundaries by looking for large jumps between consecutive LiDAR range
     values. Where one ray grazes an obstacle corner and the adjacent ray shoots far
     past it, the segment joining those two endpoints is a boundary line.

  2. EXPANSION (here, `capsule_radius`; consumed by the MPC) — a hidden agent is
     assumed to lurk anywhere ON that boundary line and move outward at up to
     v_target. Its forward reachable set at horizon time t is the boundary segment
     dilated by v_target*t: a CAPSULE (rectangle + two end circles), which is exactly
     the set of points within that radius of the segment.

  3. MPC (taxi_controller_mpc.py) — the growing capsules become stage constraints, so
     the ego keeps d_safe from each expanding danger zone at the matching timestep.

NOTE ON THE GROWTH TERM. The paper writes d_target = v_target/Δt, which is
dimensionally acceleration, not distance (m/s ÷ s = m/s²); at V_TARGET=3, DT=0.1 it
would be 30 m per step and the capsules would swallow the map immediately. The
intended quantity is distance = speed · time, so the radius used here (and already
used by the circle formulation in both controllers) is v_target · t_k.

Detection needs the ORDERED cloud — beam adjacency IS the method — so non-returns are
kept as max-range rather than dropped the way LidarCostmap._parse_cloud drops them.
"""

import math
from typing import Optional, Tuple

import numpy as np


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

    Unlike LidarCostmap._parse_cloud this preserves beam order and keeps invalid
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


def segments_from_ordered_cloud(
    xyz: np.ndarray,
    res_h: float,
    max_range: float,
    ego_xy: Tuple[float, float],
    grazing_deg: float = 10.0,
    min_jump: float = 2.0,
    sigma: float = 0.0,
    wraps: bool = True,
    merge_radius: float = 1.5,
    max_seg_len: float = 30.0,
    elev_row: Optional[int] = None,
    ego_fwd: Optional[Tuple[float, float]] = None,
    fwd_half_angle_deg: float = 100.0,
    min_corner_range: float = 1.0,
) -> Optional[np.ndarray]:
    """Occlusion boundary segments in WORLD (a0, a1) coordinates.

    xyz:      (n_h, n_v, 3) ordered sensor-frame cloud, ROS axes, NaN for non-returns.
    ego_xy:   sensor world position as (a0, a1) — the cloud is sensor-relative, so this
              is a pure translation (matching LidarCostmap._ingest's convention).
    elev_row: which elevation row to extract from; None picks the middle row, which is
              the one nearest the horizontal plane the vehicle drives in.
    ego_fwd:  (a0,a1) ego forward unit vector. When given, only boundaries whose CORNER
              lies within fwd_half_angle_deg of it are returned — a phantom emerging
              from a corner the ego has already driven past cannot be run into, and
              constraining it just brakes the ego for nothing. None ⇒ full 360°.
    min_corner_range: corners nearer than this are dropped. Beams clipping the airframe
              or the ground under the sensor jump hugely at ~0 m, and the resulting corner
              sits ON the sensor where its bearing is meaningless — so it passes the
              forward wedge at ANY half-angle. Mirrors LidarCostmap's min_range.
    fwd_half_angle_deg: half-width of that wedge. 180 is equivalent to no filter; 90
              keeps everything strictly ahead; the 100 default keeps a little past
              abeam so a corner is not dropped the instant it draws level.

    Returns (M, 2, 2): M segments, each [[a0_near, a1_near], [a0_far, a1_far]], with
    the NEAR endpoint being the corner point the hidden agent would round. None if no
    boundary is found.
    """
    n_h, n_v, _ = xyz.shape
    j = n_v // 2 if elev_row is None else int(np.clip(elev_row, 0, n_v - 1))

    # ROS (x, y) -> world-aligned (a0, a1), the same mapping LidarCostmap._ingest uses.
    a0 = xyz[:, j, 0].astype(np.float64)
    a1 = -xyz[:, j, 1].astype(np.float64)
    rng = np.hypot(a0, a1)

    # A non-return reached max range without hitting anything: a real edge when its
    # neighbour hit something, so substitute rather than discard.
    miss = ~np.isfinite(rng)
    if miss.any():
        # Rebuild the missing beams' direction from their azimuth index so the endpoint
        # is still well defined at max range.
        az = np.deg2rad(-0.5 * n_h * res_h + np.arange(n_h) * res_h)
        a0 = np.where(miss, np.cos(az) * max_range, a0)
        a1 = np.where(miss, np.sin(az) * max_range, a1)
        rng = np.where(miss, max_range, rng)

    idx = np.arange(n_h)
    nxt = (idx + 1) % n_h if wraps else idx[:-1] + 1
    cur = idx if wraps else idx[:-1]

    rA, rB = rng[cur], rng[nxt]
    near = np.minimum(rA, rB)
    delta = np.abs(rA - rB)

    # A continuous surface at range r spans r*dtheta/tan(grazing) between adjacent
    # beams; a larger step is a genuine depth discontinuity. Scaling with range keeps
    # the test valid at distance, where the same physical gap subtends fewer beams.
    thresh = np.maximum(near * math.radians(res_h) / math.tan(math.radians(grazing_deg))
                        + 3.0 * sigma, min_jump)
    # Drop degenerate near-sensor corners before the wedge test, for the reason above.
    hit = np.where((delta > thresh) & (near > min_corner_range))[0]
    if len(hit) == 0:
        return None

    # Forward wedge, as cos of the half-angle so the test is a single dot product.
    fwd = None
    if ego_fwd is not None:
        f = np.asarray(ego_fwd, float)
        n = float(np.linalg.norm(f))
        if n > 1e-9:
            fwd = f / n
            cos_lim = math.cos(math.radians(float(np.clip(fwd_half_angle_deg, 0.0, 180.0))))

    segs = []
    for h in hit:
        i, k = cur[h], nxt[h]
        # Near endpoint = the corner; far endpoint = where the escaping beam landed.
        if rng[i] <= rng[k]:
            p_near = np.array([a0[i], a1[i]])
            p_far = np.array([a0[k], a1[k]])
        else:
            p_near = np.array([a0[k], a1[k]])
            p_far = np.array([a0[i], a1[i]])

        # Drop boundaries behind the ego. Tested on the CORNER (the phantom's anchor)
        # in sensor-relative coords, which are already ego-centred.
        if fwd is not None:
            nrm = float(np.linalg.norm(p_near))
            if nrm > 1e-9 and float(p_near @ fwd) / nrm < cos_lim:
                continue

        # A beam that escaped to max range gives an unboundedly long segment; clip it
        # so one open sightline can't create a capsule spanning the whole map.
        v = p_far - p_near
        L = float(np.linalg.norm(v))
        if L > max_seg_len:
            p_far = p_near + v * (max_seg_len / L)
        segs.append((p_near, p_far))

    # Every jump can be filtered out (all behind the forward wedge, say), leaving
    # nothing to translate — np.array([]) would be 1-D and break the indexing below.
    if not segs:
        return None

    if not merge_radius:
        out = np.array([[s[0], s[1]] for s in segs])
    else:
        # Collapse segments whose corners coincide — a single physical edge fires on
        # several adjacent beams and would otherwise seed a cluster of near-identical
        # capsules, all constraining the same geometry.
        kept = []
        for p_near, p_far in segs:
            if any(np.hypot(*(p_near - q[0])) < merge_radius for q in kept):
                continue
            kept.append((p_near, p_far))
        if not kept:
            return None
        out = np.array([[s[0], s[1]] for s in kept])

    # Sensor-relative -> world (pure translation; axes are world-aligned).
    out[:, :, 0] += ego_xy[0]
    out[:, :, 1] += ego_xy[1]
    return out


def capsule_radius(d_base: float, v_target: float, t_k: float) -> float:
    """Forward-reachable-set radius around the boundary segment at horizon time t_k.

    A hidden agent anywhere on the segment moving at up to v_target in an arbitrary
    direction can be anywhere within v_target*t_k of it, so the reachable set is the
    segment dilated by that radius — a capsule. d_base is the ego's own safety margin
    (D_SAFE_OCC / D_INFL_OCC), added on top.
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


def point_segment_distance_np(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized NumPy point-to-segment distance. p:(N,2) a,b:(2,) -> (N,)."""
    v = b - a
    L2 = float(v @ v) + 1e-12
    t = np.clip(((p - a) @ v) / L2, 0.0, 1.0)
    proj = a + t[:, None] * v
    return np.linalg.norm(p - proj, axis=1)
