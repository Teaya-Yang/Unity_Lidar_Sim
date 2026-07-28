"""
free_space.py
=============
"Is the place this object used to be now EMPTY?" — dynamic-object detection by
free-space violation, using the ordered cloud's range image.

WHY THIS EXISTS. dynamic_clusters.py's original test asks whether a cluster's CENTROID
moved further than the cluster's own extent over a window. That test is on the wrong
observable, and it fails in both directions at once:

  * FALSE POSITIVES on scenery. A long wall does not come back as one stable cluster —
    it re-segments into a different set of fragments every scan as the visible span
    changes. Each fragment is small (so its extent threshold is small) while its
    centroid slides metres along the wall. Measured: a terminal face took cl 3 -> 21
    and dyn -> 19, with the reported speeds shadowing the EGO's own speed.
  * LATE DETECTION on real movers. A verdict needs a closed dyn_window (2 s), aligned
    to scan boundaries, plus min_hits confirmations. At 6 m/s that is 12-24 m of travel
    before the planner reacts.

Tuning cannot fix both — extent_frac trades one directly against the other. The two
symptoms share one cause: over a single window, a slow mover and a static body whose
VISIBLE FACE is drifting produce the same centroid signal, so no threshold on that
signal separates them.

THE DIFFERENT OBSERVABLE. Project an object's PREVIOUS position into the CURRENT scan's
beam grid. If the beams that should terminate on it instead return something FURTHER
AWAY, they passed through the space it used to occupy — so it left. This is immune to
the failure above by construction: a static body's beams always stop on it, no matter
how far its apparent centroid wanders, and no part of the test depends on stable
segmentation or on centroids at all. It also needs no window — one scan is a verdict.

THE THREE-WAY VERDICT IS THE WHOLE DESIGN. A naive version treats "not occupied" as
"moved" and immediately fires on everything the sensor merely cannot see:

    MOVED       beams reach PAST where the body was            -> it vacated
    OCCUPIED    a beam still terminates at the body's range    -> it is still there
    INCONCLUSIVE  beams stop SHORT (something else is now in front), or the position
                  falls outside the FOV, or too few beams resolve it

INCONCLUSIVE is not a weak "no" — it must not count as evidence in EITHER direction.
An object hidden behind a passing vehicle is not moving and not stationary; it is
unobserved, and the caller has to keep whatever it already believed.

OCCUPIED VETOES. If ANY beam in the window still terminates at the body's range, the
verdict is OCCUPIED even when other beams read through. That asymmetry is what makes the
elevation band safe to widen: a beam angled slightly above a low vehicle legitimately
clears its roof and reads "through", and treating that as motion would false-positive on
every parked car. One beam hitting the body is proof it is there; many beams missing it
is only evidence when NONE of them hit.

FRAMES. The cloud is sensor-relative with WORLD-ALIGNED AXES — a pure translation, the
same convention ObstacleCircles._build_circles uses. So no rotation and no ego heading
enter the projection: a world point's bearing in the beam grid is just the bearing from
the sensor to it. ROS -> controller local is (a0, a1) = (x, -y), as everywhere else.

The beam-index <-> bearing mapping is derived EMPIRICALLY from each scan's own valid
returns rather than reimplementing LaserSensor3D's angle arithmetic. Its convention
(start angle, direction, the 360-degree wrap de-duplication) then cannot drift out of
sync with the C# — the cloud defines it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Verdicts.
MOVED = 1
OCCUPIED = -1
INCONCLUSIVE = 0


class FreeSpaceChecker:
    """Range-image free-space queries against one scan.

    Call set_scan() once per scan, then verdict() per track. Everything that is shared
    across tracks (the range image, the azimuth fit) is built once in set_scan().
    """

    def __init__(self, max_range: float = 1000.0, slack: float = 2.0,
                 min_through: int = 3, core_frac: float = 0.6,
                 elev_halfband: int = 3, min_beams: int = 2):
        self.max_range = float(max_range)
        self.slack = float(slack)          # [m] range tolerance. Must cover the voxel/cluster
                                           #   quantisation, or a body's own far face reads as
                                           #   "through" and every object looks like it moved.
        self.min_through = int(min_through)  # beams that must see past before MOVED is declared.
                                             #   >1 so a single stray long return is not a verdict.
        self.core_frac = float(core_frac)  # sample the body's CORE, not its silhouette edge:
                                           #   edge beams graze and their range is unstable, so
                                           #   they manufacture "through" votes on static bodies.
        self.elev_halfband = int(elev_halfband)  # rows either side of the horizontal to test
        self.min_beams = int(min_beams)    # fewer resolvable beams than this -> INCONCLUSIVE
                                           #   (a distant body subtends too little to judge)

        self._rng: Optional[np.ndarray] = None    # (n_h, n_v) range per beam [m]
        self._ego = (0.0, 0.0)
        self._n_h = 0
        self._n_v = 0
        self._fit: Optional[Tuple[float, float]] = None   # (slope, intercept) index -> bearing
        self._wraps = False
        self._rows: Optional[np.ndarray] = None

    # ── Per-scan setup ────────────────────────────────────────────────────────

    def set_scan(self, xyz: Optional[np.ndarray], ego_xy) -> bool:
        """Ingest one ordered cloud. xyz: (n_h, n_v, 3) sensor-frame ROS xyz, NaN for
        non-returns. Returns False if the scan cannot be used, in which case every
        subsequent verdict() is INCONCLUSIVE (the safe answer — see the module docstring).
        """
        self._rng = None
        if xyz is None or xyz.ndim != 3 or xyz.shape[2] != 3:
            return False
        n_h, n_v = xyz.shape[0], xyz.shape[1]
        if n_h < 8 or n_v < 1:
            return False

        a0 = xyz[:, :, 0]
        a1 = -xyz[:, :, 1]
        rng = np.hypot(a0, a1)

        # A NON-RETURN IS FREE SPACE, not missing data: the beam was cast and nothing
        # stopped it, which is the strongest possible "the object is not there". Mapping
        # it to max_range rather than dropping it is most of this method's sensitivity —
        # an object that moves off into open apron leaves non-returns behind it.
        rng = np.where(np.isfinite(rng), rng, self.max_range)

        # Azimuth per column, from the scan's own returns. All beams in a column share an
        # azimuth, so any valid one defines it; the circular mean over the column is used
        # so a single noisy return cannot set it.
        with np.errstate(invalid="ignore"):
            az = np.arctan2(a1, a0)
        valid = np.isfinite(a0) & np.isfinite(a1) & (np.hypot(a0, a1) > 1e-3)
        cols = []
        for i in range(n_h):
            m = valid[i]
            if not m.any():
                continue
            s = np.sin(az[i][m]).mean()
            c = np.cos(az[i][m]).mean()
            if s * s + c * c < 1e-6:
                continue
            cols.append((i, float(np.arctan2(s, c))))
        # DELIBERATELY A LOW BAR. Over open apron most beams are non-returns, so only a
        # handful of columns carry a bearing — an earlier version demanded n_h/8 of them
        # and silently disabled the whole detector on exactly the sparse scenes it is
        # most needed for. The grid is uniform by construction, so three columns already
        # determine it; the residual check below is what actually guards correctness.
        if len(cols) < 3:
            return False

        idx = np.array([c[0] for c in cols], float)
        th = np.array([c[1] for c in cols], float)

        # Slope from ADJACENT columns only. np.unwrap over the bearing sequence is wrong
        # here: the columns with returns are not contiguous, so a gap of k columns is a
        # genuine jump of k*res that unwrap would "correct" back into the principal
        # branch. Neighbouring pairs are immune, being less than a beam apart.
        adj = np.where(np.diff(idx) == 1.0)[0]
        if len(adj) == 0:
            return False
        d_th = np.arctan2(np.sin(th[adj + 1] - th[adj]), np.cos(th[adj + 1] - th[adj]))
        slope = float(np.median(d_th))
        if abs(slope) < 1e-9:
            return False

        # Intercept: every column implies b_i = theta_i - slope*i, equal modulo 2pi, so
        # take their circular mean rather than a least-squares intercept that a single
        # wrapped sample could drag off.
        b_i = th - slope * idx
        intercept = float(np.arctan2(np.sin(b_i).mean(), np.cos(b_i).mean()))

        # Residual check: with a uniform grid the fit is exact, so anything worse than
        # half a beam means the cloud is not the grid we think it is (wrong scan_shape,
        # a partially-filled message) and every verdict built on it would be misaimed.
        pred = intercept + slope * idx
        resid = np.abs(np.arctan2(np.sin(th - pred), np.cos(th - pred)))
        if float(np.max(resid)) > 0.5 * abs(slope):
            return False

        self._rng = rng
        self._ego = (float(ego_xy[0]), float(ego_xy[1]))
        self._n_h, self._n_v = n_h, n_v
        self._fit = (float(slope), float(intercept))
        # A full-circle scan wraps, so column 0 and column n_h-1 are neighbours and an
        # index may be taken modulo n_h. A sector scan must NOT wrap — off the end is
        # outside the FOV, which is INCONCLUSIVE, not a lookup at the far edge.
        self._wraps = abs(abs(slope) * n_h - 2.0 * np.pi) < 0.05
        mid = (n_v - 1) // 2
        lo = max(0, mid - self.elev_halfband)
        hi = min(n_v - 1, mid + self.elev_halfband)
        self._rows = np.arange(lo, hi + 1)
        return True

    @property
    def ready(self) -> bool:
        return self._rng is not None

    # ── The query ─────────────────────────────────────────────────────────────

    def verdict(self, p_world, r_body: float) -> int:
        """Was the body at p_world (world a0, a1) with radius r_body vacated?

        Returns MOVED / OCCUPIED / INCONCLUSIVE.
        """
        if self._rng is None:
            return INCONCLUSIVE

        d0 = float(p_world[0]) - self._ego[0]
        d1 = float(p_world[1]) - self._ego[1]
        d = float(np.hypot(d0, d1))
        r = max(0.5, float(r_body))
        # Too close to resolve: the angular width explodes and the near-field is where
        # the ego's own airframe returns live.
        if d < max(2.0, r) or d > self.max_range:
            return INCONCLUSIVE

        theta = float(np.arctan2(d1, d0))
        slope, intercept = self._fit
        idx_f = (theta - intercept) / slope
        # Sample the CORE of the body only — see core_frac.
        half = float(np.arctan2(self.core_frac * r, d)) / abs(slope)
        i_lo, i_hi = int(np.floor(idx_f - half)), int(np.ceil(idx_f + half))

        cols = np.arange(i_lo, i_hi + 1)
        if self._wraps:
            cols = np.mod(cols, self._n_h)
        else:
            cols = cols[(cols >= 0) & (cols < self._n_h)]
            if len(cols) == 0:
                return INCONCLUSIVE          # outside the FOV

        # The body spans r in depth too, so "at the body's range" is a shell of
        # half-thickness r + slack around d, not a point.
        near = d - r - self.slack
        far = d + r + self.slack

        sub = self._rng[np.ix_(cols, self._rows)]
        occupied = np.count_nonzero((sub >= near) & (sub <= far))
        if occupied > 0:
            # VETO. One beam terminating on the body proves it is still there, and
            # outranks any number of beams that read through (which a slightly-high row
            # legitimately does over a low vehicle).
            return OCCUPIED

        through = int(np.count_nonzero(sub > far))
        blocked = int(np.count_nonzero(sub < near))

        # ANY BEAM STOPPING SHORT MAKES THE WHOLE QUERY INCONCLUSIVE. Something now sits
        # between the sensor and the probed space, so that space is not observed at all —
        # the body could perfectly well still be sitting there behind the blocker. An
        # earlier version let `through >= min_through` win even with blocked > 0, which
        # is how a partially-occluded static body produced a MOVED verdict.
        if blocked > 0:
            return INCONCLUSIVE

        # A MAJORITY of the beams that resolved must see past, not just min_through of
        # them. At GRAZING INCIDENCE — the ego alongside a wall, the beams nearly
        # parallel to its face — a handful of beams legitimately slip along the surface
        # and return far ranges while the surface is still plainly there. Requiring the
        # through votes to dominate the window rejects that; requiring only an absolute
        # count does not, and it was producing exactly one phantom mover per pass.
        tested = int(sub.size)
        if (through >= max(self.min_through, self.min_beams)
                and through >= 0.6 * tested):
            return MOVED
        return INCONCLUSIVE
