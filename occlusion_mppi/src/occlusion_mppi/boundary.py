"""Distance from rollout poses to the nearest occlusion boundary.

This is the ONLY thing that changes between the 2D and 3D controllers.

In 2D a boundary was a SEGMENT, so the distance query was point-to-segment and
needed `point_segments_min_distance`. ROG-Map gives a set of VOXELS instead, so
the query is plain point-to-nearest-point and a KD-tree answers it exactly. No
clustering, no capsule fitting, no principal-axis fit is required to get `d`.

Those are still worth adding later -- for tracking boundaries across frames, or
for per-boundary v_target -- but none of them affect the distance itself.

ON THE EXPANSION BEING ISOTROPIC
--------------------------------
Nothing in here expands anything. The keep-out radius r_keep(t) is applied in
cost.py and grows equally in every direction, because the hidden agent's heading
is unknown (see capsule_radius). A line-of-sight-only expansion would assume the
agent moves toward the ego and would miss one cutting laterally around the
occluder edge.

Two honest limitations of the Euclidean KD-tree distance used here:

  * It is NOT geodesic. An agent deep in a shadow must travel around the
    occluder, but Euclidean distance lets the keep-out bulge straight through
    solid geometry. Conservative (never unsafe), but it over-inflates near big
    occluders.
  * Dilating the FRONTIER (not the shadow interior) is the correct worst case:
    the nearest place a hidden agent can possibly be is the shadow mouth.
"""

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    import warnings
    warnings.warn(
        "scipy is not available: occlusion distances fall back to an O(M*N) "
        "brute-force scan. Correct, but far slower. Install scipy.",
        RuntimeWarning)


class BoundarySet:
    """Immutable snapshot of occlusion-boundary voxels, with a nearest-neighbour index.

    Rebuild one per map update; querying is then O(log n) per pose, which is what
    makes a K x H rollout batch affordable without precomputing a distance field.
    """

    def __init__(self, points, z_band=None, ego_z=None, planar=True, query_z=None):
        """points : (N,3) world-frame boundary voxel centres.

        z_band  : if not None, keep only voxels with |z - ego_z| <= z_band. Use this
                  when flying at fixed altitude so the 2D cost stays valid -- a
                  boundary 4 m overhead is not something the ego can collide with
                  laterally, and including it inflates the keep-out for no reason.
        planar  : if True the distance is computed in XY only (matching the 2D
                  controller). Set False for a genuinely 3D keep-out.
        query_z : altitude at which to place 2D queries when planar=False. The
                  plant is 2D (fixed altitude), so every caller -- rollouts
                  included -- hands over (M,2). Lifting them here keeps the 3D
                  distance available without threading a z through the planner.
                  Required when planar=False; ignored otherwise.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if z_band is not None and ego_z is not None and len(pts):
            pts = pts[np.abs(pts[:, 2] - ego_z) <= z_band]

        self.planar = planar
        self.query_z = query_z
        self.points = pts
        self._xy = pts[:, :2] if planar else pts

        self._tree = cKDTree(self._xy) if (len(pts) and _HAS_SCIPY) else None

    def __len__(self):
        return len(self.points)

    def _prep(self, query):
        """Coerce a query to the tree's dimension, lifting (M,2) to (M,3) if needed.

        A 3D set queried with a 2D pose is the normal case, not an error: the ego
        flies at fixed altitude, so its z is a property of the BoundarySet's
        configuration rather than of each query.
        """
        q = np.asarray(query, dtype=float)
        if self.planar:
            # A 3D ego querying a planar set is now the normal case (dim=3 with
            # boundary_planar): drop z rather than hand a (M,3) query to a tree
            # built on (N,2), which SciPy rejects outright.
            return q[:, :2] if (q.ndim == 2 and q.shape[1] == 3) else q
        if q.ndim != 2 or q.shape[1] != 2:
            return q
        if self.query_z is None:
            raise ValueError(
                "BoundarySet(planar=False) received a 2D query but no query_z was "
                "given at construction; the ego altitude is unknown.")
        return np.column_stack([q, np.full(len(q), float(self.query_z))])

    def distance(self, query):
        """(M,) distance from each query pose to the nearest boundary voxel [m].

        query : (M,2) if planar else (M,3).

        Returns +inf where there are no boundaries at all. That is deliberate and
        must stay: occlusion_stage_cost compares d < r_keep, so +inf yields zero
        cost, i.e. "nothing is hidden, this term is silent". Returning 0.0 instead
        would make an empty map look maximally dangerous and freeze the ego.
        """
        q = self._prep(query)
        if len(q) == 0:
            return np.empty(0)

        # +inf is returned ONLY when there genuinely are no boundaries. It must
        # never be returned because the distance could not be computed: a missing
        # SciPy once silently disabled the entire keep-out (d=inf => cost 0) and
        # the drone flew straight through a shadow with 15288 boundary voxels
        # loaded. "Cannot compute" falls back to brute force instead.
        if len(self.points) == 0:
            return np.full(len(q), np.inf)
        if self._tree is None:
            return self.distance_bruteforce(q)
        d, _ = self._tree.query(q)
        return d

    def nearest(self, query):
        """(d, idx) -- distance AND the index of the nearest boundary voxel.

        Same query as distance(), but keeps the index the KD-tree already returns
        and distance() throws away. `self.points[idx]` is then the (x,y,z) world
        point, which is what a visualisation needs to draw the sightline the cost
        is actually reacting to.

        idx is -1 wherever d is +inf (no boundaries at all), so a caller must check
        before indexing. Deliberately NOT folded into distance(): that one runs
        K x H times per plan cycle and has no use for the index.
        """
        q = self._prep(query)
        if len(q) == 0:
            return np.empty(0), np.empty(0, dtype=int)
        if len(self.points) == 0:
            return np.full(len(q), np.inf), np.full(len(q), -1, dtype=int)
        if self._tree is None:
            return self._nearest_bruteforce(q)
        d, i = self._tree.query(q)
        return d, np.asarray(i, dtype=int)

    def distance_bruteforce(self, query):
        """SciPy-free fallback. O(M*N) -- only for tests or tiny boundary sets."""
        q = self._prep(query)
        if len(self.points) == 0:
            return np.full(len(q), np.inf)
        # Chunked so a K x H query against a large boundary set cannot allocate
        # a multi-GB intermediate.
        out = np.empty(len(q))
        step = max(1, int(4e6 // max(len(self._xy), 1)))
        for i in range(0, len(q), step):
            diff = q[i:i + step, None, :] - self._xy[None, :, :]
            out[i:i + step] = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
        return out

    def _nearest_bruteforce(self, q):
        """distance_bruteforce, keeping the argmin as well. Same chunking."""
        out = np.empty(len(q))
        idx = np.empty(len(q), dtype=int)
        step = max(1, int(4e6 // max(len(self._xy), 1)))
        for i in range(0, len(q), step):
            diff = q[i:i + step, None, :] - self._xy[None, :, :]
            dd = np.sqrt((diff ** 2).sum(axis=2))
            idx[i:i + step] = np.argmin(dd, axis=1)
            out[i:i + step] = dd.min(axis=1)
        return out, idx


class OccupancySet:
    """Occupied voxels as a membership test, not a distance field.

    `inside(p)` answers only "does p fall in an occupied cell". Kept separate from
    the occlusion keep-out because that one grows with horizon time (the hidden
    agent moves) and a wall does not.

    Voxels are quantised to integer indices packed into one int64 key, so a K x H
    batch costs one searchsorted instead of a nearest-neighbour query.
    """

    _MASK = (1 << 21) - 1
    _BIAS = 1 << 20      # keeps negative world coordinates inside the mask

    def __init__(self, points, resolution=0.1, planar=True, z_band=None, ego_z=None):
        """planar : a query is occupied when ANY occupied voxel shares its (x,y)
        column, matching the 2D rollout, which carries no z.

        z_band : keep only voxels with |z - ego_z| <= z_band. NOT optional in
        practice when planar=True. The floor is occupied (MARSIM scenes have a
        ground plane, and virtual_ground_height marks everything below it
        occupied too), so folding z away without a band makes every column in
        the map solid -- every rollout then collides, every cost is identical,
        the softmax goes uniform and the drone freezes in place. Band it to
        roughly the drone's half-height so only voxels at flight altitude count.
        """
        self.resolution = float(resolution)
        self.planar = planar
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if z_band is not None and ego_z is not None and len(pts):
            pts = pts[np.abs(pts[:, 2] - ego_z) <= z_band]
        self.points = pts
        self._pts_q = pts[:, :2] if planar else pts
        self._keys = np.unique(self._encode(self._pts_q))
        # Built on first nearest() call, not here: inf_occ routinely carries tens
        # of thousands of points and the planner never needs a distance -- its
        # collision term is pure membership. Only the viz asks.
        self._tree = None

    def __len__(self):
        return len(self._keys)

    def _prep(self, query):
        """Validate a query against the dimension the keys were built at.

        A mismatch here used to be silent and total: _encode only mixes z in when
        the array has three columns, so a 2D query against a 3D set produces keys
        that can never match any stored key and inside() returns False for every
        pose, wall or no wall. Raise instead -- "no collision ever" is not a
        failure anyone reads as a bug.
        """
        q = np.asarray(query, dtype=float)
        if q.ndim != 2:
            raise ValueError("query must be (M,2) or (M,3), got shape %s" % (q.shape,))
        want = 2 if self.planar else 3
        if q.shape[1] != want:
            raise ValueError(
                "OccupancySet(planar=%s) needs (M,%d) queries, got (M,%d). The "
                "caller is working in a different dimension than the map."
                % (self.planar, want, q.shape[1]))
        return q

    def nearest(self, query):
        """(d, idx) -- distance and index of the nearest OCCUPIED voxel centre.

        Distance to the obstacle surface, as opposed to inside()'s binary "am I in
        it". Purely diagnostic: the collision cost is a membership test on purpose
        (penetration depth is not a useful gradient), so nothing in the planner
        consumes this.

        +inf / -1 when the map is empty, matching BoundarySet.nearest.
        """
        q = self._prep(query)
        if len(q) == 0:
            return np.empty(0), np.empty(0, dtype=int)
        if len(self.points) == 0:
            return np.full(len(q), np.inf), np.full(len(q), -1, dtype=int)
        if self._tree is None:
            if not _HAS_SCIPY:
                d = np.sqrt(((q[:, None, :] - self._pts_q[None, :, :]) ** 2).sum(2))
                return d.min(axis=1), np.argmin(d, axis=1)
            self._tree = cKDTree(self._pts_q)
        d, i = self._tree.query(q)
        return d, np.asarray(i, dtype=int)

    def _encode(self, pts):
        if len(pts) == 0:
            return np.empty(0, dtype=np.int64)
        # ROG-Map centres its voxels at (i + 0.5) * res -- published coordinates
        # are 0.45, 10.05, -13.25 and so on, never multiples of res. Plain floor
        # therefore maps each centre to its own cell and keeps centres away from
        # floor()'s discontinuity.
        #
        # Do NOT re-add the +0.5 that used to be here: on half-offset data it put
        # every centre exactly on the discontinuity, collapsing 12871 voxels onto
        # 5073 keys and opening holes straight through a wall, so inside_segment
        # returned False for a segment crossing 10k occupied voxels.
        idx = np.floor(np.asarray(pts, dtype=float) / self.resolution).astype(np.int64)
        idx = (idx + self._BIAS) & self._MASK
        key = (idx[:, 0] << 42) | (idx[:, 1] << 21)
        if idx.shape[1] == 3:
            key |= idx[:, 2]
        return key

    def inside(self, query):
        """(M,) bool -- True where the query pose lies in an occupied voxel.

        All-False on an empty map, for the same reason distance() returns +inf:
        an unobserved map must not read as solid.
        """
        q = self._prep(query)
        if len(q) == 0:
            return np.zeros(0, dtype=bool)
        if len(self._keys) == 0:
            return np.zeros(len(q), dtype=bool)
        k = self._encode(q)
        i = np.clip(np.searchsorted(self._keys, k), 0, len(self._keys) - 1)
        return self._keys[i] == k

    def inside_segment(self, p0, p1):
        """(M,) bool -- True where the segment p0->p1 crosses an occupied voxel.

        Testing only the endpoints tunnels: rollout poses are v*dt apart (0.2 m at
        2 m/s, 10 Hz), which is the same order as the inflated voxel size, so a
        wall one slab thick sits entirely between two consecutive samples and is
        never seen. Sub-sampling at half the voxel pitch makes that impossible.
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        if len(p0) == 0:
            return np.zeros(0, dtype=bool)

        step = self.resolution * 0.5
        longest = float(np.max(np.linalg.norm(p1 - p0, axis=1))) if len(p0) else 0.0
        n = int(np.ceil(longest / step)) if step > 0 else 1
        n = max(1, min(n, 16))          # cap: a runaway rollout must not blow up cost

        hit = self.inside(p1)
        for s in range(1, n):
            hit |= self.inside(p0 + (float(s) / n) * (p1 - p0))
        return hit

