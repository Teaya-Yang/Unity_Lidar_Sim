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

    def __init__(self, points, z_band=None, ego_z=None, planar=True):
        """points : (N,3) world-frame boundary voxel centres.

        z_band  : if not None, keep only voxels with |z - ego_z| <= z_band. Use this
                  when flying at fixed altitude so the 2D cost stays valid -- a
                  boundary 4 m overhead is not something the ego can collide with
                  laterally, and including it inflates the keep-out for no reason.
        planar  : if True the distance is computed in XY only (matching the 2D
                  controller). Set False for a genuinely 3D keep-out.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if z_band is not None and ego_z is not None and len(pts):
            pts = pts[np.abs(pts[:, 2] - ego_z) <= z_band]

        self.planar = planar
        self.points = pts
        self._xy = pts[:, :2] if planar else pts

        self._tree = cKDTree(self._xy) if (len(pts) and _HAS_SCIPY) else None

    def __len__(self):
        return len(self.points)

    def distance(self, query):
        """(M,) distance from each query pose to the nearest boundary voxel [m].

        query : (M,2) if planar else (M,3).

        Returns +inf where there are no boundaries at all. That is deliberate and
        must stay: occlusion_stage_cost compares d < r_keep, so +inf yields zero
        cost, i.e. "nothing is hidden, this term is silent". Returning 0.0 instead
        would make an empty map look maximally dangerous and freeze the ego.
        """
        q = np.asarray(query, dtype=float)
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

    def distance_bruteforce(self, query):
        """SciPy-free fallback. O(M*N) -- only for tests or tiny boundary sets."""
        q = np.asarray(query, dtype=float)
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
