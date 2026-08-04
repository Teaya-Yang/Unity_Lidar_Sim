"""Occlusion keep-out cost.

PROVENANCE
----------
`capsule_radius` and `occlusion_stage_cost` are copied VERBATIM from
    Unity_Lidar_Sim/controller/occlusion_capsules.py
so the 3D controller is scored by exactly the same function as the validated 2D
one. Do not "improve" them here -- if they need to change, change them there and
re-copy, otherwise the two controllers silently diverge and any comparison
between them becomes meaningless.

They port unchanged because they are already dimension-agnostic: both take a
scalar/array distance `d` and never touch the geometry that produced it. Going
from a 2D segment to a 3D voxel set changes only HOW `d` is computed, which
happens in boundary.py, not here.
"""

import numpy as np


def capsule_radius(d_base: float, v_target: float, t_k: float) -> float:
    """Forward-reachable-set radius around the boundary segment at horizon time t_k.

    A hidden agent anywhere on the segment moving at up to v_target in an arbitrary
    direction can be anywhere within v_target*t_k of it, so the reachable set is the
    segment dilated by that radius — a capsule. d_base is the ego's own safety margin
    (D_SAFE_HARD), added on top.
    """
    return d_base + v_target * t_k


def occlusion_stage_cost(d, v, t_k, v_target, d_safe, w_obs, fmax=None, sqrt=None, clip=None,
                         w_sight=None, a_brake=None, v_floor=None, cost_current=None, dyn=False,
                         action=None, t_grow_max=None, w_soft=None, d_infl=None):
    """Occlusion stage cost. Backend-agnostic: pass numpy or CasADi primitives.

    d        : distance from the (rollout/predicted) pose to the nearest occlusion boundary
    v        : speed at this stage [m/s] — used only by the RSS sightline term below
    t_k      : horizon time of this stage [s]
    v_target : assumed max speed of the hidden agent [m/s]
    d_safe   : base (t=0) keep-out radius [m]
    w_obs    : hard-constraint weight
    t_grow_max : cap [s] on the expansion time, or None for the (freezing) uncapped growth
    w_soft   : weight of the soft influence ring outside r_keep, or None/0 to skip it
    d_infl   : width [m] of that ring. Cost is w_soft at d = r_keep, 0 at r_keep + d_infl
    Returns the scalar/array stage cost contribution.
    """
    fmax = np.maximum if fmax is None else fmax
    sqrt = np.sqrt if sqrt is None else sqrt

    t_eff = t_k if t_grow_max is None else min(t_k, t_grow_max)
    r_grow = v_target * t_eff
    r_keep = r_grow + d_safe

    # Hard term: binary, so every breach costs the same regardless of depth — a constraint,
    cost = np.where(d < r_keep, w_obs, 0.0)

    # Soft term: the gradient the hard term cannot give. A one-sided quadratic over an
    # influence ring of width d_infl OUTSIDE r_keep, normalised so w_soft is the cost of
    # sitting exactly on the keep-out boundary and 0 at r_keep + d_infl and beyond.
    if w_soft and d_infl:
        margin = fmax(0.0, (r_keep + d_infl) - d)
        cost = cost + w_soft * margin * margin

    return cost
