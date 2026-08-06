"""Shared RViz markers for the occlusion geometry.

Lives here rather than in either node so mppi_node.py and occlusion_viz.py cannot
drift apart: the point of the standalone viz is that what it draws is what the
planner reacts to, which stops being true the moment the drawing is duplicated.

No geometry is computed here -- BoundarySet.nearest stays the single source of the
distance. This module only turns its result into markers.
"""

import numpy as np
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def nearest_markers(boundaries, pos, z, d_safe, frame_id, d_now, i_now,
                    stamp=None):
    """Arrow from the ego to the boundary voxel that sets its d_occ.

    This is the vector behind the single number the occlusion cost consumes:
    BoundarySet.distance() reports the nearest voxel and nothing else, so exactly
    one voxel in the whole map is responsible for the ego's keep-out cost at any
    instant. Drawing it turns "why did it stop / swerve" into a question with a
    visible answer.

    Two things it makes obvious that no other viz does:
      * an arrow pointing THROUGH a wall is the Euclidean-not-geodesic limitation
        biting -- the nearest voxel is one no agent can emerge from without going
        around;
      * an arrow that jumps across the map between ticks means the nearest voxel is
        flickering, i.e. two voxels are near-tied and the gradient is chattering.

    Colour is the verdict, not decoration: red once d < r_keep(t=0), i.e. the ego is
    inside the keep-out and the hard term is already charging.

    d_now / i_now come from BoundarySet.nearest(); passing them in rather than
    re-querying keeps the planner's one KD-tree lookup per tick.
    """
    arr = MarkerArray()
    stamp = stamp if stamp is not None else rospy.Time.now()

    # No boundaries (or none in the z_band) => nothing to draw, but the stale arrow
    # from the last tick must go, or it reads as a live measurement.
    if not np.isfinite(d_now) or i_now < 0 or i_now >= len(boundaries):
        for ns, mid in (("nearest", 0), ("nearest_text", 1)):
            m = Marker()
            m.header.frame_id = frame_id
            m.ns, m.id, m.action = ns, mid, Marker.DELETE
            arr.markers.append(m)
        return arr

    tgt = boundaries.points[i_now]
    # r_keep at t=0: the radius the ego is judged against RIGHT NOW. Later horizon
    # slices are larger, but those apply to future rollout poses.
    r_now = d_safe
    breached = d_now < r_now

    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns, m.id = "nearest", 0
    m.type, m.action = Marker.ARROW, Marker.ADD
    m.pose.orientation.w = 1.0
    # Two-point ARROW: scale is (shaft dia, head dia, head length), NOT a length --
    # the length comes from the points themselves.
    m.scale.x, m.scale.y, m.scale.z = 0.06, 0.16, 0.25
    m.color.a = 0.95
    m.color.r = 1.0 if breached else 0.1
    m.color.g = 0.1 if breached else 0.9
    m.color.b = 0.1
    # Tip at the voxel's true z when the set is 3D, flattened to flight altitude
    # when it is planar -- either way the arrow's drawn length is the number in
    # the label, which is the whole point of showing them together.
    tip_z = tgt[2] if not boundaries.planar else z
    m.points.append(Point(pos[0], pos[1], z))
    m.points.append(Point(tgt[0], tgt[1], tip_z))
    arr.markers.append(m)

    t = Marker()
    t.header.frame_id = frame_id
    t.header.stamp = stamp
    t.ns, t.id = "nearest_text", 1
    t.type, t.action = Marker.TEXT_VIEW_FACING, Marker.ADD
    t.pose.orientation.w = 1.0
    t.pose.position.x = 0.5 * (pos[0] + tgt[0])
    t.pose.position.y = 0.5 * (pos[1] + tgt[1])
    t.pose.position.z = 0.5 * (z + tip_z) + 0.4
    t.scale.z = 0.35
    t.color.a = 0.95
    t.color.r, t.color.g, t.color.b = m.color.r, m.color.g, m.color.b
    t.text = "d=%.2f / r=%.2f" % (d_now, r_now)
    arr.markers.append(t)

    return arr
