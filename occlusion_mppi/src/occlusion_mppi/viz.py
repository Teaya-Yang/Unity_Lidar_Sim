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


def ego_occupied_markers(occupancy, pos, z, frame_id, stamp=None, ns="ego_occupied"):
    """Is the ego itself standing in an occupied voxel? Ball above the drone.

    This is the ground truth for the collision term. `hit` in the planner is
    OccupancySet.inside_segment over ROLLOUT poses; this runs the same membership
    test on the MEASURED pose, so it answers a question the rollout costs cannot:
    whether the wall the drone is visibly inside of is in the occupancy set at all.

    Red ball  => the ego is inside an occupied voxel. The map has the obstacle and
                 the planner drove in anyway.
    Green ball => the ego is in free space. If the drone is visibly inside a wall
                 and this is still green, the occupancy set does not contain that
                 wall -- an empty/stale inf_occ or a build/query mismatch -- and no
                 amount of collision weight will ever stop it.

    Returns (MarkerArray, inside) so a caller can log the same boolean it draws.
    """
    q = np.asarray(pos, dtype=float)[None, :]
    inside = bool(len(occupancy)) and bool(occupancy.inside(q)[0])
    stamp = stamp if stamp is not None else rospy.Time.now()

    ball = Marker()
    ball.header.frame_id = frame_id
    ball.header.stamp = stamp
    ball.ns, ball.id = ns, 0
    ball.type, ball.action = Marker.SPHERE, Marker.ADD
    ball.pose.orientation.w = 1.0
    ball.pose.position.x, ball.pose.position.y = pos[0], pos[1]
    ball.pose.position.z = z + 0.9
    ball.scale.x = ball.scale.y = ball.scale.z = 0.4
    ball.color.a = 0.95
    ball.color.r = 1.0 if inside else 0.1
    ball.color.g = 0.1 if inside else 0.9
    ball.color.b = 0.1

    txt = Marker()
    txt.header.frame_id = frame_id
    txt.header.stamp = stamp
    txt.ns, txt.id = ns, 1
    txt.type, txt.action = Marker.TEXT_VIEW_FACING, Marker.ADD
    txt.pose.orientation.w = 1.0
    txt.pose.position.x, txt.pose.position.y = pos[0], pos[1]
    txt.pose.position.z = z + 1.3
    txt.scale.z = 0.3
    txt.color.a = 0.95
    txt.color.r, txt.color.g, txt.color.b = ball.color.r, ball.color.g, ball.color.b
    # occ=0 is a different failure from "free space", and they look identical on
    # a ball alone, so the count is part of the label.
    txt.text = ("IN OCCUPIED VOXEL" if inside else "free") + "  (occ=%d)" % len(occupancy)

    arr = MarkerArray()
    arr.markers.append(ball)
    arr.markers.append(txt)
    return arr, inside


def nearest_occupied_markers(occupancy, pos, z, frame_id, stamp=None,
                             ns="nearest_occupied"):
    """Segment ego -> nearest occupied voxel, with the distance as text.

    The obstacle-clearance counterpart to nearest_markers(): that one shows the
    keep-out the occlusion term reacts to, this one shows how far the wall
    actually is. Nothing in the planner consumes it -- the collision term is
    binary membership -- so it is the number to read when asking whether the
    collision test agrees with what you see.

    Returns (MarkerArray, d), d = +inf on an empty map.
    """
    d_arr, i_arr = occupancy.nearest(np.asarray(pos, dtype=float)[None, :])
    d, i = float(d_arr[0]), int(i_arr[0])
    stamp = stamp if stamp is not None else rospy.Time.now()

    line = Marker()
    line.header.frame_id = frame_id
    line.header.stamp = stamp
    line.ns, line.id = ns, 0
    line.type, line.action = Marker.LINE_LIST, Marker.ADD
    line.pose.orientation.w = 1.0
    line.scale.x = 0.04
    line.color.a, line.color.r, line.color.g, line.color.b = 0.95, 0.2, 0.6, 1.0

    txt = Marker()
    txt.header.frame_id = frame_id
    txt.header.stamp = stamp
    txt.ns, txt.id = ns, 1
    txt.type, txt.action = Marker.TEXT_VIEW_FACING, Marker.ADD
    txt.pose.orientation.w = 1.0
    txt.scale.z = 0.3
    txt.color.a, txt.color.r, txt.color.g, txt.color.b = 0.95, 0.2, 0.6, 1.0
    txt.text = ""

    if i >= 0 and np.isfinite(d):
        tgt = occupancy.points[i]
        tip_z = tgt[2] if not occupancy.planar else z
        line.points.append(Point(pos[0], pos[1], z))
        line.points.append(Point(tgt[0], tgt[1], tip_z))
        txt.pose.position.x = 0.5 * (pos[0] + tgt[0])
        txt.pose.position.y = 0.5 * (pos[1] + tgt[1])
        txt.pose.position.z = 0.5 * (z + tip_z) - 0.4
        txt.text = "d_occ_cell %.2f m" % d
    else:
        txt.pose.position.x, txt.pose.position.y = pos[0], pos[1]
        txt.pose.position.z = z - 0.4

    arr = MarkerArray()
    arr.markers.append(line)
    arr.markers.append(txt)
    return arr, d
