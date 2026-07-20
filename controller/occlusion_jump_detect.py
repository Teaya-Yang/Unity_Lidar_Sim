#!/usr/bin/env python3
"""
Range-discontinuity ("jump") occlusion detector — diagnostic subscriber.

Subscribes to the Unity LiDAR cloud and prints, per scan, the occlusion edges
found by comparing adjacent beams in azimuth. Where range steps
discontinuously between neighbouring beams, one beam is grazing an occluder's
silhouette and the next has sailed past it: that step is a blind-corner mouth.

This is INSTANTANEOUS and in BEAM SPACE — one scan, no map, no accumulation.
It is a different signal from the grid/line-of-sight shadow field in
lidar_costmap.py, which accumulates UNKNOWN volume in world space. This script
is a diagnostic and does not feed the controller.

Requires the ORDERED cloud: unlike LidarCostmap._parse_cloud, non-returns must
NOT be dropped, because beam adjacency is the whole basis of the method (and an
object->sky transition is itself a real edge). Pass the scan geometry so the
flat cloud can be reshaped back into its (azimuth x elevation) grid; it must
match the PointCloudPublisher Inspector fields.

Usage:
    python3 occlusion_jump_detect.py --fov-h 360 --fov-v 45 --res-h 1 --res-v 1
"""

import argparse
import math
import sys

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
except ImportError:
    sys.exit("rclpy / sensor_msgs not importable — source your ROS 2 setup first.")


def scan_shape(fov_h, fov_v, res_h, res_v):
    """Beam-grid dimensions, mirroring the arithmetic in LaserSensor3D's constructor
    (including its 360-degree wrap-around de-duplication)."""
    n_h = int(math.floor(fov_h / res_h)) + 1
    n_v = int(math.floor(fov_v / res_v)) + 1
    if fov_h == 360:
        n_h -= 1
    return n_h, n_v


class JumpDetector(Node):
    def __init__(self, args):
        super().__init__("occlusion_jump_detect")
        self.n_h, self.n_v = scan_shape(args.fov_h, args.fov_v, args.res_h, args.res_v)
        self.res_h = args.res_h
        self.max_range = args.max_range
        self.min_jump = args.min_jump
        self.grazing = math.radians(args.grazing_deg)
        self.sigma = args.sigma
        self.warned_shape = False

        # Sensor-data QoS: the ros_tcp_endpoint bridge publishes streamed sensor topics
        # best-effort, and a default RELIABLE subscriber would silently receive nothing.
        self.create_subscription(PointCloud2, args.topic, self._cloud, qos_profile_sensor_data)
        print(f"[jump] subscribed '{args.topic}'  expecting {self.n_h}x{self.n_v} "
              f"= {self.n_h * self.n_v} beams")

    def _ranges(self, msg):
        """Flat cloud -> (n_h, n_v) range grid, non-returns kept as max_range."""
        step = msg.point_step
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        n = len(raw) // step
        if n == 0:
            return None
        xyz = raw[:n * step].reshape(n, step)[:, :12].copy().view(np.float32).reshape(n, 3)

        if n != self.n_h * self.n_v:
            if not self.warned_shape:
                self.warned_shape = True
                print(f"[jump] ERROR: got {n} points, expected {self.n_h * self.n_v}. "
                      f"The --fov/--res flags must match the PointCloudPublisher "
                      f"Inspector fields. Ignoring scans until they do.")
            return None

        rng = np.linalg.norm(xyz, axis=1)
        # A non-return is a beam that reached max range without hitting anything —
        # meaningful for jump detection, so substitute rather than discard.
        rng[~np.isfinite(rng)] = self.max_range
        # LaserSensor3D fills elevation in the inner loop, azimuth in the outer.
        return rng.reshape(self.n_h, self.n_v)

    def _cloud(self, msg):
        rng = self._ranges(msg)
        if rng is None:
            return

        near = np.minimum(rng[:-1, :], rng[1:, :])
        delta = np.abs(np.diff(rng, axis=0))

        # A continuous surface at range r spans r*dtheta/tan(lambda) between adjacent
        # beams; anything beyond that is a genuine depth step, not a receding surface.
        # Scaling with range is what keeps the test valid at distance, where the same
        # physical gap subtends fewer beams.
        thresh = near * math.radians(self.res_h) / math.tan(self.grazing) + 3.0 * self.sigma
        thresh = np.maximum(thresh, self.min_jump)

        edges = delta > thresh
        if not edges.any():
            print("[jump] no occlusion edge")
            return

        ai, vj = np.where(edges)
        # Report the deepest few: the largest steps hide the most.
        order = np.argsort(delta[ai, vj])[::-1][:5]
        print(f"[jump] OCCLUSION: {edges.sum()} edge beams "
              f"({len(np.unique(ai))} distinct azimuths)")
        for k in order:
            i, j = ai[k], vj[k]
            r_near, r_far = rng[i, j], rng[i + 1, j]
            # Sign tells which side the shadow opens toward: the nearer endpoint is the
            # corner point the hidden agent would round.
            side = "right" if r_far > r_near else "left"
            az = -self.fov_h_half() + i * self.res_h
            print(f"         az={az:+7.1f} deg  {min(r_near, r_far):6.2f} -> "
                  f"{max(r_near, r_far):6.2f} m  (step {delta[i, j]:5.2f} m, "
                  f"opens {side})")

    def fov_h_half(self):
        return (self.n_h * self.res_h) / 2.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topic", default="/point_cloud")
    p.add_argument("--fov-h", type=float, default=360.0, help="must match Inspector")
    p.add_argument("--fov-v", type=float, default=45.0, help="must match Inspector")
    p.add_argument("--res-h", type=float, default=1.0, help="must match Inspector")
    p.add_argument("--res-v", type=float, default=1.0, help="must match Inspector")
    p.add_argument("--max-range", type=float, default=1000.0, help="must match Inspector")
    p.add_argument("--grazing-deg", type=float, default=12.0,
                   help="most oblique surface treated as continuous; lower = fewer edges")
    p.add_argument("--min-jump", type=float, default=0.5,
                   help="absolute floor on a reportable step, metres")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="range noise stddev; 0 is correct for noiseless raycasting")
    args = p.parse_args()

    rclpy.init()
    node = JumpDetector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


