"""
detection_source.py
===================
Subscriber for /detections from lidar_detector_node.py, shaped to be a drop-in
replacement for ObstacleCircles.clusters().

The point of matching that signature is that the detector does NOT get its own
tracking, gating or cost path: it swaps only the MEASUREMENT that
DynamicClusterTracker consumes. Everything downstream — the Kalman filter, the
static/dynamic test, select_nearest, occlusion_stage_cost — is identical between the
two modes, so a difference in behaviour is attributable to the measurement and
nothing else.

Row format on the wire (see lidar_detector_node): [c0, c1, r, score, vx, vy, yaw, cls],
world (a0, a1). clusters() returns the first three plus a synthetic point count,
which is the (M,4) the tracker expects.

A no-op when rclpy is unavailable, matching how ObstacleCircles degrades.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False

ROW = 8


class DetectionSource:

    TOPIC = "/detections"

    def __init__(self, max_age: float = 3.0):
        # Larger than ObstacleCircles' 1.5 s: inference takes ~1 s on CPU on top of the
        # ~1 Hz publish rate, so a detection is legitimately older than a cluster is.
        self.max_age = float(max_age)
        self._lock = threading.Lock()
        self._rows: Optional[np.ndarray] = None
        self._stamp = 0.0
        self._node = None
        self._thread = None
        self._n = 0

    def start(self, topic: str = TOPIC) -> bool:
        if not _HAS_RCLPY:
            print("[DetectionSource] rclpy not importable — detector input disabled.")
            return False
        if not rclpy.ok():
            rclpy.init()
        outer = self

        class _Sub(Node):
            def __init__(self):
                super().__init__("detection_source")
                self.create_subscription(Float32MultiArray, topic, self._cb, 10)

            def _cb(self, msg):
                data = np.asarray(msg.data, dtype=float)
                if not len(data):
                    return
                n = int(data[0])
                rows = (data[1:1 + n * ROW].reshape(n, ROW) if n > 0
                        else np.zeros((0, ROW)))
                with outer._lock:
                    outer._rows = rows
                    outer._stamp = time.monotonic()
                if outer._n == 0:
                    print(f"[DetectionSource] first message on '{topic}' ({n} detections)")
                outer._n += 1

        self._node = _Sub()
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,),
                                        daemon=True)
        self._thread.start()
        print(f"[DetectionSource] subscribed '{topic}'")
        return True

    @property
    def ready(self) -> bool:
        with self._lock:
            return (self._rows is not None
                    and (time.monotonic() - self._stamp) <= self.max_age)

    @property
    def stamp(self) -> float:
        """Arrival time of the newest message. The controller drives the tracker off
        CHANGES to this, exactly as it does off ObstacleCircles.stamp."""
        with self._lock:
            return self._stamp

    def clusters(self, **_ignored) -> Optional[np.ndarray]:
        """(M,4) [c0, c1, r, n_points] — the DynamicClusterTracker contract.

        Keyword arguments are accepted and ignored so this is call-compatible with
        ObstacleCircles.clusters(cell=..., min_points=..., max_radius=...). Those are
        grid-clustering parameters with no meaning for a detector: the extent comes
        from the regressed box, and scenery is rejected by the class head rather than
        by dynamic_clusters.max_radius.
        """
        if not self.ready:
            return None
        with self._lock:
            rows = self._rows
        if rows is None or not len(rows):
            return None
        # min_points gates raw returns; a detection has no point count, so a constant
        # above any plausible threshold keeps the tracker's test a no-op here.
        n = np.full(len(rows), 999.0)
        return np.column_stack([rows[:, 0], rows[:, 1], rows[:, 2], n])

    def extras(self) -> Optional[np.ndarray]:
        """(M,5) [score, vx, vy, yaw, cls] aligned with clusters()' rows.

        Not consumed yet. This is the signal that makes an ANISOTROPIC keep-out
        possible — a regressed yaw and velocity rather than a centroid difference —
        and it is carried through the wire format so the cost side can be changed
        without touching this node or the detector.
        """
        if not self.ready:
            return None
        with self._lock:
            rows = self._rows
        return None if rows is None or not len(rows) else rows[:, 3:8]

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
