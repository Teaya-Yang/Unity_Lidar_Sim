"""
dynamic_obstacles.py
====================
Reads the moving-object footprints published by DynamicObstaclePublisher.cs so the
trajectory plot can draw where the traffic REALLY was at the solve it visualises.

Topic: /dynamic_obstacles, std_msgs/String carrying
    {"boxes":[{"cx":..,"cy":..,"sx":..,"sy":..,"yaw":..}, ...]}
in the controller's world axes — identical payload and frame to /static_obstacles, so
static_obstacles._parse / box_polygon are reused as-is. The only difference is that this
one streams: the controller snapshots boxes() each step and keeps the history.

Plot-only: nothing in the planner reads this. Runs without ROS — start() returns False
if rclpy isn't importable and the plot simply omits the objects.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from static_obstacles import _parse

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False


class DynamicObstacles:
    """
        do = DynamicObstacles(); do.start()
        B = do.boxes()      # (M,5) [cx, cy, sx, sy, yaw] as of the last message, or None
        do.shutdown()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._boxes = None
        self._node = None
        self._exec = None
        self._thread = None
        self._stop = threading.Event()

    def start(self, topic: str = "/dynamic_obstacles") -> bool:
        if not _HAS_RCLPY:
            print("[DynamicObstacles] rclpy not importable — dynamic obstacles disabled.")
            return False
        if not rclpy.ok():
            rclpy.init()
        outer = self

        class _Sub(Node):
            def __init__(self):
                super().__init__("dynamic_obstacles_reader")
                self.create_subscription(String, topic, self._cb, qos_profile_sensor_data)
                self._got = False

            def _cb(self, msg):
                boxes = _parse(msg.data)
                with outer._lock:
                    outer._boxes = boxes
                if not self._got and boxes is not None:
                    self._got = True
                    print(f"[DynamicObstacles] received {len(boxes)} footprints on '{topic}'")

        self._node = _Sub()
        # OWN executor — see the same note in static_obstacles.py.
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        print(f"[DynamicObstacles] subscribed '{topic}'")
        return True

    def _spin(self) -> None:
        while not self._stop.is_set():
            try:
                self._exec.spin_once(timeout_sec=0.1)
            except Exception:
                break

    def boxes(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._boxes is None else self._boxes.copy()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._exec is not None:
            self._exec.shutdown()
            self._exec = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
