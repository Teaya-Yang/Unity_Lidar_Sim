"""
static_obstacles.py
===================
Reads the static-obstacle footprints published by StaticObstaclePublisher.cs so the
trajectory plot can draw the real walls/buildings.

Topic: /static_obstacles, std_msgs/String carrying
    {"boxes":[{"cx":..,"cy":..,"sx":..,"sy":..,"yaw":..}, ...]}
in the controller's world axes (a0 = Unity Z, a1 = Unity X, yaw = +Unity eulerY) — the
same frame TaxiAgent reports the ego/goal in and the LiDAR returns are placed in, so no
conversion happens here. Note this is NOT the ROS frame /ground_truth/agents uses, whose
y is -Unity x.

Plot-only: nothing in the planner reads this. Runs without ROS — start() returns False
if rclpy isn't importable and the plot simply omits the obstacles.
"""

from __future__ import annotations

import json
import threading
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False


class StaticObstacles:
    """
        so = StaticObstacles(); so.start()
        B = so.boxes()      # (M,5) [cx, cy, sx, sy, yaw], or None
        so.shutdown()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._boxes = None
        self._node = None
        self._exec = None
        self._thread = None
        self._stop = threading.Event()

    def start(self, topic: str = "/static_obstacles") -> bool:
        if not _HAS_RCLPY:
            print("[StaticObstacles] rclpy not importable — static obstacles disabled.")
            return False
        if not rclpy.ok():
            rclpy.init()
        outer = self

        class _Sub(Node):
            def __init__(self):
                super().__init__("static_obstacles_reader")
                self.create_subscription(String, topic, self._cb, qos_profile_sensor_data)
                self._got = False

            def _cb(self, msg):
                boxes = _parse(msg.data)
                if boxes is None:
                    return
                with outer._lock:
                    outer._boxes = boxes
                if not self._got:
                    self._got = True
                    print(f"[StaticObstacles] received {len(boxes)} footprints on '{topic}'")

        self._node = _Sub()
        # OWN executor: rclpy.spin() drives the process-wide default one, which
        # ObstacleCircles is already spinning on its own thread — two threads inside one
        # executor raise "generator already executing".
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        print(f"[StaticObstacles] subscribed '{topic}'")
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


def _parse(payload: str) -> Optional[np.ndarray]:
    try:
        d = json.loads(payload)
    except json.JSONDecodeError:
        return None
    rows = [(b["cx"], b["cy"], b["sx"], b["sy"], b.get("yaw", 0.0))
            for b in d.get("boxes", [])]
    return np.asarray(rows, float).reshape(-1, 5) if rows else None


def box_polygon(cx: float, cy: float, sx: float, sy: float, yaw: float = 0.0) -> np.ndarray:
    """Closed (5,2) outline of one footprint, for plotting."""
    hx, hy = 0.5 * sx, 0.5 * sy
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]])
    c, s = np.cos(yaw), np.sin(yaw)
    return corners @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])
