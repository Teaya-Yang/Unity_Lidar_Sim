#!/usr/bin/env python3
"""
lidar_detector_node.py
======================
PointPillars 3D object detection on the Unity LiDAR cloud, published as an
alternative measurement source to the grid clustering in dynamic_clusters.py.

Runs in its OWN process and its OWN interpreter (the `lidar3d` conda env, which has
torch + mmdet3d), not in the controller's venv. The controller only ever sees the
Float32MultiArray this node publishes, so it never imports torch.

Topics
  in   /point_cloud        sensor_msgs/PointCloud2   sensor frame, ROS xyz
       /laser_scan_pose    geometry_msgs/Pose        ego pose, (z, x) -> world (a0, a1)
  out  /detections         std_msgs/Float32MultiArray
       /viz/detections     visualization_msgs/MarkerArray   wireframe boxes for RViz

/detections payload: [n_rows] ++ [c0, c1, r, score, vx, vy, yaw, cls] * n_rows,
in controller world (a0, a1) — the same frame obstacle_circles.py emits, so the
controller can hand the first three columns straight to DynamicClusterTracker.

FRAMES. The network runs in the SENSOR frame, which is what it was trained in.
Boxes come back there and are converted on the way out, matching obstacle_circles:
    a0 = x_ros + ego0        a1 = -y_ros + ego1        yaw_world = -yaw_ros
The a1 flip is a reflection, hence the sign flip on yaw and on the velocity's y.

THE WEIGHTS ARE NOT FINE-TUNED. They are the upstream nuScenes checkpoint — road
scenes, a different beam pattern, and a head whose output convs were re-initialised
for this class set. Detections are therefore NOT trustworthy yet; this node exists
so the plumbing and the RViz overlay can be exercised end to end before training.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray

ROW = 8          # floats per detection row in /detections


# ── Cloud parsing ─────────────────────────────────────────────────────────────

def parse_cloud(msg) -> Optional[np.ndarray]:
    """PointCloud2 -> (N,4) [x, y, z, intensity] in the sensor frame.

    Mirrors ObstacleCircles._parse_cloud but keeps the 4th channel: the network's
    pillar encoder is built for 4 input features and a zero column is not the same
    thing as the intensity it was trained on.
    """
    step = msg.point_step
    if step < 12:
        return None
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n = len(raw) // step
    if n == 0:
        return None
    rows = raw[:n * step].reshape(n, step)
    xyz = rows[:, :12].copy().view(np.float32).reshape(n, 3)
    if step >= 16:
        inten = rows[:, 12:16].copy().view(np.float32).reshape(n, 1)
    else:
        inten = np.zeros((n, 1), dtype=np.float32)
    pts = np.hstack([xyz, inten])
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts if len(pts) else None


# ── Detector ──────────────────────────────────────────────────────────────────

class PointPillarsDetector:
    """Thin wrapper over the mmdet3d model. Torch, CPU, single scan at a time."""

    def __init__(self, config: str, checkpoint: Optional[str], score_thr: float,
                 threads: int = 8):
        import torch
        from mmengine.config import Config
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules
        from mmdet3d.structures import LiDARInstance3DBoxes

        register_all_modules()
        torch.set_num_threads(threads)
        self._torch = torch
        self._box_type = LiDARInstance3DBoxes

        cfg = Config.fromfile(config)
        if score_thr is not None:
            cfg.model.test_cfg.pts.score_thr = float(score_thr)
        self.class_names = list(cfg.get("class_names", []))
        # Ground-plane alignment: the network keys hard on height above ground, and its
        # anchors sit at a fixed z relative to the sensor. Unity's laser_link is higher
        # than the nuScenes LiDAR it was trained with, so the cloud is lowered to match.
        self.z_offset = float(cfg.get("z_offset", 0.0))
        veh = cfg.get("vehicle_classes", None)
        self.vehicle_ids = (None if not veh else
                            {self.class_names.index(c) for c in veh
                             if c in self.class_names})
        self.model = MODELS.build(cfg.model).eval()
        if self.z_offset:
            print(f"[detector] z offset {self.z_offset:+.2f} m applied to the cloud "
                  f"(ground-plane alignment)")

        # load_from is written repo-relative, but this node runs with cwd=controller/.
        # Resolve against the repo root (the config's parent directory) so the launch
        # file and a bare `python lidar_detector_node.py` both work.
        ckpt = checkpoint or cfg.get("load_from")
        if ckpt and not os.path.isabs(ckpt) and not os.path.exists(ckpt):
            repo = os.path.dirname(os.path.dirname(os.path.abspath(config)))
            ckpt = os.path.join(repo, ckpt)
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu")
            sd = sd.get("state_dict", sd)
            msd = self.model.state_dict()
            keep = {k: v for k, v in sd.items()
                    if k in msd and msd[k].shape == v.shape}
            self.model.load_state_dict(keep, strict=False)
            print(f"[detector] loaded {len(keep)}/{len(msd)} tensors from {ckpt}")
            if len(keep) < len(msd):
                print(f"[detector] {len(msd) - len(keep)} re-initialised "
                      f"(head output convs — class count differs from the checkpoint)")
        else:
            print("[detector] NO CHECKPOINT — random weights, output is noise")

    def __call__(self, pts: np.ndarray):
        """(N,4) sensor-frame points -> (M,9) boxes [x,y,z,w,l,h,yaw,vx,vy], (M,) score,
        (M,) label. All still in the sensor frame."""
        from mmdet3d.structures import Det3DDataSample
        torch = self._torch

        if self.z_offset:
            pts = pts.copy()
            pts[:, 2] += self.z_offset
        t = torch.from_numpy(np.ascontiguousarray(pts, dtype=np.float32))
        ds = Det3DDataSample()
        ds.set_metainfo({"box_type_3d": self._box_type})
        data = {"inputs": {"points": [t]}, "data_samples": [ds]}
        with torch.no_grad():
            data = self.model.data_preprocessor(data, False)
            res = self.model.predict(data["inputs"], data["data_samples"])[0]
        inst = res.pred_instances_3d
        return (inst.bboxes_3d.tensor.numpy(),
                inst.scores_3d.numpy(),
                inst.labels_3d.numpy())


# ── RViz markers ──────────────────────────────────────────────────────────────

def _pt(a0: float, a1: float, z: float = 0.0) -> Point:
    """Controller world (a0, a1) -> RViz map point. Same convention as rviz_viz.py."""
    return Point(x=float(a0), y=float(-a1), z=float(z))


def _box_edges(c0, c1, yaw, length, width, z_lo, z_hi):
    """Wireframe of one oriented box as a LINE_LIST point pair sequence, in world."""
    hl, hw = 0.5 * length, 0.5 * width
    corners = np.array([[+hl, +hw], [+hl, -hw], [-hl, -hw], [-hl, +hw]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    w = corners @ rot.T + np.array([c0, c1])

    pts = []
    for z in (z_lo, z_hi):                       # two horizontal rings
        for i in range(4):
            pts += [_pt(*w[i], z), _pt(*w[(i + 1) % 4], z)]
    for i in range(4):                           # four verticals
        pts += [_pt(*w[i], z_lo), _pt(*w[i], z_hi)]
    return pts


class DetectorNode(Node):

    def __init__(self, args):
        super().__init__("lidar_detector")
        self.det = PointPillarsDetector(args.config, args.checkpoint,
                                        args.score_thr, args.threads)
        self.min_score = float(args.score_thr)

        self._lock = threading.Lock()
        self._pts = None
        self._pose = None
        self._stamp = 0.0
        self._busy = False
        self._n_scans = 0

        qos = qos_profile_sensor_data
        self.create_subscription(PointCloud2, args.lidar_topic, self._cloud_cb, qos)
        self.create_subscription(Pose, args.pose_topic, self._pose_cb, qos)
        self.pub = self.create_publisher(Float32MultiArray, args.out_topic, 10)
        self.viz = self.create_publisher(MarkerArray, args.viz_topic, 10)
        self.create_timer(0.2, self._tick)

        print(f"[detector] cloud='{args.lidar_topic}' pose='{args.pose_topic}' "
              f"-> '{args.out_topic}' + '{args.viz_topic}'  score_thr={self.min_score}")

    def _cloud_cb(self, msg):
        pts = parse_cloud(msg)
        if pts is None:
            return
        with self._lock:
            self._pts = pts
            self._stamp = time.monotonic()
        if self._n_scans == 0:
            print(f"[detector] first cloud: {len(pts)} pts")

    def _pose_cb(self, msg):
        p = msg.position
        with self._lock:
            self._pose = (float(p.z), float(p.x))

    def _tick(self):
        """Inference is far slower than the timer, so drop scans rather than queue them:
        a stale detection is worse than a missing one — the tracker predicts through
        gaps, but it cannot undo a box placed where the object no longer is."""
        if self._busy:
            return
        with self._lock:
            pts, pose, stamp = self._pts, self._pose, self._stamp
            self._pts = None
        if pts is None or pose is None:
            return

        self._busy = True
        try:
            t0 = time.perf_counter()
            boxes, scores, labels = self.det(pts)
            dt = time.perf_counter() - t0
            self._n_scans += 1

            keep = scores >= self.min_score
            if self.det.vehicle_ids is not None:
                # Only vehicle-like classes become obstacles. traffic_cone / barrier /
                # pedestrian are the classes a road-trained net most readily hallucinates
                # on apron clutter, and they are not what the planner is avoiding.
                keep &= np.isin(labels, list(self.det.vehicle_ids))
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # Boxes come back in the z-shifted frame the network saw; undo it so the
            # markers line up with the actual cloud in RViz.
            if self.det.z_offset and len(boxes):
                boxes = boxes.copy()
                boxes[:, 2] -= self.det.z_offset
            rows = self._to_world(boxes, scores, labels, pose)
            self._publish(rows)
            self._publish_markers(boxes, labels, pose)
            names = [self.det.class_names[int(i)] if int(i) < len(self.det.class_names)
                     else str(int(i)) for i in labels]
            hist = {n: names.count(n) for n in sorted(set(names))}
            print(f"[detector] scan {self._n_scans}: {len(pts)} pts -> "
                  f"{len(rows)} det in {1000*dt:.0f} ms  {hist}")
        finally:
            self._busy = False

    @staticmethod
    def _to_world(boxes, scores, labels, pose):
        """(M,9) sensor-frame boxes -> (M,8) world rows [c0,c1,r,score,vx,vy,yaw,cls]."""
        if not len(boxes):
            return np.zeros((0, ROW))
        ego0, ego1 = pose
        x, y = boxes[:, 0], boxes[:, 1]
        dx, dy = boxes[:, 3], boxes[:, 4]      # mmdet3d LiDARInstance3DBoxes: (dx, dy, dz)
        yaw = boxes[:, 6]
        vx = boxes[:, 7] if boxes.shape[1] > 7 else np.zeros(len(boxes))
        vy = boxes[:, 8] if boxes.shape[1] > 8 else np.zeros(len(boxes))

        a0 = x + ego0
        a1 = -y + ego1
        # Covering radius of the AMODAL footprint — unlike a cluster radius this does
        # not shrink when only one face of the object is visible.
        r = 0.5 * np.hypot(dx, dy)
        return np.column_stack([a0, a1, r, scores, vx, -vy, -yaw, labels])

    def _publish(self, rows):
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="params", size=1, stride=1),
            MultiArrayDimension(label="dets", size=len(rows),
                                stride=ROW * max(1, len(rows)))]
        msg.data = [float(len(rows))] + [float(v) for v in np.asarray(rows).ravel()]
        self.pub.publish(msg)

    def _publish_markers(self, boxes, labels, pose):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "detections"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.25
        m.color = ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.9)
        m.pose.orientation.w = 1.0

        ego0, ego1 = pose
        pts = []
        for b in boxes:
            c0 = b[0] + ego0
            c1 = -b[1] + ego1
            dx, dy, dz = b[3], b[4], b[5]
            z = float(b[2])
            pts += _box_edges(c0, c1, -float(b[6]), float(dx), float(dy), z, z + float(dz))
        m.points = pts
        # An empty LINE_LIST is what CLEARS the previous frame's boxes; skipping the
        # publish would leave them frozen in RViz.
        arr.markers.append(m)
        self.viz.publish(arr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="../detector/pointpillars_apron.py")
    p.add_argument("--checkpoint", default=None,
                   help="Override the config's load_from.")
    p.add_argument("--lidar-topic", default="/point_cloud")
    p.add_argument("--pose-topic", default="/laser_scan_pose")
    p.add_argument("--out-topic", default="/detections")
    p.add_argument("--viz-topic", default="/viz/detections")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--threads", type=int, default=8)
    args, _ = p.parse_known_args()

    rclpy.init()
    node = DetectorNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
