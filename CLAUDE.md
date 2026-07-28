# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A Unity-based LiDAR simulator that raycasts a 3D point cloud and publishes it to **ROS 2** over TCP. The repo is two cooperating halves:

- `unity_ros_lidar/` — the Unity project (simulation, sensor model, ROS publishers). Authored in Unity 2021.3.16f1, currently opened with Unity 6 (6000.3.16f1).
- `ROS-TCP-Endpoint/` — a vendored copy of Unity's [ROS-TCP-Endpoint](https://github.com/Unity-Technologies/ROS-TCP-Endpoint) (**ROS2 branch only**), the ament_python package that bridges Unity TCP messages to ROS 2 topics.

The Unity side talks to ROS via the `com.unity.robotics.ros-tcp-connector` package (declared in `unity_ros_lidar/Packages/manifest.json`, pulled from git). Build is `ROS2` (see the `#if ROS2` branches in C# — message headers differ between ROS1/ROS2).

## Running the simulator

There is no CLI build/test for the Unity project — it runs from the Unity Editor. End-to-end:

1. ROS side — create a workspace with `ROS-TCP-Endpoint/` under `src/`, `colcon build`, then:
   ```bash
   source install/local_setup.bash
   ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1
   ```
2. Open `unity_ros_lidar` in Unity Hub, open `Assets/Scenes/SampleScene.unity`, press Play. Blue arrows on the ROS Settings indicator = connected.
3. Visualize in `rviz2`: add the `/point_cloud/PointCloud2` topic (bump point size to ~0.05 m).

### ROS-TCP-Endpoint tests (ROS package only)
```bash
cd ROS-TCP-Endpoint
python3 -m pytest test/                 # all tests
python3 -m pytest test/test_server.py   # a single file
```
Lint mirrors the package's ament hooks: `ament_flake8`, `ament_pep257`, `ament_copyright`.

## Architecture

### Publisher pattern
All Unity→ROS publishers are MonoBehaviours that grab a singleton `ROSConnection.GetOrCreateInstance()` in `Start()`, `RegisterPublisher<T>(topic)`, then publish in `Update()` gated by a rate limiter (`ShouldPublishMessage` comparing elapsed time to `1/rateHz`). Follow this pattern for new publishers rather than publishing every frame.

- `PointCloudPublisher.cs` — the main sensor entry point (attached to the "ROS Publishers" object in-scene). Public fields (FOV, angular resolution, range, rate, topics) are tuned in the Inspector, **not** in code. It owns a plain `LaserSensor3D` instance and publishes `PointCloud2Msg` + a `PoseMsg` of the sensor link.
- `ROSTransformTreePublisher.cs` — walks a GameObject hierarchy and publishes the `/tf` tree (`TFMessageMsg`), using `TransformTreeNode.cs` + `TransformExtensions.cs`.
- `Robot_Controller.cs` — subscribes to `cmd_vel` (`TwistMsg`) and drives a 4-wheel `ArticulationBody` robot (differential drive); zeroes velocity after `ROSTimeout`.

### LaserSensor3D (the sensor model)
`LaserSensor3D.cs` is a pure C# class (not a MonoBehaviour). It precomputes horizontal/vertical scan-angle arrays from FOV + angular resolution, then in `getScanMsg()` casts a `Physics.Raycast` per (h,v) angle pair and packs hits into a 16-byte-per-point buffer (x,y,z,intensity as FLOAT32). **Objects are only detected if they have a mesh collider.**

Two coordinate caveats live here:
- **Unity→ROS axis swap is done manually** when packing each point: ROS `x = Unity z`, ROS `y = -Unity x`, ROS `z = Unity y`. Preserve this when editing the packing loop.
- Known bug (see README "Notes"): points are currently emitted relative to sensor position but the frame handling means the cloud can appear in world rather than true lidar coordinates — flagged as a TODO, not yet fixed.

### Timestamps
`Clock.cs` (`Unity.Robotics.Core`) is the single source of ROS time, defaulting to `ClockMode.UnityScaled` (Unity sim time). Use `Clock.time` / `Clock.Now` with `TimeStamp.cs` to stamp message headers; don't read `Time.*` directly for ROS stamps.

### Utility
`MoveAlongAxis.cs` — moves any GameObject along one axis at a set speed; attached to the ego vehicle and other scene objects for simple motion.

## Conventions & gotchas

- Sensor/robot parameters are configured via serialized Inspector fields, so behavior changes often live in the `.unity` scene file, not in `.cs` files.
- The ego vehicle in the sample scene is "Airplane"; the sensor origin is the "laser_link" GameObject assigned to `PointCloudPublisher.laser_sensor_link`.
- Every `.cs` has a paired `.cs.meta` — let Unity manage these; don't hand-edit or delete meta files.
- `unity_ros_lidar/Library/` and `unity_ros_lidar/Logs/` are Unity-generated and gitignored — never commit them.
- Dont write too many comments just the essential documentation for the code