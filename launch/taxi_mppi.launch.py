"""
taxi_mppi.launch.py
===================
Everything needed to watch the occlusion-aware MPPI run, in one command:

    ros2 launch launch/taxi_mppi.launch.py

  * ros_tcp_endpoint  — the Unity <-> ROS 2 bridge (the same
    `ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1`)
  * rviz2             — preloaded with rviz/taxi_mppi.rviz
  * the MPPI controller with --rviz-viz, publishing /viz/* markers

Unity itself is NOT launched: press Play in the Editor after this comes up (or pass
unity_exec:=/path/to/build to have the controller start a headless build).

The controller runs under the repo venv's interpreter, which is where mlagents lives,
but it also imports rclpy from the sourced ROS 2 install — so source ROS 2 in the shell
you launch from.

Common overrides:
    ros2 launch launch/taxi_mppi.launch.py rviz:=false
    ros2 launch launch/taxi_mppi.launch.py controller:=false      # bridge + RViz only
    ros2 launch launch/taxi_mppi.launch.py dynamic_obstacles:=true rviz_rollouts:=100
    ros2 launch launch/taxi_mppi.launch.py save_traj:=out/run1
    ros2 launch launch/taxi_mppi.launch.py use_gpu:=true        # GPU (RGL) lidar
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_DIR = os.path.join(REPO, "controller")
VENV_PYTHON = os.path.join(REPO, ".venv", "bin", "python")


def _controller(context, *_a, **_kw):
    """Build the controller command at launch time.

    Resolved here rather than with substitutions because optional flags must be ABSENT,
    not empty: an empty argv element reaches argparse as an unparseable positional.
    """
    def cfg(name):
        return LaunchConfiguration(name).perform(context)

    if cfg("controller").lower() != "true":
        return []

    cmd = [cfg("python"), "taxi_controller_mppi.py",
           "--lidar-costmap",
           "--lidar-topic", cfg("lidar_topic"),
           "--port", cfg("port"),
           "--rviz-viz",
           "--rviz-rollouts", cfg("rviz_rollouts")]
    if cfg("occlusion_aware").lower() == "true":
        cmd.append("--occlusion-aware")
    if cfg("dynamic_obstacles").lower() == "true":
        cmd.append("--dynamic-obstacles")
    if cfg("use_gpu").lower() == "true":
        cmd.append("--use-gpu-lidar")
    if cfg("traj_video").lower() == "true":
        cmd += ["--traj-video",
                "--video-fps", cfg("video_fps"),
                "--video-stride", cfg("video_stride")]
    for flag, arg in (("--exec", "unity_exec"), ("--save-traj", "save_traj"),
                      ("--config", "config")):
        if cfg(arg):
            cmd += [flag, cfg(arg)]

    return [ExecuteProcess(cmd=cmd, cwd=CONTROLLER_DIR, output="screen",
                           emulate_tty=True)]


def generate_launch_description():
    args = [
        DeclareLaunchArgument("ros_ip", default_value="127.0.0.1",
                             description="ROS_IP the TCP endpoint binds to."),
        DeclareLaunchArgument("ros_tcp_port", default_value="10000",
                             description="TCP port the Unity connector talks to."),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config",
                             default_value=os.path.join(REPO, "rviz", "taxi_mppi.rviz")),
        DeclareLaunchArgument("controller", default_value="true",
                             description="Also start taxi_controller_mppi.py."),
        DeclareLaunchArgument("python", default_value=VENV_PYTHON,
                             description="Interpreter for the controller (needs mlagents "
                                         "+ rclpy)."),
        DeclareLaunchArgument("port", default_value="5004",
                             description="ML-Agents base port to Unity."),
        DeclareLaunchArgument("unity_exec", default_value="",
                             description="Unity build to run headless. Empty = attach to "
                                         "the Editor."),
        DeclareLaunchArgument("lidar_topic", default_value="/point_cloud"),
        DeclareLaunchArgument("occlusion_aware", default_value="false"),
        DeclareLaunchArgument("dynamic_obstacles", default_value="false"),
        DeclareLaunchArgument("use_gpu", default_value="false",
                             description="Raycast the LiDAR on the GPU in Unity (Robotec "
                                         "GPU Lidar) instead of Physics.Raycast. Reaches "
                                         "Unity as the 'use_gpu' ML-Agents environment "
                                         "parameter, so it works in the Editor too. The "
                                         "published cloud is identical either way."),
        DeclareLaunchArgument("rviz_rollouts", default_value="100",
                             description="Sampled rollouts drawn per solve (0 = plan only)."),
        DeclareLaunchArgument("save_traj", default_value="",
                             description="Directory for the trajectory CSV + PNGs."),
        DeclareLaunchArgument("traj_video", default_value="true",
                             description="Also write traj.mp4 (traj.gif without ffmpeg): "
                                         "the map figure replayed over every solve. "
                                         "Needs save_traj."),
        DeclareLaunchArgument("video_fps", default_value="10",
                             description="Frame rate of traj_video."),
        DeclareLaunchArgument("video_stride", default_value="1",
                             description="Draw only every Nth solve in traj_video."),
        DeclareLaunchArgument("config", default_value="",
                             description="Tuning YAML (default: controller/config.yaml)."),
    ]

    endpoint = Node(
        package="ros_tcp_endpoint",
        executable="default_server_endpoint",
        name="ros_tcp_endpoint",
        output="screen",
        emulate_tty=True,
        parameters=[{"ROS_IP": LaunchConfiguration("ros_ip"),
                     "ROS_TCP_PORT": LaunchConfiguration("ros_tcp_port")}],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # The controller is a plain script, not an ament package, so ExecuteProcess with
    # cwd=controller/ — its imports (taxi_config, occlusion_capsules, rviz_viz) are
    # sibling modules resolved relative to the script.
    return LaunchDescription(args + [endpoint, rviz, OpaqueFunction(function=_controller)])
