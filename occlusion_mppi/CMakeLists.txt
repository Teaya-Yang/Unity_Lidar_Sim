cmake_minimum_required(VERSION 3.10)
project(occlusion_mppi)

find_package(catkin REQUIRED COMPONENTS
        rospy std_msgs nav_msgs sensor_msgs geometry_msgs
        visualization_msgs quadrotor_msgs)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS rospy nav_msgs sensor_msgs visualization_msgs quadrotor_msgs)

catkin_install_python(PROGRAMS scripts/mppi_node.py scripts/rollout_sim.py
        scripts/occlusion_viz.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
