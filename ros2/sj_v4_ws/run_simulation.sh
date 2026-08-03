#!/usr/bin/env bash
set -eo pipefail

SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$SIM_DIR"

source /opt/ros/jazzy/setup.bash
"$SIM_DIR/build_ros2.sh"
source "$WS/install_sj/setup.bash"

# Optional arguments:
#   ./run_simulation.sh rviz:=false record_video:=false
ros2 launch sj_fitts_sim sj_fitts_ur5e_gazebo.launch.py "$@"
