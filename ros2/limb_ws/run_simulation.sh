#!/usr/bin/env bash
set -eo pipefail

SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$SIM_DIR"

source /opt/ros/jazzy/setup.bash
"$SIM_DIR/build_ros2.sh"
source "$WS/install_video/setup.bash"

# Extra launch arguments can be appended, for example:
#   ./run_simulation.sh subject_id:=1 fold_index:=1 rviz:=false
ros2 launch limb_fitts_sim fitts_ur5e_gazebo.launch.py "$@"
