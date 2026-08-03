#!/usr/bin/env bash
set -eo pipefail

SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$SIM_DIR"

source /opt/ros/jazzy/setup.bash
if [[ ! -f "$WS/install_video/setup.bash" ]]; then
  "$SIM_DIR/build_ros2.sh"
fi
source "$WS/install_video/setup.bash"
ros2 run limb_fitts_sim model_smoke_test "$@"
