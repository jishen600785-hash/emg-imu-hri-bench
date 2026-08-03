#!/usr/bin/env bash
set -eo pipefail

SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SIM_DIR/.." && pwd)"
WS="$SIM_DIR"

source /opt/ros/jazzy/setup.bash
if [[ ! -f "$WS/install_sj/setup.bash" ]]; then
  "$SIM_DIR/build_ros2.sh"
fi
source "$WS/install_sj/setup.bash"

ros2 run sj_fitts_sim model_smoke_test \
  --project-root "$PROJECT_ROOT" \
  "$@"
