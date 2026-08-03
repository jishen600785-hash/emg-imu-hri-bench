#!/usr/bin/env bash
set -eo pipefail

SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$SIM_DIR"

source /opt/ros/jazzy/setup.bash

cd "$WS"
colcon --log-base log_video build \
  --build-base build_video \
  --install-base install_video \
  --packages-select limb_fitts_sim \
  --event-handlers console_cohesion+

echo "Build complete: $WS/install_video"
