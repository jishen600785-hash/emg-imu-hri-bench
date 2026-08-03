#!/usr/bin/env bash
set -eo pipefail

SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$SIM_DIR"

source /opt/ros/jazzy/setup.bash

cd "$WS"
colcon --log-base log_sj build \
  --build-base build_sj \
  --install-base install_sj \
  --packages-select sj_fitts_sim \
  --event-handlers console_cohesion+

echo "Build complete: $WS/install_sj"
