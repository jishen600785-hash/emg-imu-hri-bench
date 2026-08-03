# Reproduction guide

## 1. Clone and prepare Python

```bash
git clone https://github.com/jishen600785-hash/emg-imu-hri-bench.git
cd emg-imu-hri-bench
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the appropriate PyTorch build and `requirements-dl.txt` for deep-learning runs.

## 2. Place data

Follow `data/README.md`. Raw data remain local and are ignored by Git.

## 3. Run classification experiments

Use the commands in the root README. Every experiment writes to `outputs/`, which is ignored because outputs can be large and machine-specific.

## 4. Prepare model artifacts

Follow `models/README.md`. Validate model hashes against the corresponding configuration or result metadata when reusing archived models.

## 5. Build a ROS 2 workspace

```bash
export EMG_HRI_PROJECT_ROOT="$(git rev-parse --show-toplevel)"
source /opt/ros/jazzy/setup.bash
cd ros2/sj_v4_ws
./build_ros2.sh
source install_sj/setup.bash
./run_simulation.sh
```

Use `ros2/limb_ws` or `ros2/sj_v3_ws` to reproduce the other branches.

## 6. Validate the release

```bash
python -m compileall -q src ros2
git status --short
```

Before publishing new results, verify that no raw recordings, participant images, absolute machine paths, model binaries, videos, `build/`, `install/`, or `log/` directories are staged.
