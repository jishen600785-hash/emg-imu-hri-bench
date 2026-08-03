# EMG/IMU Human-Robot Interaction Research Bench

This repository contains the reproducible code release for **Establishing a Research Bench for Human-Robot Interaction**. It connects EMG/IMU gesture recognition to planar Cartesian control of a simulated UR5e robot and evaluates the interaction with an adapted multidirectional Fitts-style target-acquisition task.

The release includes public-dataset experiments, personalized Limb Position modelling, an independently collected SJ dataset pipeline, ROS 2 / MoveIt 2 / Gazebo simulation packages, evaluation protocols, and compact numerical results. Raw human-subject recordings, participant photographs, trained-model binaries, videos, and generated ROS build folders are intentionally excluded.

## System overview

```text
Raw EMG/IMU recording
        |
        v
Segment-safe sliding windows
        |
        v
Feature extraction or deep representation
        |
        v
Gesture classifier and causal filtering
        |
        v
ROS 2 gesture message -> planar MoveIt Servo command
        |
        v
UR5e Gazebo target acquisition -> Fitts-style metrics and video metadata
```

## Gesture-to-robot mapping

The Limb Position and SJ branches use semantically different gesture names but the same four planar directions and stop/confirm command.

| Robot command | Limb Position gesture | SJ gesture | Cartesian direction |
|---|---|---|---|
| Move right | Hand Open | Hand Open | +X |
| Move left | Lateral Grip | Hand Close | -X |
| Move up | Pinch Grip | Hand Up | +Y |
| Move down | Power Grip | Hand Down | -Y |
| Stop / confirm selection | Rest | Rest | zero velocity |

## Repository layout

```text
configs/                    English model and experiment metadata
data/                       Dataset placement instructions only
docs/                       Architecture, datasets, results, and reproduction notes
models/                     Model placement instructions; binaries are not committed
protocols/                  Frozen split and data-quality protocol assets
results/
  public_datasets/          Compact benchmark tables
  sj_training/              SJ model comparison summaries
  robot/                    Limb, SJ V3, and SJ V4 robot metrics
ros2/
  limb_ws/                  Limb Position ROS 2 Fitts-style platform
  sj_v3_ws/                 SJ baseline controller
  sj_v4_ws/                 SJ adaptive coarse-to-fine controller
src/
  public_datasets/          Public, Limb, and Bingbin training/evaluation code
  sj/                       Standalone SJ ML and dual-branch CNN pipelines
```

## Verified headline results

The following values are copied from the numerical result files in this repository. Each row has a different evaluation protocol and must be interpreted separately.

| Branch | Selected model / system | Accuracy | Macro-F1 | Evaluation note |
|---|---|---:|---:|---|
| UCI EMG Data for Gestures | LSTM-Transformer + feature fusion | 82.27% | 82.26% | Subject-aware 3-fold |
| CapgMyo DB-a | Spatial-Temporal ResNet-SE | 95.22% | 95.23% | Trial-aware calibrated 3-fold |
| Limb Position | Personalized selected pipeline | 90.56% | 90.50% | 9 subjects, 72 segment-safe folds |
| Bingbin_Realtime V3 | Recording-level traditional ML | 92.86% | 92.72% | Exploratory; earlier test results were already known |
| SJ dynamic hold-out | Extra Trees, raw windows | 81.07% | 80.26% | Dynamic condition held out from model fitting |
| SJ dynamic hold-out | Extra Trees + causal 7-window history | 92.14% | 92.08% | Current and past predictions only |

Robot-task results:

| Platform | Trials | Success | Classification accuracy | Mean simulated movement time | Throughput |
|---|---:|---:|---:|---:|---:|
| Limb Position | 42 | 83.33% | 92.65% | 17.34 s | 0.162 bit/s |
| SJ V3 baseline | 42 | 80.95% | 84.05% | 18.98 s | 0.163 bit/s |
| SJ V4 adaptive control | 42 | 100.00% | 90.20% | 11.94 s | 0.371 bit/s |

See [docs/results.md](docs/results.md) for protocol boundaries and exact source files.

## Python environment

Python 3.12 is supported by the curated source tree.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For deep-learning experiments, install a PyTorch build appropriate for the host and then install the optional requirements:

```bash
python -m pip install -r requirements-dl.txt
```

## Main training entry points

```bash
# Public deep-learning benchmark
python src/public_datasets/public_dataset_benchmark.py --datasets both --download

# UCI feature-fusion experiment
python src/public_datasets/train_uci_feature_fusion.py

# CapgMyo spatial-temporal experiment
python src/public_datasets/train_capgmyo_spatial_resnet.py --datasets capgmyo

# Limb Position personalized model search
python src/public_datasets/train_limb_personalized.py

# Export the eight validation-selected, held-out-condition Limb models
ros2/limb_ws/build_heldout_models.sh 1

# Bingbin recording-level V3 experiment
python src/public_datasets/train_bingbin_v3.py

# SJ traditional machine-learning comparison
python src/sj/train_ml.py /path/to/sj/raw_data outputs/sj_ml

# SJ dual-branch 1D-CNN
python src/sj/train_dl.py /path/to/sj/raw_data outputs/sj_dl
```

The raw data must first be placed according to [data/README.md](data/README.md). Model selection is performed on training/validation data; the dynamic SJ condition is used as the held-out test condition in the released pipeline.

## ROS 2 simulation

The robot platform was developed for Ubuntu 24.04, ROS 2 Jazzy, MoveIt 2, and Gazebo Harmonic. Set the repository root before launching so model, protocol, and output paths remain machine-independent:

```bash
export EMG_HRI_PROJECT_ROOT="$(pwd)"
source /opt/ros/jazzy/setup.bash

cd ros2/limb_ws
./build_ros2.sh
source install_video/setup.bash
./run_simulation.sh
```

The equivalent SJ commands are in `ros2/sj_v3_ws` and `ros2/sj_v4_ws`. Model artifacts are required locally but are not stored in Git; see [models/README.md](models/README.md).

## Reproducibility and data policy

- Original recordings are the atomic split unit; windows derived from one recording or labelled segment are never split across train, validation, and test.
- Standardizers and feature selectors are fitted only inside training pipelines.
- Raw human data, participant images, and local absolute paths are excluded.
- Model binaries and videos are excluded because they are large generated artifacts; hashes and compact metrics are retained where available.
- The Fitts implementation is an adapted multidirectional research benchmark. EMG gesture input is outside the physical-input-device scope of ISO 9241-411 and is therefore not claimed as a conformant ISO test.

## Author

Ji Shen (K25024443), King's College London.

This repository is provided for academic review and reproducibility. No separate software licence is granted in this release.
