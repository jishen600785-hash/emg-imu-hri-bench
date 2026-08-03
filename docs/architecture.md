# Platform architecture

## Offline modelling path

1. A dataset adapter reads MAT, TXT, ZIP-contained MAT, or HPF recordings.
2. The original recording or contiguous labelled segment is assigned atomically to train, validation, or test.
3. Sliding windows are created without crossing action boundaries.
4. The pipeline either extracts engineered EMG/IMU features or learns a deep representation.
5. Normalization, feature selection, and classification are fitted on training data only.
6. Validation data select model and hyperparameter candidates.
7. Held-out test data are evaluated after selection.

## Online and replay path

The ROS 2 implementations replay raw windows through the same inference interface used by a live source. The replay node does not replay precomputed class labels. It publishes signal windows, the classifier performs inference, and the filtered gesture drives robot motion.

| Node responsibility | Main output |
|---|---|
| EMG/IMU replay or acquisition | Timestamped signal window |
| Gesture classifier | Label, probability, inference latency |
| Gesture servo | Planar Cartesian twist |
| Fitts task manager | Current target, selection state, trial state |
| Metrics recorder | Trial CSV and compact JSON summary |
| Video recorder | Annotated simulation video and metadata |

## Fitts-style task

Targets are placed on a fixed two-dimensional plane at three nominal amplitudes (0.08, 0.18, and 0.28 m) and two target widths (0.025 and 0.06 m). Four gestures move the robot endpoint along the planar axes. Rest stops motion and confirms a selection. The task records success, misses, timeouts, movement time, classification errors, inference latency, control latency, endpoint error, and effective throughput.

The V4 SJ controller adds width-aware stopping, axis hysteresis, and coarse-to-fine speed scaling. The classifier is still signal-driven; the target geometry only determines which gesture should be requested by the evaluation replay policy and how control performance is scored.
