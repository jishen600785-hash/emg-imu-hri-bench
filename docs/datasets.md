# Dataset branches

| Dataset | Task | Classes | Main protocol | Main source |
|---|---|---:|---|---|
| UCI EMG Data for Gestures | Public sEMG benchmark | 6 | Subject-aware 3-fold | `src/public_datasets/train_uci_feature_fusion.py` |
| CapgMyo DB-a | High-density sEMG benchmark | 8 | Trial-aware calibrated 3-fold | `src/public_datasets/train_capgmyo_spatial_resnet.py` |
| Limb Position | Subject-dependent gesture recognition | 5 | Eight segment-safe folds per subject | `src/public_datasets/train_limb_personalized.py` |
| Bingbin_Realtime | Recording-level gesture recognition | 7 | Frozen recording-level exploratory split | `src/public_datasets/train_bingbin_v3.py` |
| SJ EMG+IMU | Five-gesture application dataset | 5 | Static train/validation; dynamic held-out test | `src/sj/train_ml.py` and `src/sj/train_dl.py` |

## Limb Position signal representation

- Sampling rate: 1,260 Hz.
- Input: 42 channels in columns 1-42; action label in column 43.
- EMG-only candidates use six EMG channels; multimodal candidates additionally use IMU channels.
- Candidate windows: 0.25, 0.5, and 1.0 s with 50% overlap.
- The selected feature vectors include time-domain and signal-shape statistics; the training script preserves original segments as split units.
- Candidate models: shrinkage LDA, logistic regression, RBF-SVM, and random forest after a first-stage representation screen.

## SJ signal representation

- Five gestures: hand close, hand down, hand open, hand up, and rest.
- Window length: 0.50 s; step: 0.25 s.
- The traditional ML branch compares multiple classifiers and selects using validation evidence.
- The deep branch uses a dual-branch 1D-CNN for EMG and IMU.
- The dynamic recording condition is held out from fitting in the released evaluation pipeline.
