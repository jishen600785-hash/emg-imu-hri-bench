# Model artifacts

Trained model files are generated artifacts and are not committed. The configuration files preserve artifact names and SHA-256 values where available.

Expected layout:

```text
models/
  limb_personalized/
    limb_subject01_deployment.joblib
    heldout_conditions/
      limb_subject01_fold01_heldout.joblib
  bingbin_realtime/
  uci_feature_fusion/
  capgmyo_spatiotemporal/
artifacts/
  ml/optimized/improved_strict_smoothed_model.joblib
  prepared/window_dataset.npz
```

Generate the files with the training scripts in `src/`, or copy locally archived artifacts into these paths. For the Limb evaluation artifacts, first run the personalized training pipeline and then execute:

```bash
ros2/limb_ws/build_heldout_models.sh 1
```

Never commit models trained on private data unless data governance and repository visibility have been reviewed.
