# Dataset placement

Raw data are not distributed in this repository. This prevents disclosure of participant data and avoids redistributing third-party datasets.

Recommended layout:

```text
data/
  limb/
    subject01/StaticP1/subject1_limb_EMG.mat
    subject01/StaticP2/subject1_limb_EMG.mat
    ...
  bingbin/
    subject01/...
  sj/
    position_1/*.hpf
    position_2_down/*.hpf
    position_3_up/*.hpf
    dynamic/*.hpf
  uci/
  capgmyo/
```

The exact local roots may be recorded in `configs/data_paths.json`; use `configs/data_paths.example.json` as the template. Do not commit `configs/data_paths.json` if it contains machine-specific or participant-identifying paths.

For Limb Position, the released manifest uses repository-relative example paths. If the downloaded dataset uses another directory layout, update only the `source_file` column while preserving segment identifiers, labels, and split rules.
