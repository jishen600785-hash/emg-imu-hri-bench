from __future__ import annotations

import argparse
import csv
import os
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

from .constants import LABEL_NAMES
from .model_adapter import PersonalizedLimbModel


DEFAULT_ROOT = Path(os.environ.get("EMG_HRI_PROJECT_ROOT", Path.cwd()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test one Limb deployment model on five real segments")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--condition", default="StaticP1")
    args, _ = parser.parse_known_args()

    model_path = (
        args.project_root
        / "models"
        / "limb_personalized"
        / f"limb_subject{args.subject:02d}_deployment.joblib"
    )
    model = PersonalizedLimbModel(model_path)
    manifest = args.project_root / "protocols" / "limb" / "LIMB_RECORDING_MANIFEST.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["subject_id"]) == args.subject
            and row["condition"] == args.condition
            and int(row["action_label"]) in range(1, 6)
            and row["usable_status"].startswith("usable")
        ]
    source = args.project_root / rows[0]["source_file"].replace("\\", "/")
    data = np.asarray(sio.loadmat(str(source), simplify_cells=True)["limbEMG_Data"][args.condition], dtype=np.float32)

    correct = 0
    for row in rows:
        truth = int(row["action_label"]) - 1
        start = int(row["start_index"]) - 1
        end = int(row["end_index"])
        segment = data[start:end, :42]
        offset = max((len(segment) - model.window_samples) // 2, 0)
        prediction = model.predict(segment[offset : offset + model.window_samples])
        correct += int(prediction.label == truth)
        print(
            f"truth={truth}:{LABEL_NAMES[truth]:<9} "
            f"pred={prediction.label}:{LABEL_NAMES[prediction.label]:<9} "
            f"conf={prediction.confidence:.3f} inference={prediction.inference_ms:.3f} ms"
        )
    print(f"single-window smoke accuracy: {correct}/{len(rows)}")


if __name__ == "__main__":
    main()
