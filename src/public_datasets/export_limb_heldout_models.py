from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import train_limb_personalized as training  # noqa: E402


ARTIFACT_ROLE = "strict_leave_one_condition_out_fold_model"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export eight strict held-out-condition Limb models for one subject."
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "limb_personalized" / "heldout_conditions",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subject not in training.SUBJECTS:
        raise ValueError(f"Unsupported subject {args.subject}; choose from {training.SUBJECTS}")

    # Do not append exporter messages to the original Stage 8 benchmark log.
    training.log = lambda message: print(message, flush=True)

    all_segments = training.load_limb_segments()
    split_manifest = training.build_split_manifest(all_segments)
    subject_segments = all_segments[all_segments["subject_id"] == args.subject].copy()
    # The audited manifest was generated on Windows and stores backslashes.
    # Normalize only the in-memory paths so the same manifest works under WSL.
    subject_segments["source_file"] = subject_segments["source_file"].str.replace(
        "\\", "/", regex=False
    )
    subject_splits = split_manifest[split_manifest["subject_id"] == args.subject].copy()

    selection_path = training.OUT / "PERSONALIZED_PER_FOLD.csv"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Validation-selected fold results not found: {selection_path}")
    selected = pd.read_csv(selection_path, encoding="utf-8-sig")
    selected = selected[selected["subject_id"] == args.subject].copy()
    if len(selected) != 8:
        raise RuntimeError(f"Expected 8 selected folds for Subject {args.subject}, got {len(selected)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"limb_subject{args.subject:02d}_heldout_split_manifest.csv"
    subject_splits.to_csv(manifest_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    split_hash = sha256_file(manifest_path)
    selection_hash = sha256_file(selection_path)

    channel_groups = training.load_channel_groups()
    arrays = training.load_subject_arrays(subject_segments)
    feature_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    index_rows: list[dict] = []

    for fold_index in range(1, 9):
        fold_id = f"S{args.subject:02d}_personal_fold{fold_index:02d}"
        row = selected[selected["fold_id"] == fold_id]
        if len(row) != 1:
            raise RuntimeError(f"Expected one selected configuration for {fold_id}, got {len(row)}")
        choice = row.iloc[0]
        window_id = str(choice["window_config_id"])
        trim_id = str(choice["trim_config_id"])
        modality = str(choice["modality"])
        model_type = str(choice["model_type"])
        config_key = (window_id, trim_id, modality)

        if config_key not in feature_cache:
            win = training.WINDOW_CONFIGS[window_id]
            trim = training.TRIM_CONFIGS[trim_id]
            print(f"Extracting Subject {args.subject} features for {config_key}", flush=True)
            feature_cache[config_key] = training.make_windows_for_subject(
                subject_segments,
                arrays,
                channel_groups[modality],
                win["window_length"],
                win["step"],
                trim["trim_start_fraction"],
                trim["trim_end_fraction"],
            )

        x_all, y_all, meta_all = feature_cache[config_key]
        fold_rows = subject_splits[subject_splits["fold_id"] == fold_id].copy()
        split_map = dict(zip(fold_rows["segment_id"], fold_rows["split"]))
        split_values = meta_all["segment_id"].map(split_map).to_numpy()
        train_val_mask = np.isin(split_values, ["train", "validation"])
        test_mask = split_values == "test"

        test_conditions = sorted(fold_rows.loc[fold_rows["split"] == "test", "condition"].unique())
        validation_conditions = sorted(
            fold_rows.loc[fold_rows["split"] == "validation", "condition"].unique()
        )
        if len(test_conditions) != 1 or len(validation_conditions) != 1:
            raise RuntimeError(
                f"{fold_id} must hold out exactly one test and one validation condition: "
                f"test={test_conditions}, validation={validation_conditions}"
            )
        held_out_condition = test_conditions[0]
        validation_condition = validation_conditions[0]

        training_segment_ids = sorted(
            fold_rows.loc[fold_rows["split"].isin(["train", "validation"]), "segment_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        test_segment_ids = sorted(
            fold_rows.loc[fold_rows["split"] == "test", "segment_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        if set(training_segment_ids) & set(test_segment_ids):
            raise RuntimeError(f"Leakage detected before fitting {fold_id}")

        artifact_path = (
            args.output_dir / f"limb_subject{args.subject:02d}_fold{fold_index:02d}_heldout.joblib"
        )
        if artifact_path.exists() and not args.force:
            print(f"Keeping existing artifact: {artifact_path}", flush=True)
            artifact_hash = sha256_file(artifact_path)
            payload = joblib.load(artifact_path)
            metrics = dict(payload["metadata"].get("held_out_test_metrics", {}))
        else:
            print(
                f"Fitting {fold_id}: train+validation={len(training_segment_ids)} segments, "
                f"held out={held_out_condition}",
                flush=True,
            )
            model = training.make_model(model_type)
            model.fit(x_all[train_val_mask], y_all[train_val_mask])
            test_prob = training.probabilities(model, x_all[test_mask])
            metrics, _ = training.evaluate_predictions(
                meta_all[test_mask].reset_index(drop=True),
                y_all[test_mask],
                test_prob,
                "test",
            )
            win = training.WINDOW_CONFIGS[window_id]
            trim = training.TRIM_CONFIGS[trim_id]
            metadata = {
                "artifact_role": ARTIFACT_ROLE,
                "benchmark_metric_role": "held-out condition only",
                "evaluation_protocol": "8-fold leave-one-condition-out with validation-only selection",
                "subject_id": int(args.subject),
                "fold_id": fold_id,
                "fold_index": int(fold_index),
                "held_out_condition": held_out_condition,
                "validation_condition": validation_condition,
                "classes": list(training.LABELS),
                "class_names": dict(training.LABEL_NAMES),
                "selected_candidate_id": str(choice["selected_candidate_id"]),
                "selection_rule": "configuration selected from validation metrics; held-out test not used",
                "model_type": model_type,
                "modality": modality,
                "channel_indices_zero_based": list(channel_groups[modality]),
                "window_length_samples": int(win["window_length"]),
                "window_length_seconds": float(win["window_length_sec"]),
                "step_samples": int(win["step"]),
                "trim_start_fraction": float(trim["trim_start_fraction"]),
                "trim_end_fraction": float(trim["trim_end_fraction"]),
                "feature_dim": int(x_all.shape[1]),
                "training_window_count": int(train_val_mask.sum()),
                "training_segment_count": len(training_segment_ids),
                "test_window_count": int(test_mask.sum()),
                "test_segment_count": len(test_segment_ids),
                "all_labeled_subject_segments_used": False,
                "training_segment_ids": training_segment_ids,
                "test_segment_ids": test_segment_ids,
                "split_manifest_path": str(manifest_path),
                "split_manifest_sha256": split_hash,
                "selection_results_path": str(selection_path),
                "selection_results_sha256": selection_hash,
                "held_out_test_metrics": metrics,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "software_versions": {
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__,
                    "joblib": joblib.__version__,
                },
            }
            joblib.dump({"pipeline": model, "metadata": metadata}, artifact_path, compress=3)
            artifact_hash = sha256_file(artifact_path)

        index_rows.append(
            {
                "subject_id": int(args.subject),
                "fold_id": fold_id,
                "fold_index": int(fold_index),
                "held_out_condition": held_out_condition,
                "validation_condition": validation_condition,
                "artifact_path": str(artifact_path),
                "artifact_sha256": artifact_hash,
                "split_manifest_sha256": split_hash,
                "test_segment_count": len(test_segment_ids),
                "test_window_macro_f1": metrics.get("test_window_macro_f1"),
                "test_segment_macro_f1": metrics.get("test_segment_macro_f1"),
            }
        )

    index_path = args.output_dir / f"limb_subject{args.subject:02d}_heldout_index.json"
    index_path.write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(index_rows)} strict fold models. Index: {index_path}", flush=True)


if __name__ == "__main__":
    main()
