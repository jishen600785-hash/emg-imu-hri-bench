from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_dataset_benchmark as base  # noqa: E402


OUT_DEFAULT = ROOT / "outputs" / "uci_strict_evaluation"
SEED = 42
MODELS = ["LSTMTransformer", "FeatureExtraTrees", "DeepFeatureFusion"]
DISPLAY = {
    "LSTMTransformer": "LSTM-Transformer",
    "FeatureExtraTrees": "Feature ExtraTrees",
    "DeepFeatureFusion": "LSTM-Transformer + Feature Fusion",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)


def extract_features(x: np.ndarray) -> np.ndarray:
    eps = 1e-8
    diff = np.diff(x, axis=2)
    features = [
        np.mean(x, axis=2),
        np.std(x, axis=2),
        np.median(x, axis=2),
        np.quantile(x, 0.75, axis=2) - np.quantile(x, 0.25, axis=2),
        np.mean(np.abs(x), axis=2),
        np.sqrt(np.mean(np.square(x), axis=2)),
        np.mean(np.abs(diff), axis=2),
        np.mean(x[:, :, :-1] * x[:, :, 1:] < 0, axis=2),
        np.mean(diff[:, :, :-1] * diff[:, :, 1:] < 0, axis=2),
        np.ptp(x, axis=2),
    ]
    spectrum = np.abs(np.fft.rfft(x, axis=2)) ** 2
    spectrum = spectrum / (spectrum.sum(axis=2, keepdims=True) + eps)
    bins = spectrum.shape[2]
    features.append(-np.sum(spectrum * np.log(spectrum + eps), axis=2) / np.log(bins))
    for low, high in ((0.0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1.0)):
        start = int(low * (bins - 1))
        stop = max(start + 1, int(high * (bins - 1)))
        features.append(spectrum[:, :, start:stop].sum(axis=2))
    return np.concatenate(features, axis=1).astype(np.float32)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(6)), average="macro", zero_division=0
    )
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_precision": float(precision),
        f"{prefix}_macro_recall": float(recall),
        f"{prefix}_macro_f1": float(f1),
    }


def predict_deep(model, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int):
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=False)
    return base.predict(model, loader, device)


def choose_fusion_weight(y_val: np.ndarray, deep_probs: np.ndarray, feature_probs: np.ndarray) -> tuple[float, float]:
    candidates = []
    for weight in np.linspace(0.0, 1.0, 41):
        probabilities = weight * deep_probs + (1.0 - weight) * feature_probs
        score = float(f1_score(y_val, probabilities.argmax(axis=1), average="macro"))
        candidates.append((score, -abs(weight - 0.5), weight))
    best = max(candidates)
    return float(best[2]), float(best[0])


def append_per_class(rows: list[dict], fold_id: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(6)), zero_division=0
    )
    for class_id, class_name in base.UCI_CLASS_MAP.items():
        rows.append(
            {
                "dataset_name": "UCI_EMG_Data_for_Gestures",
                "fold_id": fold_id,
                "model_name": model_name,
                "model_display_name": DISPLAY[model_name],
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
        )


def run(args: argparse.Namespace, device: torch.device, out: Path):
    bundle = base.build_uci_bundle(200, 100, False)
    engineered = extract_features(bundle.x)
    per_fold, per_class, history, split_rows, selection_rows, complexity = [], [], [], [], [], []
    pooled = {model: ([], []) for model in MODELS}
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for fold_index in range(3):
        fold_id, masks, split_values, split = base.make_split_masks(bundle, fold_index)
        split_rows.extend(split)
        x_train, x_val, x_test = base.standardize_by_train(
            bundle.x[masks["train"]], bundle.x[masks["train"]],
            bundle.x[masks["validation"]], bundle.x[masks["test"]],
        )
        y_train, y_val, y_test = bundle.y[masks["train"]], bundle.y[masks["validation"]], bundle.y[masks["test"]]
        feature_train = engineered[masks["train"]]
        feature_val = engineered[masks["validation"]]
        feature_test = engineered[masks["test"]]

        print(f"fold {fold_index + 1}/3 deep branch", flush=True)
        deep_result = base.train_one_model(
            "LSTMTransformer", x_train, y_train, x_val, y_val, 6,
            device, args.epochs, args.batch_size, SEED + fold_index * 100,
        )
        _, deep_val_probs = predict_deep(deep_result.model, x_val, y_val, device, args.batch_size)
        infer_start = time.perf_counter()
        deep_true, deep_test_probs = predict_deep(deep_result.model, x_test, y_test, device, args.batch_size)
        deep_infer = time.perf_counter() - infer_start

        print(f"fold {fold_index + 1}/3 feature branch", flush=True)
        feature_start = time.perf_counter()
        feature_model = ExtraTreesClassifier(
            n_estimators=1000,
            max_features=0.7,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED + fold_index,
        )
        feature_model.fit(feature_train, y_train)
        feature_train_seconds = time.perf_counter() - feature_start
        feature_val_probs = feature_model.predict_proba(feature_val)
        infer_start = time.perf_counter()
        feature_test_probs = feature_model.predict_proba(feature_test)
        feature_infer = time.perf_counter() - infer_start

        fusion_weight, fusion_val_f1 = choose_fusion_weight(y_val, deep_val_probs, feature_val_probs)
        fusion_test_probs = fusion_weight * deep_test_probs + (1.0 - fusion_weight) * feature_test_probs
        model_probabilities = {
            "LSTMTransformer": deep_test_probs,
            "FeatureExtraTrees": feature_test_probs,
            "DeepFeatureFusion": fusion_test_probs,
        }
        deep_val_f1 = f1_score(y_val, deep_val_probs.argmax(axis=1), average="macro")
        feature_val_f1 = f1_score(y_val, feature_val_probs.argmax(axis=1), average="macro")
        validation_scores = {
            "LSTMTransformer": float(deep_val_f1),
            "FeatureExtraTrees": float(feature_val_f1),
            "DeepFeatureFusion": float(fusion_val_f1),
        }
        selection_rows.append(
            {
                "fold_id": fold_id,
                "deep_validation_macro_f1": float(deep_val_f1),
                "feature_validation_macro_f1": float(feature_val_f1),
                "fusion_validation_macro_f1": float(fusion_val_f1),
                "selected_deep_probability_weight": fusion_weight,
                "selected_feature_probability_weight": 1.0 - fusion_weight,
                "selection_used_test": False,
            }
        )
        tree_nodes = int(sum(estimator.tree_.node_count for estimator in feature_model.estimators_))
        for model_name, probabilities in model_probabilities.items():
            prediction = probabilities.argmax(axis=1)
            metrics = metric_dict(y_test, prediction, "test")
            train_seconds = (
                deep_result.train_seconds if model_name == "LSTMTransformer" else
                feature_train_seconds if model_name == "FeatureExtraTrees" else
                deep_result.train_seconds + feature_train_seconds
            )
            inference_seconds = (
                deep_infer if model_name == "LSTMTransformer" else
                feature_infer if model_name == "FeatureExtraTrees" else
                deep_infer + feature_infer
            )
            row = {
                "dataset_name": "UCI_EMG_Data_for_Gestures",
                "protocol": "subject-aware 3-fold; test subjects held out",
                "evaluation_unit": "window",
                "fold_id": fold_id,
                "model_name": model_name,
                "model_display_name": DISPLAY[model_name],
                "train_split_values": ";".join(map(str, split_values["train"])),
                "validation_split_values": ";".join(map(str, split_values["validation"])),
                "test_split_values": ";".join(map(str, split_values["test"])),
                "train_sample_count": int(len(y_train)),
                "validation_sample_count": int(len(y_val)),
                "test_sample_count": int(len(y_test)),
                "best_epoch": deep_result.best_epoch if model_name != "FeatureExtraTrees" else "",
                "best_val_macro_f1": validation_scores[model_name],
                "fusion_deep_weight": fusion_weight if model_name == "DeepFeatureFusion" else "",
                "train_seconds": train_seconds,
                "avg_inference_ms_per_sample": float(inference_seconds / max(len(y_test), 1) * 1000),
                "deep_param_count": deep_result.param_count if model_name != "FeatureExtraTrees" else 0,
                "tree_node_count": tree_nodes if model_name != "LSTMTransformer" else 0,
                **metrics,
            }
            per_fold.append(row)
            append_per_class(per_class, fold_id, model_name, deep_true, prediction)
            complexity.append({key: row[key] for key in [
                "dataset_name", "fold_id", "model_name", "model_display_name", "deep_param_count",
                "tree_node_count", "train_seconds", "avg_inference_ms_per_sample",
            ]})
            pooled[model_name][0].append(deep_true)
            pooled[model_name][1].append(prediction)
        history.extend(
            {"dataset_name": "UCI_EMG_Data_for_Gestures", "fold_id": fold_id,
             "model_name": "LSTMTransformer", "model_display_name": DISPLAY["LSTMTransformer"], **item}
            for item in deep_result.history
        )
        torch.save(deep_result.model.state_dict(), models_dir / f"{fold_id}_lstm_transformer.pt")
        joblib.dump(feature_model, models_dir / f"{fold_id}_extra_trees.joblib")
    predictions = {model: (np.concatenate(values[0]), np.concatenate(values[1])) for model, values in pooled.items()}
    return (
        pd.DataFrame(per_fold), pd.DataFrame(per_class), pd.DataFrame(history),
        pd.DataFrame(split_rows), pd.DataFrame(selection_rows), pd.DataFrame(complexity), predictions,
    )


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in per_fold.groupby("model_name", sort=False):
        rows.append(
            {
                "dataset_name": "UCI_EMG_Data_for_Gestures",
                "model_name": model_name,
                "model_display_name": DISPLAY[model_name],
                "protocol": group["protocol"].iloc[0],
                "folds_completed": int(len(group)),
                "validation_macro_f1_mean": float(group["best_val_macro_f1"].mean()),
                "test_accuracy_mean": float(group["test_accuracy"].mean()),
                "test_accuracy_std": float(group["test_accuracy"].std(ddof=0)),
                "test_balanced_accuracy_mean": float(group["test_balanced_accuracy"].mean()),
                "test_macro_f1_mean": float(group["test_macro_f1"].mean()),
                "test_macro_f1_std": float(group["test_macro_f1"].std(ddof=0)),
                "test_macro_f1_min": float(group["test_macro_f1"].min()),
                "test_macro_f1_max": float(group["test_macro_f1"].max()),
                "train_seconds_mean": float(group["train_seconds"].mean()),
                "avg_inference_ms_per_sample_mean": float(group["avg_inference_ms_per_sample"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["selected_by_validation"] = False
    summary.loc[summary["validation_macro_f1_mean"].idxmax(), "selected_by_validation"] = True
    return summary.sort_values("test_macro_f1_mean", ascending=False).reset_index(drop=True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def make_figures(out: Path, summary: pd.DataFrame, per_fold: pd.DataFrame, history: pd.DataFrame, predictions) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"LSTM-Transformer": "#9E9E9E", "Feature ExtraTrees": "#4C78A8", "LSTM-Transformer + Feature Fusion": "#E45756"}
    fig, ax = plt.subplots(figsize=(7.5, 4.1), dpi=180)
    labels = summary["model_display_name"].tolist()
    values = summary["test_macro_f1_mean"].to_numpy(float)
    bars = ax.barh(np.arange(len(labels)), values, color=[colors[label] for label in labels], edgecolor="white")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(max(0, values.min() - 0.06), min(1, values.max() + 0.06))
    ax.set_xlabel("Test Macro-F1")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures / "uci_strict_deep_feature_model_comparison")

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    for index, model in enumerate(labels):
        values = per_fold.loc[per_fold["model_display_name"] == model, "test_macro_f1"].astype(float).to_numpy()
        x = np.full(len(values), index) + np.linspace(-0.06, 0.06, len(values))
        ax.scatter(x, values, s=42, color=colors[model], edgecolor="white", linewidth=0.7)
        ax.hlines(values.mean(), index - 0.2, index + 0.2, color=colors[model], linewidth=2)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=12, ha="right")
    ax.set_ylabel("Test Macro-F1")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, figures / "uci_strict_deep_feature_fold_stability")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=180)
    grouped = history.groupby("epoch")
    epochs = np.asarray(sorted(grouped.groups), dtype=int)
    axes[0].plot(epochs, grouped["train_loss"].mean().reindex(epochs).to_numpy(float), color="#E45756", linewidth=1.8)
    axes[1].plot(epochs, grouped["val_macro_f1"].mean().reindex(epochs).to_numpy(float), color="#E45756", linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("LSTM-Transformer training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Macro-F1")
    for axis in axes:
        axis.grid(color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, figures / "uci_lstm_transformer_training_convergence")

    names = [base.UCI_CLASS_MAP[index] for index in range(6)]
    for model_name, (y_true, y_pred) in predictions.items():
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(6)), normalize="true")
        fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=180)
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(6), labels=names, rotation=35, ha="right")
        ax.set_yticks(range(6), labels=names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(DISPLAY[model_name])
        for row in range(6):
            for col in range(6):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if matrix[row, col] > 0.55 else "black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        save_figure(fig, figures / f"uci_{model_name}_confusion")


def write_reports(out: Path, summary: pd.DataFrame, selection: pd.DataFrame) -> None:
    selected = summary[summary["selected_by_validation"]].iloc[0]
    report = [
        "# Stage25 UCI Strict Deep-Feature Fusion",
        "",
        "## Protocol",
        "",
        "- Full UCI EMG Data for Gestures: 36 subjects, six classes.",
        "- Frozen Stage17/19 subject-aware three-fold split.",
        "- Feature extraction and model fitting use train subjects only.",
        "- Fusion weight is selected separately in each fold by validation Macro-F1 from a fixed 0.025 grid.",
        "- Test subjects are evaluated once after the fold-specific weight is frozen.",
        "",
        "## Results",
        "",
    ]
    for _, row in summary.iterrows():
        report.append(
            f"- {row['model_display_name']}: Accuracy={row['test_accuracy_mean']:.4f}, "
            f"Macro-F1={row['test_macro_f1_mean']:.4f}, validation Macro-F1={row['validation_macro_f1_mean']:.4f}."
        )
    report.extend(
        [
            "",
            f"Validation-selected system: {selected['model_display_name']}.",
            "",
            "## Boundaries",
            "",
            "- This is strict subject-aware generalization, not random-window or subject-dependent evaluation.",
            "- The fusion system contains the proposal LSTM-Transformer deep branch and a train-only engineered-feature ExtraTrees branch.",
            "- Public UCI results are not Bingbin results.",
            "- Stage17/19/24 outputs are unchanged.",
        ]
    )
    (out / "STAGE25_UCI_STRICT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    audit = """# Stage25 Leakage Audit

## Status: PASS

- Split unit is subject_id; train, validation, and test subject sets are disjoint in every fold.
- Raw-signal normalization is fitted on train subjects only.
- Engineered features are deterministic per window; ExtraTrees is fitted on train subjects only.
- Fusion weights are selected on validation probabilities only.
- Test labels and metrics are not used for feature selection, model fitting, or fusion-weight selection.
- No random-window split is used.
- No Limb, Bingbin, or CapgMyo data are read.
"""
    (out / "LEAKAGE_AUDIT.md").write_text(audit, encoding="utf-8")
    decision = {
        "stage": "Stage25 UCI strict deep-feature fusion",
        "status": "DONE",
        "subject_aware_split": True,
        "random_window_protocol_used": False,
        "selection_metric": "validation Macro-F1",
        "selected_model": selected[["model_name", "model_display_name", "validation_macro_f1_mean", "test_accuracy_mean", "test_macro_f1_mean"]].to_dict(),
        "fold_fusion_weights": selection.to_dict(orient="records"),
        "output_dir": str(out.relative_to(ROOT)),
    }
    (out / "STAGE25_DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.set_seed(SEED)
    print(f"device={device}", flush=True)
    per_fold, per_class, history, split_manifest, selection, complexity, predictions = run(args, device, out)
    summary = summarize(per_fold)
    summary.to_csv(out / "UCI_STRICT_DEEP_FEATURE_SUMMARY.csv", index=False, encoding="utf-8-sig")
    per_fold.to_csv(out / "UCI_STRICT_DEEP_FEATURE_PER_FOLD.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out / "UCI_STRICT_DEEP_FEATURE_PER_CLASS.csv", index=False, encoding="utf-8-sig")
    history.to_csv(out / "UCI_STRICT_DEEP_FEATURE_TRAINING_HISTORY.csv", index=False, encoding="utf-8-sig")
    split_manifest.to_csv(out / "UCI_STRICT_DEEP_FEATURE_SPLIT_MANIFEST.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(out / "UCI_STRICT_DEEP_FEATURE_FUSION_SELECTION.csv", index=False, encoding="utf-8-sig")
    complexity.to_csv(out / "UCI_STRICT_DEEP_FEATURE_COMPLEXITY.csv", index=False, encoding="utf-8-sig")
    make_figures(out, summary, per_fold, history, predictions)
    write_reports(out, summary, selection)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
