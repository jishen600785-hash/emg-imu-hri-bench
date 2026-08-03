from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = Path(__file__).resolve().parent / "generate_nature_figures.py"
OUT = ROOT / "outputs" / "figures"
FIG = OUT / "figures"
SRC = OUT / "source_data"


def load_base_module():
    spec = importlib.util.spec_from_file_location("stage26_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_fold_data(base, data: dict[str, pd.DataFrame], dataset: str) -> pd.DataFrame:
    rows: list[dict] = []
    if dataset == "UCI EMG":
        baseline = data["stage19"][
            (data["stage19"]["dataset_name"] == "UCI_EMG_Data_for_Gestures")
            & (data["stage19"]["model_name"] == "CNN1D")
        ]
        for _, row in baseline.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "fold_id": row["fold_id"],
                    "model": "Stage19 converged 1D-CNN",
                    "accuracy": row["test_accuracy"],
                    "balanced_accuracy": row["test_balanced_accuracy"],
                    "macro_f1": row["test_macro_f1"],
                    "selected_by_validation": False,
                    "source_file": str(base.STAGE19_SUMMARY.relative_to(ROOT)),
                }
            )
        selected_model = "LSTM-Transformer + Feature Fusion"
        for _, row in data["uci_fold"].iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "fold_id": row["fold_id"],
                    "model": row["model_display_name"],
                    "accuracy": row["test_accuracy"],
                    "balanced_accuracy": row["test_balanced_accuracy"],
                    "macro_f1": row["test_macro_f1"],
                    "selected_by_validation": row["model_display_name"] == selected_model,
                    "source_file": str(base.UCI_FOLD.relative_to(ROOT)),
                }
            )
    elif dataset == "CapgMyo DB-a":
        baseline = data["stage19"][
            (data["stage19"]["dataset_name"] == "CapgMyo_DBa_subset")
            & (data["stage19"]["model_name"].isin(["CNN1D", "LSTMTransformer"]))
        ]
        for _, row in baseline.iterrows():
            model = (
                "Stage19 converged 1D-CNN"
                if row["model_name"] == "CNN1D"
                else "Stage19 converged LSTM-Transformer"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "fold_id": row["fold_id"],
                    "model": model,
                    "accuracy": row["test_accuracy"],
                    "balanced_accuracy": row["test_balanced_accuracy"],
                    "macro_f1": row["test_macro_f1"],
                    "selected_by_validation": False,
                    "source_file": str(base.STAGE19_SUMMARY.relative_to(ROOT)),
                }
            )
        selected_model = "HD-sEMG Spatial-Temporal ResNet-SE"
        for _, row in data["capg_fold"].iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "fold_id": row["fold_id"],
                    "model": row["model_display_name"],
                    "accuracy": row["test_accuracy"],
                    "balanced_accuracy": row["test_balanced_accuracy"],
                    "macro_f1": row["test_macro_f1"],
                    "selected_by_validation": row["model_display_name"] == selected_model,
                    "source_file": str(base.CAPG_FOLD.relative_to(ROOT)),
                }
            )
    else:
        raise ValueError(dataset)
    frame = pd.DataFrame(rows)
    for column in ("accuracy", "balanced_accuracy", "macro_f1"):
        frame[column] = pd.to_numeric(frame[column])
    return frame


def render_comparison(
    base,
    data: dict[str, pd.DataFrame],
    dataset: str,
    model_order: list[str],
    model_labels: list[str],
    protocol: str,
    y_limits: tuple[float, float],
    stem: str,
) -> tuple[dict[str, str], Path]:
    fold_data = build_fold_data(base, data, dataset)
    source_path = base.save_source(f"{stem}_source", fold_data)
    means = (
        fold_data.groupby(["model", "selected_by_validation"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .set_index("model")
        .loc[model_order]
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    x = np.arange(len(model_order), dtype=float)
    styles = {
        "accuracy": ("Accuracy", -0.20, "o", base.COLORS["navy"]),
        "balanced_accuracy": ("Balanced accuracy", 0.0, "s", base.COLORS["cyan"]),
        "macro_f1": ("Macro-F1", 0.20, "D", base.COLORS["red"]),
    }
    for index, model in enumerate(model_order):
        if bool(means.loc[model, "selected_by_validation"]):
            ax.axvspan(index - 0.46, index + 0.46, color=base.COLORS["red"], alpha=0.065, zorder=0)

    for metric, (label, offset, marker, color) in styles.items():
        metric_x = x + offset
        values = means[f"{metric}_mean"].to_numpy(dtype=float)
        errors = means[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
        ax.errorbar(
            metric_x,
            values,
            yerr=errors,
            fmt=marker,
            markersize=6.2,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            capthick=0.8,
            label=label,
            zorder=4,
        )
        for model_index, model in enumerate(model_order):
            fold_values = fold_data[fold_data["model"] == model][metric].to_numpy(dtype=float)
            jitter = np.linspace(-0.035, 0.035, len(fold_values))
            ax.scatter(
                np.full(len(fold_values), metric_x[model_index]) + jitter,
                fold_values,
                s=12,
                facecolor="white",
                edgecolor=color,
                linewidth=0.7,
                zorder=5,
            )
        if metric == "macro_f1":
            for model_index, value in enumerate(values):
                ax.text(
                    metric_x[model_index],
                    value + errors[model_index] + 0.009,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=base.COLORS["red"],
                )

    ax.set_xticks(x, model_labels)
    for index, tick in enumerate(ax.get_xticklabels()):
        if bool(means.iloc[index]["selected_by_validation"]):
            tick.set_color(base.COLORS["red"])
            tick.set_fontweight("bold")
    ax.set_ylim(*y_limits)
    ax.set_ylabel("Held-out test score")
    ax.set_xlabel("Model")
    ax.set_title(dataset, loc="left", fontweight="bold")
    ax.text(
        1.0,
        1.015,
        protocol,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=base.COLORS["gray"],
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.29), ncol=3)
    base.clean_axis(ax)
    return base.save_figure(fig, stem), source_path


def create_contact_sheet(png_paths: list[Path]) -> Path:
    canvases = []
    for path in png_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1200, 700), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1220, 740), "white")
        canvas.paste(image, ((canvas.width - image.width) // 2, 15))
        canvases.append(canvas)
    sheet = Image.new("RGB", (1220, 740 * len(canvases)), "white")
    for index, canvas in enumerate(canvases):
        sheet.paste(canvas, (0, index * 740))
    path = OUT / "separate_model_comparison_contact_sheet.png"
    sheet.save(path, dpi=(200, 200))
    return path


def verify(paths: list[Path]) -> None:
    for path in paths:
        minimum_size = 1000 if path.suffix.lower() in {".png", ".tiff", ".svg", ".pdf"} else 20
        if not path.exists() or path.stat().st_size < minimum_size:
            raise RuntimeError(f"Missing or empty output: {path}")
        if path.suffix.lower() in {".png", ".tiff"}:
            with Image.open(path) as image:
                image.verify()


def main() -> None:
    base = load_base_module()
    base.ensure_inputs()
    base.configure_style()
    data = base.load_data()
    uci_paths, uci_source = render_comparison(
        base,
        data,
        "UCI EMG",
        [
            "Stage19 converged 1D-CNN",
            "LSTM-Transformer",
            "Feature ExtraTrees",
            "LSTM-Transformer + Feature Fusion",
        ],
        ["1D-CNN", "LSTM-\nTransformer", "Feature\nExtraTrees", "LSTM-Transformer\n+ Feature Fusion"],
        "36 subjects | subject-aware 3-fold",
        (0.73, 0.865),
        "fig09_uci_independent_model_comparison",
    )
    capg_paths, capg_source = render_comparison(
        base,
        data,
        "CapgMyo DB-a",
        [
            "Stage19 converged LSTM-Transformer",
            "Stage19 converged 1D-CNN",
            "HD-sEMG Spatial RMS CNN",
            "HD-sEMG Spatial-Temporal ResNet-SE",
        ],
        ["LSTM-\nTransformer", "1D-CNN", "Spatial RMS\nCNN", "Spatial-Temporal\nResNet-SE"],
        "18 subjects | trial-aware calibrated 3-fold",
        (0.58, 1.01),
        "fig10_capgmyo_independent_model_comparison",
    )
    all_paths = [ROOT / relative for relative in [*uci_paths.values(), *capg_paths.values()]]
    verify(all_paths)
    contact = create_contact_sheet([ROOT / uci_paths["png"], ROOT / capg_paths["png"]])
    readme = OUT / "SEPARATE_MODEL_COMPARISON_README.md"
    readme.write_text(
        "# Separate public-dataset model comparison figures\n\n"
        "- Fig. 9: UCI EMG, 36 subjects, subject-aware 3-fold.\n"
        "- Fig. 10: CapgMyo DB-a, 18 subjects, trial-aware calibrated 3-fold.\n"
        "- Each figure reports Accuracy, Balanced Accuracy, Macro-F1, mean +/- SD, and all fold points.\n"
        "- Red model label and pale red band indicate the validation-selected model.\n"
        "- No training, split, or metric was changed.\n",
        encoding="utf-8",
    )
    verify([uci_source, capg_source, contact, readme])
    print(f"UCI figure: {ROOT / uci_paths['png']}")
    print(f"CapgMyo figure: {ROOT / capg_paths['png']}")
    print(f"Contact sheet: {contact}")
    print("QA: PASS")


if __name__ == "__main__":
    main()
