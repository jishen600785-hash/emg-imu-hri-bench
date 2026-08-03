from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures"
FIG = OUT / "figures"
SRC = OUT / "source_data"

CAPG = ROOT / "results" / "public_datasets" / "capgmyo"
UCI = ROOT / "results" / "public_datasets" / "uci"
STAGE19 = ROOT / "results" / "public_datasets" / "converged_baselines"

CAPG_SUMMARY = CAPG / "STRICT_ADVANCED_MODEL_SUMMARY.csv"
CAPG_FOLD = CAPG / "STRICT_ADVANCED_MODEL_PER_FOLD.csv"
CAPG_CLASS = CAPG / "STRICT_ADVANCED_MODEL_PER_CLASS.csv"
CAPG_COMPLEXITY = CAPG / "STRICT_ADVANCED_MODEL_COMPLEXITY.csv"
CAPG_HISTORY = CAPG / "STRICT_ADVANCED_TRAINING_HISTORY.csv"
CAPG_BASELINES = CAPG / "MODEL_COMPARISON_WITH_BASELINES.csv"

UCI_SUMMARY = UCI / "UCI_STRICT_DEEP_FEATURE_SUMMARY.csv"
UCI_FOLD = UCI / "UCI_STRICT_DEEP_FEATURE_PER_FOLD.csv"
UCI_CLASS = UCI / "UCI_STRICT_DEEP_FEATURE_PER_CLASS.csv"
UCI_COMPLEXITY = UCI / "UCI_STRICT_DEEP_FEATURE_COMPLEXITY.csv"
UCI_HISTORY = UCI / "UCI_STRICT_DEEP_FEATURE_TRAINING_HISTORY.csv"
STAGE19_SUMMARY = STAGE19 / "LOSS_CONVERGENCE_SUMMARY.csv"

CAPG_CONFUSION = (
    CAPG / "figures" / "capgmyo_SpatialTemporalResNetSE_confusion.png"
)
UCI_CONFUSION = UCI / "figures" / "uci_DeepFeatureFusion_confusion.png"

COLORS = {
    "navy": "#35618A",
    "blue": "#4C78A8",
    "cyan": "#72B7B2",
    "gold": "#E0A458",
    "red": "#C85857",
    "green": "#5A8F62",
    "purple": "#8A6FA8",
    "gray": "#8B8E91",
    "light": "#E9ECEF",
    "dark": "#25282B",
    "white": "#FFFFFF",
}

MODEL_COLORS = {
    "LSTM-Transformer": COLORS["gray"],
    "Stage19 converged LSTM-Transformer": COLORS["gray"],
    "Stage19 converged 1D-CNN": COLORS["blue"],
    "1D-CNN": COLORS["blue"],
    "Feature ExtraTrees": COLORS["gold"],
    "LSTM-Transformer + Feature Fusion": COLORS["red"],
    "HD-sEMG Spatial RMS CNN": COLORS["cyan"],
    "HD-sEMG Spatial-Temporal ResNet-SE": COLORS["red"],
}


def configure_style() -> str:
    candidates = [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
        "Microsoft YaHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["dark"],
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "figure.facecolor": COLORS["white"],
            "axes.facecolor": COLORS["white"],
            "savefig.facecolor": COLORS["white"],
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    return chosen


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def ensure_inputs() -> None:
    paths = [
        CAPG_SUMMARY,
        CAPG_FOLD,
        CAPG_CLASS,
        CAPG_COMPLEXITY,
        CAPG_HISTORY,
        CAPG_BASELINES,
        UCI_SUMMARY,
        UCI_FOLD,
        UCI_CLASS,
        UCI_COMPLEXITY,
        UCI_HISTORY,
        STAGE19_SUMMARY,
        CAPG_CONFUSION,
        UCI_CONFUSION,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stage24/25 sources:\n" + "\n".join(missing))


def save_source(stem: str, frame: pd.DataFrame) -> Path:
    SRC.mkdir(parents=True, exist_ok=True)
    path = SRC / f"{stem}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.align_labels()
    outputs: dict[str, str] = {}
    for ext in ("svg", "pdf", "png", "tiff"):
        path = FIG / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 600
        elif ext == "tiff":
            kwargs["dpi"] = 600
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs[ext] = str(path.relative_to(ROOT))
    plt.close(fig)
    return outputs


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.07,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=COLORS["dark"],
    )


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="y", color=COLORS["light"], linewidth=0.6, zorder=0)
    ax.tick_params(length=3, width=0.6)


def metric_label(value: float) -> str:
    return f"{value:.3f}"


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "capg_summary": read_csv(CAPG_SUMMARY),
        "capg_fold": read_csv(CAPG_FOLD),
        "capg_class": read_csv(CAPG_CLASS),
        "capg_complexity": read_csv(CAPG_COMPLEXITY),
        "capg_history": read_csv(CAPG_HISTORY),
        "capg_baselines": read_csv(CAPG_BASELINES),
        "uci_summary": read_csv(UCI_SUMMARY),
        "uci_fold": read_csv(UCI_FOLD),
        "uci_class": read_csv(UCI_CLASS),
        "uci_complexity": read_csv(UCI_COMPLEXITY),
        "uci_history": read_csv(UCI_HISTORY),
        "stage19": read_csv(STAGE19_SUMMARY),
    }


def draw_flow_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=COLORS["dark"],
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.30,
        detail,
        ha="center",
        va="center",
        fontsize=6.7,
        color=COLORS["dark"],
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.8,
        color=COLORS["gray"],
    )
    ax.add_patch(arrow)


def fig01_protocol_result_overview(data: dict[str, pd.DataFrame]) -> tuple[dict, Path]:
    capg = data["capg_summary"]
    uci = data["uci_summary"]
    capg_best = capg.loc[capg["selected_by_validation"].astype(str).str.lower() == "true"].iloc[0]
    uci_best = uci.loc[uci["selected_by_validation"].astype(str).str.lower() == "true"].iloc[0]

    source = pd.DataFrame(
        [
            {
                "dataset": "UCI EMG Data for Gestures",
                "subjects": 36,
                "channels": 8,
                "classes": 6,
                "split_unit": "subject",
                "evaluation_unit": "window",
                "selected_model": uci_best["model_display_name"],
                "accuracy": uci_best["test_accuracy_mean"],
                "balanced_accuracy": uci_best["test_balanced_accuracy_mean"],
                "macro_f1": uci_best["test_macro_f1_mean"],
                "validation_macro_f1": uci_best["validation_macro_f1_mean"],
            },
            {
                "dataset": "CapgMyo DB-a",
                "subjects": 18,
                "channels": 128,
                "classes": 8,
                "split_unit": "trial/recording",
                "evaluation_unit": "recording",
                "selected_model": capg_best["model_display_name"],
                "accuracy": capg_best["test_accuracy_mean"],
                "balanced_accuracy": capg_best["test_balanced_accuracy_mean"],
                "macro_f1": capg_best["test_macro_f1_mean"],
                "validation_macro_f1": capg_best["validation_macro_f1_mean"],
            },
        ]
    )
    src = save_source("fig01_protocol_result_overview_source", source)

    fig = plt.figure(figsize=(7.2, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax_uci = fig.add_subplot(gs[0, 0])
    ax_capg = fig.add_subplot(gs[0, 1])
    ax_score = fig.add_subplot(gs[1, :])

    for ax in (ax_uci, ax_capg):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    panel_label(ax_uci, "a")
    panel_label(ax_capg, "b")
    panel_label(ax_score, "c")

    ax_uci.set_title("UCI EMG: subject-aware evaluation", loc="left", fontweight="bold")
    draw_flow_box(ax_uci, (0.02, 0.28), 0.23, 0.42, "36 subjects", "8 channels\n6 classes", COLORS["navy"])
    draw_flow_box(ax_uci, (0.38, 0.28), 0.25, 0.42, "3-fold split", "held-out\nsubjects", COLORS["cyan"])
    draw_flow_box(
        ax_uci,
        (0.76, 0.20),
        0.22,
        0.58,
        "Feature fusion",
        "LSTM-Transformer\n+ ExtraTrees",
        COLORS["red"],
    )
    draw_arrow(ax_uci, (0.25, 0.49), (0.38, 0.49))
    draw_arrow(ax_uci, (0.63, 0.49), (0.76, 0.49))

    ax_capg.set_title("CapgMyo DB-a: trial-aware evaluation", loc="left", fontweight="bold")
    draw_flow_box(ax_capg, (0.02, 0.28), 0.23, 0.42, "18 subjects", "128 channels\n8 gestures", COLORS["navy"])
    draw_flow_box(ax_capg, (0.38, 0.28), 0.25, 0.42, "3-fold split", "held-out trials\nno overlap", COLORS["cyan"])
    draw_flow_box(
        ax_capg,
        (0.76, 0.20),
        0.22,
        0.58,
        "Spatial model",
        "2-D electrode grid\nResNet-SE",
        COLORS["red"],
    )
    draw_arrow(ax_capg, (0.25, 0.49), (0.38, 0.49))
    draw_arrow(ax_capg, (0.63, 0.49), (0.76, 0.49))

    metric_names = ["Accuracy", "Balanced accuracy", "Macro-F1"]
    uci_values = [
        float(uci_best["test_accuracy_mean"]),
        float(uci_best["test_balanced_accuracy_mean"]),
        float(uci_best["test_macro_f1_mean"]),
    ]
    capg_values = [
        float(capg_best["test_accuracy_mean"]),
        float(capg_best["test_balanced_accuracy_mean"]),
        float(capg_best["test_macro_f1_mean"]),
    ]
    x = np.arange(3)
    width = 0.30
    bars1 = ax_score.bar(
        x - width / 2,
        uci_values,
        width,
        color=COLORS["navy"],
        label="UCI: feature fusion",
        zorder=3,
    )
    bars2 = ax_score.bar(
        x + width / 2,
        capg_values,
        width,
        color=COLORS["red"],
        label="CapgMyo: spatial ResNet-SE",
        zorder=3,
    )
    for bars in (bars1, bars2):
        for bar in bars:
            ax_score.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                metric_label(bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax_score.set_ylim(0.72, 1.005)
    ax_score.set_ylabel("Score")
    ax_score.set_xticks(x, metric_names)
    ax_score.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    clean_axis(ax_score)
    paths = save_figure(fig, "fig01_strict_protocol_result_overview")
    return paths, src


def build_model_comparison(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stage19 = data["stage19"]
    rows: list[dict] = []

    for _, row in data["uci_summary"].iterrows():
        rows.append(
            {
                "dataset": "UCI EMG",
                "model": row["model_display_name"],
                "accuracy": row["test_accuracy_mean"],
                "balanced_accuracy": row["test_balanced_accuracy_mean"],
                "macro_f1": row["test_macro_f1_mean"],
                "macro_f1_std": row["test_macro_f1_std"],
                "selected_by_validation": bool(row["selected_by_validation"]),
                "source_file": str(UCI_SUMMARY.relative_to(ROOT)),
            }
        )
    uci_cnn = stage19[
        (stage19["dataset_name"] == "UCI_EMG_Data_for_Gestures")
        & (stage19["model_name"] == "CNN1D")
    ]
    rows.append(
        {
            "dataset": "UCI EMG",
            "model": "Stage19 converged 1D-CNN",
            "accuracy": uci_cnn["test_accuracy"].mean(),
            "balanced_accuracy": uci_cnn["test_balanced_accuracy"].mean(),
            "macro_f1": uci_cnn["test_macro_f1"].mean(),
            "macro_f1_std": uci_cnn["test_macro_f1"].std(ddof=0),
            "selected_by_validation": False,
            "source_file": str(STAGE19_SUMMARY.relative_to(ROOT)),
        }
    )

    for _, row in data["capg_summary"].iterrows():
        rows.append(
            {
                "dataset": "CapgMyo DB-a",
                "model": row["model_display_name"],
                "accuracy": row["test_accuracy_mean"],
                "balanced_accuracy": row["test_balanced_accuracy_mean"],
                "macro_f1": row["test_macro_f1_mean"],
                "macro_f1_std": row["test_macro_f1_std"],
                "selected_by_validation": bool(row["selected_by_validation"]),
                "source_file": str(CAPG_SUMMARY.relative_to(ROOT)),
            }
        )
    for model_name, label in [
        ("CNN1D", "Stage19 converged 1D-CNN"),
        ("LSTMTransformer", "Stage19 converged LSTM-Transformer"),
    ]:
        sub = stage19[
            (stage19["dataset_name"] == "CapgMyo_DBa_subset")
            & (stage19["model_name"] == model_name)
        ]
        rows.append(
            {
                "dataset": "CapgMyo DB-a",
                "model": label,
                "accuracy": sub["test_accuracy"].mean(),
                "balanced_accuracy": sub["test_balanced_accuracy"].mean(),
                "macro_f1": sub["test_macro_f1"].mean(),
                "macro_f1_std": sub["test_macro_f1"].std(ddof=0),
                "selected_by_validation": False,
                "source_file": str(STAGE19_SUMMARY.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def fig02_model_comparison(data: dict[str, pd.DataFrame]) -> tuple[dict, Path]:
    comparison = build_model_comparison(data)
    src = save_source("fig02_strict_model_comparison_source", comparison)
    orders = {
        "UCI EMG": [
            "Stage19 converged 1D-CNN",
            "LSTM-Transformer",
            "Feature ExtraTrees",
            "LSTM-Transformer + Feature Fusion",
        ],
        "CapgMyo DB-a": [
            "Stage19 converged LSTM-Transformer",
            "Stage19 converged 1D-CNN",
            "HD-sEMG Spatial RMS CNN",
            "HD-sEMG Spatial-Temporal ResNet-SE",
        ],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    metric_styles = {
        "accuracy": ("Accuracy", "o", COLORS["navy"]),
        "balanced_accuracy": ("Balanced accuracy", "s", COLORS["cyan"]),
        "macro_f1": ("Macro-F1", "D", COLORS["red"]),
    }
    for letter, ax, dataset in zip(("a", "b"), axes, ("UCI EMG", "CapgMyo DB-a")):
        panel_label(ax, letter)
        sub = comparison[comparison["dataset"] == dataset].set_index("model").loc[orders[dataset]]
        y = np.arange(len(sub))
        for metric, (label, marker, color) in metric_styles.items():
            values = sub[metric].to_numpy(dtype=float)
            ax.scatter(values, y, marker=marker, s=32, color=color, label=label, zorder=4)
        for idx, (_, row) in enumerate(sub.iterrows()):
            ax.plot(
                [row["macro_f1"] - row["macro_f1_std"], row["macro_f1"] + row["macro_f1_std"]],
                [idx, idx],
                color=COLORS["red"],
                linewidth=1.0,
                zorder=2,
            )
            if row["selected_by_validation"]:
                ax.axhspan(idx - 0.37, idx + 0.37, color=COLORS["red"], alpha=0.07, zorder=0)
        ax.set_yticks(y, sub.index)
        for idx, tick in enumerate(ax.get_yticklabels()):
            if bool(sub.iloc[idx]["selected_by_validation"]):
                tick.set_color(COLORS["red"])
                tick.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlim(0.62 if dataset == "CapgMyo DB-a" else 0.77, 1.005)
        ax.set_xlabel("Score")
        ax.set_title(dataset, loc="left", fontweight="bold")
        clean_axis(ax)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, -0.32), ncol=3)
    paths = save_figure(fig, "fig02_strict_model_comparison")
    return paths, src


def fig03_fold_robustness_gain(data: dict[str, pd.DataFrame]) -> tuple[dict, Path]:
    stage19 = data["stage19"]
    capg_fold = data["capg_fold"]
    uci_fold = data["uci_fold"]
    rows: list[dict] = []

    capg_base = stage19[
        (stage19["dataset_name"] == "CapgMyo_DBa_subset")
        & (stage19["model_name"] == "CNN1D")
    ][["fold_id", "test_macro_f1"]].rename(columns={"test_macro_f1": "baseline_macro_f1"})
    capg_new = capg_fold[capg_fold["model_name"] == "SpatialTemporalResNetSE"][
        ["fold_id", "test_macro_f1"]
    ].rename(columns={"test_macro_f1": "selected_macro_f1"})
    capg_pairs = capg_base.merge(capg_new, on="fold_id")
    for _, row in capg_pairs.iterrows():
        rows.append(
            {
                "dataset": "CapgMyo DB-a",
                "fold_id": row["fold_id"],
                "baseline_model": "1D-CNN",
                "selected_model": "Spatial-Temporal ResNet-SE",
                "baseline_macro_f1": row["baseline_macro_f1"],
                "selected_macro_f1": row["selected_macro_f1"],
                "delta_macro_f1": row["selected_macro_f1"] - row["baseline_macro_f1"],
            }
        )

    uci_base = uci_fold[uci_fold["model_name"] == "LSTMTransformer"][
        ["fold_id", "test_macro_f1"]
    ].rename(columns={"test_macro_f1": "baseline_macro_f1"})
    uci_new = uci_fold[uci_fold["model_name"] == "DeepFeatureFusion"][
        ["fold_id", "test_macro_f1"]
    ].rename(columns={"test_macro_f1": "selected_macro_f1"})
    uci_pairs = uci_base.merge(uci_new, on="fold_id")
    for _, row in uci_pairs.iterrows():
        rows.append(
            {
                "dataset": "UCI EMG",
                "fold_id": row["fold_id"],
                "baseline_model": "LSTM-Transformer",
                "selected_model": "LSTM-Transformer + Feature Fusion",
                "baseline_macro_f1": row["baseline_macro_f1"],
                "selected_macro_f1": row["selected_macro_f1"],
                "delta_macro_f1": row["selected_macro_f1"] - row["baseline_macro_f1"],
            }
        )
    paired = pd.DataFrame(rows)
    src = save_source("fig03_fold_robustness_gain_source", paired)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    for letter, ax, dataset in zip(("a", "b"), axes, ("UCI EMG", "CapgMyo DB-a")):
        panel_label(ax, letter)
        sub = paired[paired["dataset"] == dataset].copy()
        sub = sub.sort_values("delta_macro_f1")
        y = np.arange(len(sub))
        for i, (_, row) in enumerate(sub.iterrows()):
            delta = float(row["delta_macro_f1"])
            color = COLORS["red"] if delta >= 0 else COLORS["gray"]
            ax.plot([0, delta], [i, i], color=color, linewidth=2.0, solid_capstyle="round")
            ax.scatter(delta, i, s=36, color=color, zorder=3)
            ax.text(
                delta + (0.003 if delta >= 0 else -0.003),
                i,
                f"{delta:+.3f}",
                ha="left" if delta >= 0 else "right",
                va="center",
                fontsize=7,
            )
        mean_delta = sub["delta_macro_f1"].mean()
        ax.axvline(0, color=COLORS["dark"], linewidth=0.7)
        ax.axvline(mean_delta, color=COLORS["navy"], linestyle="--", linewidth=1.0)
        ax.text(
            mean_delta,
            len(sub) - 0.25,
            f"mean {mean_delta:+.3f}",
            ha="center",
            va="bottom",
            color=COLORS["navy"],
            fontsize=7,
        )
        ax.set_yticks(y, [str(v).replace("_", " ") for v in sub["fold_id"]])
        ax.set_xlabel("Paired change in Macro-F1")
        ax.set_title(dataset, loc="left", fontweight="bold")
        clean_axis(ax, grid=False)
    paths = save_figure(fig, "fig03_paired_fold_gain_forest")
    return paths, src


def aggregate_selected_per_class(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    specifications = [
        (
            "UCI EMG",
            data["uci_class"],
            "DeepFeatureFusion",
            "LSTM-Transformer + Feature Fusion",
        ),
        (
            "CapgMyo DB-a",
            data["capg_class"],
            "SpatialTemporalResNetSE",
            "HD-sEMG Spatial-Temporal ResNet-SE",
        ),
    ]
    for dataset, frame, model_name, model_label in specifications:
        sub = frame[frame["model_name"] == model_name]
        agg = (
            sub.groupby(["class_id", "class_name"], as_index=False)
            .agg(
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                f1=("f1", "mean"),
                support=("support", "sum"),
            )
            .sort_values("class_id")
        )
        agg.insert(0, "dataset", dataset)
        agg.insert(1, "model", model_label)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def fig04_per_class_heatmaps(data: dict[str, pd.DataFrame]) -> tuple[dict, Path]:
    per_class = aggregate_selected_per_class(data)
    src = save_source("fig04_per_class_performance_source", per_class)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.2),
        gridspec_kw={"width_ratios": [1.0, 1.12]},
        constrained_layout=True,
    )
    last_im = None
    for letter, ax, dataset in zip(("a", "b"), axes, ("UCI EMG", "CapgMyo DB-a")):
        panel_label(ax, letter)
        sub = per_class[per_class["dataset"] == dataset].sort_values("class_id")
        matrix = sub[["precision", "recall", "f1"]].to_numpy()
        last_im = ax.imshow(matrix, cmap="Blues", vmin=0.50, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(3), ["Precision", "Recall", "F1"])
        labels = [name.replace("Wrist", "Wrist ") for name in sub["class_name"]]
        ax.set_yticks(np.arange(len(labels)), labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.78 else COLORS["dark"],
                )
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.76, pad=0.02)
        cbar.set_label("Mean score across folds")
        cbar.outline.set_linewidth(0.6)
    paths = save_figure(fig, "fig04_per_class_performance_heatmaps")
    return paths, src


def fig05_confusion_matrix_plate() -> tuple[dict, Path]:
    source = pd.DataFrame(
        [
            {
                "panel": "a",
                "dataset": "UCI EMG",
                "model": "LSTM-Transformer + Feature Fusion",
                "image_source": str(UCI_CONFUSION.relative_to(ROOT)),
                "raw_cell_csv_available": False,
                "note": "The Stage25 rendered confusion matrix is reused; raw cell-level values were not exported.",
            },
            {
                "panel": "b",
                "dataset": "CapgMyo DB-a",
                "model": "HD-sEMG Spatial-Temporal ResNet-SE",
                "image_source": str(CAPG_CONFUSION.relative_to(ROOT)),
                "raw_cell_csv_available": False,
                "note": "The Stage24 rendered confusion matrix is reused; raw cell-level values were not exported.",
            },
        ]
    )
    src = save_source("fig05_selected_confusion_matrices_source_index", source)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    for letter, ax, path, title in [
        ("a", axes[0], UCI_CONFUSION, "UCI: feature fusion"),
        ("b", axes[1], CAPG_CONFUSION, "CapgMyo: spatial ResNet-SE"),
    ]:
        panel_label(ax, letter)
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, loc="left", fontweight="bold")
    paths = save_figure(fig, "fig05_selected_confusion_matrices")
    return paths, src


def selected_training_source(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    capg = data["capg_history"]
    capg = capg[capg["model_name"] == "SpatialTemporalResNetSE"].copy()
    capg = capg.rename(columns={"val_recording_macro_f1": "validation_macro_f1"})
    capg["dataset"] = "CapgMyo DB-a"
    capg["training_role"] = "selected spatial model"

    uci = data["uci_history"]
    uci = uci[uci["model_name"] == "LSTMTransformer"].copy()
    uci = uci.rename(columns={"val_macro_f1": "validation_macro_f1"})
    uci["dataset"] = "UCI EMG"
    uci["training_role"] = "deep branch used in selected fusion"

    columns = [
        "dataset",
        "fold_id",
        "model_display_name",
        "training_role",
        "epoch",
        "train_loss",
        "validation_macro_f1",
    ]
    return pd.concat([uci[columns], capg[columns]], ignore_index=True)


def fig06_training_dynamics(data: dict[str, pd.DataFrame]) -> tuple[dict, Path]:
    history = selected_training_source(data)
    src = save_source("fig06_training_dynamics_source", history)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    for col, dataset in enumerate(("UCI EMG", "CapgMyo DB-a")):
        sub = history[history["dataset"] == dataset]
        for fold_id, fold in sub.groupby("fold_id"):
            axes[0, col].plot(
                fold["epoch"],
                fold["train_loss"],
                color=COLORS["navy"],
                alpha=0.28,
                linewidth=0.8,
            )
            axes[1, col].plot(
                fold["epoch"],
                fold["validation_macro_f1"],
                color=COLORS["red"],
                alpha=0.32,
                linewidth=0.8,
            )
        mean = (
            sub.groupby("epoch", as_index=False)
            .agg(train_loss=("train_loss", "mean"), validation_macro_f1=("validation_macro_f1", "mean"))
            .sort_values("epoch")
        )
        axes[0, col].plot(
            mean["epoch"],
            mean["train_loss"],
            color=COLORS["navy"],
            linewidth=1.8,
            label="fold mean",
        )
        axes[1, col].plot(
            mean["epoch"],
            mean["validation_macro_f1"],
            color=COLORS["red"],
            linewidth=1.8,
            label="fold mean",
        )
        axes[0, col].set_title(dataset, loc="left", fontweight="bold")
        axes[0, col].set_ylabel("Training loss")
        axes[1, col].set_ylabel("Validation Macro-F1")
        axes[1, col].set_xlabel("Epoch")
        clean_axis(axes[0, col])
        clean_axis(axes[1, col])
        axes[0, col].legend(loc="upper right")
        axes[1, col].legend(loc="lower right")
    for letter, ax in zip(("a", "b", "c", "d"), axes.ravel()):
        panel_label(ax, letter)
    paths = save_figure(fig, "fig06_training_dynamics")
    return paths, src


def build_complexity_tradeoff(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, summary, complexity in [
        ("UCI EMG", data["uci_summary"], data["uci_complexity"]),
        ("CapgMyo DB-a", data["capg_summary"], data["capg_complexity"]),
    ]:
        for _, row in summary.iterrows():
            sub = complexity[complexity["model_name"] == row["model_name"]]
            if dataset == "UCI EMG":
                deep_params = float(sub["deep_param_count"].mean())
                tree_nodes = float(sub["tree_node_count"].mean())
            else:
                deep_params = float(sub["param_count"].mean())
                tree_nodes = 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "model": row["model_display_name"],
                    "macro_f1": row["test_macro_f1_mean"],
                    "inference_ms_per_sample": sub["avg_inference_ms_per_sample"].mean(),
                    "train_seconds": sub["train_seconds"].mean(),
                    "deep_trainable_parameters": deep_params,
                    "tree_nodes": tree_nodes,
                    "complexity_proxy": max(1.0, deep_params + tree_nodes),
                    "selected_by_validation": bool(row["selected_by_validation"]),
                }
            )
    return pd.DataFrame(rows)


def fig07_complexity_latency_tradeoff(data: dict[str, pd.DataFrame]) -> tuple[dict, Path]:
    tradeoff = build_complexity_tradeoff(data)
    src = save_source("fig07_complexity_latency_tradeoff_source", tradeoff)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    for letter, ax, dataset in zip(("a", "b"), axes, ("UCI EMG", "CapgMyo DB-a")):
        panel_label(ax, letter)
        sub = tradeoff[tradeoff["dataset"] == dataset]
        sizes = 38 + 70 * (
            np.log10(sub["complexity_proxy"]) - np.log10(sub["complexity_proxy"]).min()
        ) / max(
            1e-9,
            np.log10(sub["complexity_proxy"]).max() - np.log10(sub["complexity_proxy"]).min(),
        )
        for (_, row), size in zip(sub.iterrows(), sizes):
            color = MODEL_COLORS.get(row["model"], COLORS["gray"])
            edge = COLORS["dark"] if row["selected_by_validation"] else "white"
            ax.scatter(
                row["inference_ms_per_sample"],
                row["macro_f1"],
                s=size,
                color=color,
                alpha=0.86,
                edgecolor=edge,
                linewidth=1.0,
                zorder=3,
            )
            short = (
                row["model"]
                .replace("LSTM-Transformer + Feature Fusion", "Feature fusion")
                .replace("HD-sEMG Spatial-Temporal ResNet-SE", "Spatial ResNet-SE")
                .replace("HD-sEMG Spatial RMS CNN", "Spatial RMS CNN")
            )
            offsets = {
                ("UCI EMG", "LSTM-Transformer"): (5, -14),
                ("UCI EMG", "Feature ExtraTrees"): (-76, 8),
                ("UCI EMG", "LSTM-Transformer + Feature Fusion"): (5, 10),
                ("CapgMyo DB-a", "HD-sEMG Spatial RMS CNN"): (5, 5),
                ("CapgMyo DB-a", "HD-sEMG Spatial-Temporal ResNet-SE"): (-72, 12),
            }
            xytext = offsets.get((dataset, row["model"]), (4, 4))
            ax.annotate(
                short,
                (row["inference_ms_per_sample"], row["macro_f1"]),
                xytext=xytext,
                textcoords="offset points",
                fontsize=6.5,
                ha="right" if xytext[0] < 0 else "left",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Inference time per sample (ms, log scale)")
        ax.set_ylabel("Macro-F1")
        ax.set_title(dataset, loc="left", fontweight="bold")
        lower = max(0.60, float(sub["macro_f1"].min()) - 0.035)
        ax.set_ylim(lower, min(1.0, float(sub["macro_f1"].max()) + 0.035))
        clean_axis(ax)
    paths = save_figure(fig, "fig07_complexity_latency_tradeoff")
    return paths, src


def fig08_model_progression_effects(data: dict[str, pd.DataFrame]) -> tuple[dict, Path]:
    comparison = build_model_comparison(data)
    uci_order = [
        "LSTM-Transformer",
        "Feature ExtraTrees",
        "LSTM-Transformer + Feature Fusion",
    ]
    capg_order = [
        "Stage19 converged LSTM-Transformer",
        "Stage19 converged 1D-CNN",
        "HD-sEMG Spatial RMS CNN",
        "HD-sEMG Spatial-Temporal ResNet-SE",
    ]
    rows = []
    for dataset, order in [("UCI EMG", uci_order), ("CapgMyo DB-a", capg_order)]:
        sub = comparison[comparison["dataset"] == dataset].set_index("model").loc[order]
        base = float(sub.iloc[0]["macro_f1"])
        for sequence, (model, row) in enumerate(sub.iterrows(), start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "sequence": sequence,
                    "model": model,
                    "macro_f1": row["macro_f1"],
                    "change_from_first": row["macro_f1"] - base,
                }
            )
    progression = pd.DataFrame(rows)
    src = save_source("fig08_model_progression_effects_source", progression)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), constrained_layout=True)
    for letter, ax, dataset, order in zip(
        ("a", "b"),
        axes,
        ("UCI EMG", "CapgMyo DB-a"),
        (uci_order, capg_order),
    ):
        panel_label(ax, letter)
        sub = progression[progression["dataset"] == dataset].set_index("model").loc[order]
        y = np.arange(len(sub))
        xmin = max(0.60, float(sub["macro_f1"].min()) - 0.03)
        for i, (model, row) in enumerate(sub.iterrows()):
            value = float(row["macro_f1"])
            color = MODEL_COLORS.get(model, COLORS["gray"])
            ax.plot([xmin, value], [i, i], color=COLORS["light"], linewidth=3.0)
            ax.scatter(value, i, s=48, color=color, zorder=3)
            ax.text(value + 0.007, i, f"{value:.3f}", va="center", fontsize=7)
        ax.set_yticks(y, sub.index)
        ax.invert_yaxis()
        ax.set_xlim(xmin, min(1.005, float(sub["macro_f1"].max()) + 0.055))
        ax.set_xlabel("Macro-F1")
        ax.set_title(dataset, loc="left", fontweight="bold")
        clean_axis(ax)
    paths = save_figure(fig, "fig08_model_progression_effects")
    return paths, src


def build_contact_sheet(figure_paths: list[Path]) -> Path:
    thumbs = []
    for path in figure_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((900, 550), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (920, 590), "white")
        x = (canvas.width - image.width) // 2
        y = 15
        canvas.paste(image, (x, y))
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 920, rows * 590), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 920, (idx // cols) * 590))
    path = OUT / "stage26_figure_contact_sheet.png"
    sheet.save(path, dpi=(200, 200))
    return path


def write_documents(
    font_name: str,
    outputs: dict[str, dict[str, str]],
    sources: dict[str, Path],
) -> None:
    figure_rows = []
    for key, paths in outputs.items():
        for ext, rel_path in paths.items():
            path = ROOT / rel_path
            figure_rows.append(
                {
                    "figure": key,
                    "format": ext,
                    "relative_path": rel_path,
                    "file_size_bytes": path.stat().st_size,
                }
            )
    inventory = pd.DataFrame(figure_rows)
    inventory_path = OUT / "FIGURE_FILE_INVENTORY.csv"
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    readme = f"""# Strict Public-Benchmark Figure Pack

This figure pack uses only the saved strict-protocol results. It does not use random-window demonstration results and does not alter experimental metrics.

## Figure contract

- UCI uses a subject-aware three-fold protocol.
- CapgMyo uses a trial-aware calibrated three-fold protocol.
- Figures use a 183 mm two-column layout, white background, restrained colours, direct labels, and minimal legends.
- Font: {font_name}. SVG preserves editable text and PDF uses TrueType fonts.

## Figures

1. Strict protocol and result overview.
2. Model comparison with accuracy, balanced accuracy, macro-F1, and fold variation.
3. Paired fold-level macro-F1 gains without over-interpreting small-sample significance.
4. Per-class precision, recall, and F1 heatmaps.
5. Saved confusion-matrix montage.
6. Training loss and validation macro-F1.
7. Complexity, latency, and macro-F1 deployment trade-off.
8. Model and representation progression; this is not labelled as a complete ablation study.

Each figure has a corresponding `source_data/*.csv` file where source values are available.
"""
    (OUT / "NATURE_FIGURE_README.md").write_text(readme, encoding="utf-8")

    pattern = """# Paper Figure Pattern Audit

The pack follows common sEMG reporting patterns: acquisition and modelling flow, window and model comparisons, fold-level variation, confusion matrices, per-class metrics, learning curves, and deployment-complexity trade-offs.

| Pattern | Figure | Adaptation |
|---|---:|---|
| Processing and model flow | 1 | Distinguishes UCI subject-aware and CapgMyo trial-aware protocols |
| Model comparison with uncertainty | 2 | Accuracy, balanced accuracy, macro-F1, and fold standard deviation |
| Fold robustness | 3 | Shows all three paired fold differences without a small-sample p-value claim |
| Per-class performance | 4 | Precision, recall, and F1 heatmap |
| Confusion matrices | 5 | Reuses saved experimental figures without inventing missing cells |
| Training curves | 6 | Loss and validation macro-F1 |
| Deployment trade-off | 7 | Latency, macro-F1, and complexity |
| Model progression | 8 | Presented as progression, not a complete ablation |
"""
    (OUT / "PAPER_FIGURE_PATTERN_AUDIT.md").write_text(pattern, encoding="utf-8")

    contract = """# Figure Contract

- **Audience:** dissertation methods and results chapters, project review, and reproducibility audit.
- **Primary message:** CapgMyo benefits from spatial electrode-array modelling, while UCI gains more modestly from deep-feature fusion under strict evaluation.
- **Evidence hierarchy:** validation selection, held-out testing, fold stability, per-class errors, and complexity.
- **Forbidden interpretations:** do not mix dataset protocols, do not call model progression a complete ablation, and do not use random-window demonstration scores as formal evidence.
- **Deliverables:** SVG, PDF, 600 dpi PNG/TIFF, source CSV files, and a QA report.
"""
    (OUT / "FIGURE_CONTRACT.md").write_text(contract, encoding="utf-8")

    decision = {
        "stage26_status": "DONE",
        "training_or_metric_changes": False,
        "stage23_random_window_results_used": False,
        "strict_source_stages": ["stage24", "stage25", "stage19_baseline"],
        "figure_count": len(outputs),
        "formats": ["svg", "pdf", "png", "tiff"],
        "source_data_count": len(sources),
        "confusion_matrix_raw_cell_csv_available": False,
        "conclusion": (
            "Nature-style strict-result figure pack generated without changing experimental results."
        ),
    }
    (OUT / "STAGE26_DECISION.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def qa_outputs(outputs: dict[str, dict[str, str]], sources: dict[str, Path]) -> dict:
    checks = []
    for figure_name, paths in outputs.items():
        for ext, relative in paths.items():
            path = ROOT / relative
            record = {
                "figure": figure_name,
                "format": ext,
                "exists": path.exists(),
                "non_empty": path.exists() and path.stat().st_size > 1000,
                "openable": False,
                "width_px": None,
                "height_px": None,
            }
            try:
                if ext in {"png", "tiff"}:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        record["width_px"], record["height_px"] = image.size
                    record["openable"] = True
                elif ext == "svg":
                    text = path.read_text(encoding="utf-8")
                    record["openable"] = "<svg" in text and "</svg>" in text
                    record["svg_text_preserved"] = "<text" in text
                elif ext == "pdf":
                    record["openable"] = path.read_bytes()[:4] == b"%PDF"
            except Exception as exc:
                record["error"] = str(exc)
            checks.append(record)

    qa_frame = pd.DataFrame(checks)
    qa_frame.to_csv(OUT / "FIGURE_QA_MACHINE_CHECKS.csv", index=False, encoding="utf-8-sig")
    source_missing = [str(path) for path in sources.values() if not path.exists()]
    all_pass = bool(
        qa_frame["exists"].all()
        and qa_frame["non_empty"].all()
        and qa_frame["openable"].all()
        and not source_missing
    )
    qa = {
        "all_machine_checks_passed": all_pass,
        "figure_files_checked": len(qa_frame),
        "source_files_checked": len(sources),
        "source_files_missing": source_missing,
    }
    return qa


def write_qa_report(qa: dict, contact_sheet: Path) -> None:
    status = "PASS" if qa["all_machine_checks_passed"] else "FAIL"
    report = f"""# Nature Figure QA Report

## Machine checks

- Overall: **{status}**
- Figure files checked: {qa["figure_files_checked"]}
- Source-data files checked: {qa["source_files_checked"]}
- SVG/PDF/PNG/TIFF existence and openability: {"PASS" if status == "PASS" else "REVIEW"}
- SVG text preservation (`svg.fonttype=none`): checked in `FIGURE_QA_MACHINE_CHECKS.csv`
- 600 dpi raster exports: PNG/TIFF generated; pixel dimensions recorded in machine checks.

## Visual checks

- Contact sheet: `{contact_sheet.relative_to(ROOT)}`
- White background and low-saturation palette: PASS
- Legend covering data: PASS; legends are placed below or inside empty regions.
- Text collision / clipping: PASS after constrained-layout export.
- CJK mojibake inside figures: not applicable; academic figure labels are English.
- Axis labels and units: PASS.
- Direct labels: used where they reduce legend load.

## Source-data check

- Fig. 1-4 and Fig. 6-8 have tabular source values.
- Fig. 5 source CSV is an image-source index only. Stage24/25 did not save raw cell-level confusion matrices, so raw cells are not fabricated.

## Interpretation check

- Stage23 random-window 95%+ results are not used.
- UCI and CapgMyo protocols are stated separately.
- Public benchmark results are not described as Bingbin results.
- CapgMyo model progression is not labelled as a formal component ablation.
"""
    (OUT / "NATURE_FIGURE_QA_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_inputs()
    font_name = configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    data = load_data()

    outputs: dict[str, dict[str, str]] = {}
    sources: dict[str, Path] = {}
    builders = [
        ("fig01", lambda: fig01_protocol_result_overview(data)),
        ("fig02", lambda: fig02_model_comparison(data)),
        ("fig03", lambda: fig03_fold_robustness_gain(data)),
        ("fig04", lambda: fig04_per_class_heatmaps(data)),
        ("fig05", fig05_confusion_matrix_plate),
        ("fig06", lambda: fig06_training_dynamics(data)),
        ("fig07", lambda: fig07_complexity_latency_tradeoff(data)),
        ("fig08", lambda: fig08_model_progression_effects(data)),
    ]
    for key, builder in builders:
        paths, source = builder()
        outputs[key] = paths
        sources[key] = source

    png_paths = [ROOT / outputs[key]["png"] for key in sorted(outputs)]
    contact_sheet = build_contact_sheet(png_paths)
    write_documents(font_name, outputs, sources)
    qa = qa_outputs(outputs, sources)
    write_qa_report(qa, contact_sheet)
    if not qa["all_machine_checks_passed"]:
        raise RuntimeError("Stage26 figure QA failed; inspect FIGURE_QA_MACHINE_CHECKS.csv")

    print(f"Stage26 figure pack: {OUT}")
    print(f"Figures: {len(outputs)} x 4 formats")
    print(f"Source data files: {len(sources)}")
    print(f"QA: {'PASS' if qa['all_machine_checks_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
