from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def load_limb(subject_id: int):
    path = ROOT / "models" / "limb_personalized" / f"limb_subject{subject_id:02d}_deployment.joblib"
    artifact = joblib.load(path)
    return artifact["pipeline"], artifact["metadata"]


def load_bingbin(subject_id: int):
    path = ROOT / "models" / "bingbin_realtime" / f"bingbin_subject{subject_id:02d}_deployment.joblib"
    artifact = joblib.load(path)
    return artifact["pipeline"], artifact["metadata"]


def load_uci_components(fold_id: str):
    deep_state = torch.load(
        ROOT / "models" / "uci_feature_fusion" / f"{fold_id}_lstm_transformer.pt",
        map_location="cpu",
        weights_only=True,
    )
    feature_model = joblib.load(ROOT / "models" / "uci_feature_fusion" / f"{fold_id}_extra_trees.joblib")
    config = json.loads((ROOT / "configs" / "uci_feature_fusion.json").read_text(encoding="utf-8"))
    return deep_state, feature_model, config


def load_capgmyo(fold_id: str):
    checkpoint = torch.load(
        ROOT / "models" / "capgmyo_spatiotemporal" / f"{fold_id}_spatial_temporal_resnet_se.pt",
        map_location="cpu",
        weights_only=False,
    )
    normalization = np.load(
        ROOT / "models" / "capgmyo_spatiotemporal" / f"{fold_id}_subject_normalization.npz"
    )
    return checkpoint, normalization


if __name__ == "__main__":
    limb, limb_meta = load_limb(1)
    bingbin, bingbin_meta = load_bingbin(1)
    print("Limb model:", type(limb).__name__, limb_meta["feature_dim"])
    print("Bingbin model:", type(bingbin).__name__, bingbin_meta["candidate_name"])
    print("Use the preprocessing functions in the corresponding training script before prediction.")
