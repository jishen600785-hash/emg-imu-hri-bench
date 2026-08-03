from __future__ import annotations


FS = 1260
INPUT_CHANNELS = 42

CONDITION_ORDER = [
    "StaticP1",
    "StaticP2",
    "StaticP3",
    "StaticP4",
    "StaticP9",
    "StaticP10",
    "StaticP14",
    "Dynamic",
]
STRICT_FOLD_ARTIFACT_ROLE = "strict_leave_one_condition_out_fold_model"

HAND_OPEN = 0
LATERAL_GRIP = 1
PINCH_GRIP = 2
POWER_GRIP = 3
REST = 4

LABEL_NAMES = {
    HAND_OPEN: "HandOpen",
    LATERAL_GRIP: "Lateral",
    PINCH_GRIP: "Pinch",
    POWER_GRIP: "Power",
    REST: "Rest",
}

# Fixed-plane command convention requested by the experiment.
LABEL_TO_XY = {
    HAND_OPEN: (1.0, 0.0),
    LATERAL_GRIP: (-1.0, 0.0),
    PINCH_GRIP: (0.0, 1.0),
    POWER_GRIP: (0.0, -1.0),
    REST: (0.0, 0.0),
}
