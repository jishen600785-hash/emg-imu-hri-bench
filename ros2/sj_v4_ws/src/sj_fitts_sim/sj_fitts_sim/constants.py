from __future__ import annotations


CLASS_NAMES = ["hand_close", "hand_down", "hand_open", "hand_up", "rest"]
INPUT_FEATURES = 350
EMG_CHANNELS = 5
EMG_POINTS = 256
IMU_CHANNELS = 30
IMU_POINTS = 64
WINDOW_SECONDS = 0.50
STEP_SECONDS = 0.25
HELD_OUT_CONDITION = "dynamic"

HAND_CLOSE = 0
HAND_DOWN = 1
HAND_OPEN = 2
HAND_UP = 3
REST = 4

LABEL_NAMES = {
    HAND_CLOSE: "Hand Close",
    HAND_DOWN: "Hand Down",
    HAND_OPEN: "Hand Open",
    HAND_UP: "Hand Up",
    REST: "Rest",
}

# Compatibility aliases used by the generic Fitts task implementation.
# Their numeric values deliberately point to the SJ gestures that command the
# corresponding planar direction.
LATERAL_GRIP = HAND_CLOSE   # -X
PINCH_GRIP = HAND_UP        # +Y
POWER_GRIP = HAND_DOWN      # -Y

LABEL_TO_XY = {
    HAND_CLOSE: (-1.0, 0.0),
    HAND_DOWN: (0.0, -1.0),
    HAND_OPEN: (1.0, 0.0),
    HAND_UP: (0.0, 1.0),
    REST: (0.0, 0.0),
}
