import numpy as np
import pytest
from std_msgs.msg import Float32MultiArray

from sj_fitts_sim.classifier_node import (
    FEATURE_VALUE_COUNT,
    GestureClassifierNode,
    MESSAGE_VALUE_COUNT,
)
from sj_fitts_sim.constants import (
    EMG_CHANNELS,
    EMG_POINTS,
    IMU_CHANNELS,
    IMU_POINTS,
    INPUT_FEATURES,
)


def test_window_message_keeps_truth_and_stream_with_features():
    message = Float32MultiArray()
    features = np.arange(FEATURE_VALUE_COUNT, dtype=np.float32)
    message.data = [2.0, 17.0, *features.tolist()]

    truth, stream, basic, emg, imu = GestureClassifierNode._decode(message)

    assert len(message.data) == MESSAGE_VALUE_COUNT
    assert truth == 2
    assert stream == 17
    assert basic.shape == (INPUT_FEATURES,)
    assert emg.shape == (EMG_CHANNELS, EMG_POINTS)
    assert imu.shape == (IMU_CHANNELS, IMU_POINTS)
    assert basic[0] == 0.0
    assert imu[-1, -1] == FEATURE_VALUE_COUNT - 1


def test_window_message_rejects_invalid_embedded_label():
    message = Float32MultiArray()
    message.data = [99.0, 0.0] + [0.0] * FEATURE_VALUE_COUNT

    with pytest.raises(ValueError, match="ground-truth"):
        GestureClassifierNode._decode(message)
