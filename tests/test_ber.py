import numpy as np

from powder_otfs.metrics.ber import bit_error_rate


def test_zero_ber():
    tx = np.array([0, 1, 1, 0], dtype=np.uint8)
    rx = np.array([0, 1, 1, 0], dtype=np.uint8)

    assert bit_error_rate(tx, rx) == 0.0


def test_half_ber():
    tx = np.array([0, 1, 1, 0], dtype=np.uint8)
    rx = np.array([1, 1, 0, 0], dtype=np.uint8)

    assert bit_error_rate(tx, rx) == 0.5