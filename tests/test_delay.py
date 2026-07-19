import numpy as np

from powder_otfs.channel.delay import (
    apply_circular_delay,
    apply_delay,
)


def test_apply_delay() -> None:
    waveform = np.array(
        [1, 2, 3, 4],
        dtype=np.complex128,
    )

    delayed = apply_delay(
        waveform,
        delay_samples=2,
    )

    expected = np.array(
        [0, 0, 1, 2],
        dtype=np.complex128,
    )

    np.testing.assert_array_equal(
        delayed,
        expected,
    )


def test_apply_circular_delay() -> None:
    waveform = np.array(
        [1, 2, 3, 4],
        dtype=np.complex128,
    )

    delayed = apply_circular_delay(
        waveform,
        delay_samples=2,
    )

    expected = np.array(
        [3, 4, 1, 2],
        dtype=np.complex128,
    )

    np.testing.assert_array_equal(
        delayed,
        expected,
    )