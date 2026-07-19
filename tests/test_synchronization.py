import numpy as np

from powder_otfs.ota.synchronization import (
    find_payload_start,
    find_payload_starts,
)


def test_find_payload_start() -> None:
    preamble = np.array(
        [1, -1, 1j, -1j],
        dtype=np.complex64,
    )

    received = np.concatenate(
        (
            np.zeros(7, dtype=np.complex64),
            preamble,
            np.zeros(10, dtype=np.complex64),
        )
    )

    payload_start = find_payload_start(
        received=received,
        preamble=preamble,
    )

    assert payload_start == 11


def test_find_multiple_payload_starts() -> None:
    preamble = np.array(
        [1, -1, 1j, -1j],
        dtype=np.complex64,
    )

    received = np.zeros(
        60,
        dtype=np.complex64,
    )

    received[8:12] = preamble
    received[36:40] = preamble

    payload_starts = find_payload_starts(
        received=received,
        preamble=preamble,
        threshold=0.99,
        minimum_separation=14,
    )

    np.testing.assert_array_equal(
        payload_starts,
        np.array([12, 40]),
    )