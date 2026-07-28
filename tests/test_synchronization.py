import numpy as np

from powder_otfs.ota.framing import create_training_preamble
from powder_otfs.ota.synchronization import (
    correct_fractional_timing,
    estimate_fractional_timing_offset,
    find_payload_start,
    find_payload_starts,
    find_training_preamble_starts,
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


def test_find_training_preamble_starts() -> None:
    training = create_training_preamble()
    received = np.zeros(1500, dtype=np.complex64)
    expected_starts = np.array([100, 900])

    for start in expected_starts:
        received[
            start:start + len(training.samples)
        ] = training.samples

    detected = find_training_preamble_starts(
        received=received,
        preamble=training,
        stf_threshold=0.75,
        ltf_threshold=0.90,
        minimum_separation=500,
    )

    np.testing.assert_array_equal(detected, expected_starts)


def test_fractional_timing_estimation_and_correction() -> None:
    training = create_training_preamble()
    aligned = np.concatenate(
        (
            training.samples,
            np.zeros(128, dtype=np.complex64),
        )
    )
    received = correct_fractional_timing(
        aligned,
        offset_samples=-0.30,
    )

    offset = estimate_fractional_timing_offset(
        received=received,
        known_sequence=training.ltf_symbol,
        integer_start=training.ltf_symbol_offset,
    )
    corrected = correct_fractional_timing(
        received,
        offset_samples=offset,
    )

    assert np.isclose(offset, 0.30, atol=0.08)
    before_error = np.mean(
        np.abs(received[40:-40] - aligned[40:-40]) ** 2
    )
    after_error = np.mean(
        np.abs(corrected[40:-40] - aligned[40:-40]) ** 2
    )
    assert after_error < before_error
