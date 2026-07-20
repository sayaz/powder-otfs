import numpy as np

from powder_otfs.ota.framing import (
    build_ota_frame,
    create_preamble,
    normalize_waveform,
)

from powder_otfs.ota.framing import (
    build_ota_frame,
    create_preamble,
)


def test_create_preamble_has_identical_halves() -> None:
    preamble = create_preamble(
        half_length=16,
        seed=1,
    )

    assert len(preamble) == 32

    np.testing.assert_array_equal(
        preamble[:16],
        preamble[16:],
    )


def test_build_ota_frame() -> None:
    payload = np.array(
        [1 + 1j, 2 + 2j],
        dtype=np.complex64,
    )
    preamble = np.array(
        [1 + 0j, -1 + 0j],
        dtype=np.complex64,
    )

    frame = build_ota_frame(
        payload=payload,
        preamble=preamble,
        guard_samples=2,
    )

    expected = np.array(
        [
            0 + 0j,
            0 + 0j,
            1 + 0j,
            -1 + 0j,
            1 + 1j,
            2 + 2j,
            0 + 0j,
            0 + 0j,
        ],
        dtype=np.complex64,
    )

    np.testing.assert_array_equal(frame, expected)


def test_build_ota_frame_with_cyclic_prefix() -> None:
    payload = np.array(
        [1, 2, 3, 4],
        dtype=np.complex64,
    )
    preamble = np.array(
        [5, 6],
        dtype=np.complex64,
    )

    frame = build_ota_frame(
        payload=payload,
        preamble=preamble,
        guard_samples=1,
        cyclic_prefix_samples=2,
    )

    expected = np.array(
        [
            0,
            5,
            6,
            3,
            4,
            1,
            2,
            3,
            4,
            0,
        ],
        dtype=np.complex64,
    )

    np.testing.assert_array_equal(
        frame,
        expected,
    )


def test_normalize_waveform() -> None:
    waveform = np.array(
        [1 + 1j, 2 + 2j],
        dtype=np.complex64,
    )

    normalized = normalize_waveform(
        waveform,
        peak_amplitude=0.8,
    )

    assert np.isclose(
        np.max(np.abs(normalized)),
        0.8,
    )
