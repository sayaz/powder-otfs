import numpy as np

from powder_otfs.ota.framing import (
    build_ota_frame,
    create_training_preamble,
    normalize_waveform,
)

def test_create_training_preamble_structure() -> None:
    preamble = create_training_preamble()

    assert len(preamble.stf_short_symbol) == 16
    assert len(preamble.stf) == 160
    assert len(preamble.ltf_symbol) == 64
    assert len(preamble.ltf) == 160
    assert len(preamble.samples) == 320

    np.testing.assert_array_equal(
        preamble.stf[:16],
        preamble.stf[16:32],
    )
    np.testing.assert_array_equal(
        preamble.ltf[32:96],
        preamble.ltf[96:160],
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
