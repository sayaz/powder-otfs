import numpy as np

from powder_otfs.channel.doppler import apply_doppler


def test_zero_doppler_returns_original_waveform():
    waveform = np.array(
        [1 + 0j, 2 + 0j, 3 + 0j],
        dtype=np.complex128,
    )

    shifted = apply_doppler(
        waveform,
        doppler_hz=0.0,
        sample_rate=1000.0,
    )

    np.testing.assert_allclose(
        shifted,
        waveform,
    )


def test_apply_known_doppler_shift():
    waveform = np.ones(
        4,
        dtype=np.complex128,
    )

    shifted = apply_doppler(
        waveform,
        doppler_hz=250.0,
        sample_rate=1000.0,
    )

    expected = np.array(
        [
            1 + 0j,
            0 + 1j,
            -1 + 0j,
            0 - 1j,
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        shifted,
        expected,
        atol=1e-12,
    )


def test_invalid_sample_rate():
    waveform = np.ones(
        4,
        dtype=np.complex128,
    )

    try:
        apply_doppler(
            waveform,
            doppler_hz=100.0,
            sample_rate=0.0,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError")