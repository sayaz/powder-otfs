import numpy as np

from powder_otfs.ota.frequency_offset import (
    correct_cfo,
    estimate_cfo,
)


def test_estimate_and_correct_cfo() -> None:
    sample_rate = 1_000_000.0
    expected_cfo_hz = 2_500.0

    rng = np.random.default_rng(12345)
    preamble_half = (
        2 * rng.integers(0, 2, 64) - 1
    ).astype(np.complex128)
    preamble = np.concatenate(
        (
            preamble_half,
            preamble_half,
        )
    )

    sample_indices = np.arange(
        len(preamble),
        dtype=np.float64,
    )
    received = preamble * np.exp(
        1j
        * 2.0
        * np.pi
        * expected_cfo_hz
        * sample_indices
        / sample_rate
    )

    estimated_cfo_hz = estimate_cfo(
        repeated_preamble=received,
        sample_rate=sample_rate,
    )
    corrected = correct_cfo(
        samples=received,
        cfo_hz=estimated_cfo_hz,
        sample_rate=sample_rate,
    )

    assert np.isclose(
        estimated_cfo_hz,
        expected_cfo_hz,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        corrected,
        preamble,
        atol=1e-12,
    )
