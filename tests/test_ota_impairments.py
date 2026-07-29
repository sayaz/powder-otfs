import numpy as np

from powder_otfs.ota.impairments import (
    add_delayed_path,
    apply_fractional_delay,
    apply_frequency_offset,
)


def test_add_delayed_path() -> None:
    waveform = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.complex128,
    )

    impaired = add_delayed_path(
        waveform=waveform,
        delay_samples=1,
        gain=0.5 + 0.0j,
    )

    np.testing.assert_allclose(
        impaired,
        np.array([1.0, 2.5, 4.0, 5.5]),
    )


def test_zero_gain_leaves_waveform_unchanged() -> None:
    waveform = np.array(
        [1.0 + 1.0j, 2.0 - 1.0j],
        dtype=np.complex128,
    )

    impaired = add_delayed_path(
        waveform=waveform,
        delay_samples=1,
        gain=0.0 + 0.0j,
    )

    np.testing.assert_array_equal(
        impaired,
        waveform,
    )


def test_apply_frequency_offset() -> None:
    waveform = np.ones(4, dtype=np.complex128)

    impaired = apply_frequency_offset(
        waveform=waveform,
        cfo_hz=2.0,
        sample_rate=8.0,
    )

    np.testing.assert_allclose(
        impaired,
        np.array([1.0, 1.0j, -1.0, -1.0j]),
        atol=1e-12,
    )


def test_apply_fractional_delay_to_tone() -> None:
    sample_indices = np.arange(512)
    tone_frequency = 0.05
    delay_samples = 0.25
    waveform = np.exp(
        1j * 2.0 * np.pi * tone_frequency * sample_indices
    )

    impaired = apply_fractional_delay(
        waveform=waveform,
        delay_samples=delay_samples,
    )
    expected = waveform * np.exp(
        -1j * 2.0 * np.pi * tone_frequency * delay_samples
    )

    np.testing.assert_allclose(
        impaired[24:-24],
        expected[24:-24],
        atol=2e-3,
    )
