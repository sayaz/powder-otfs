import numpy as np

from powder_otfs.ota.synchronization import correct_fractional_timing


def add_delayed_path(
    waveform: np.ndarray,
    delay_samples: int,
    gain: complex,
) -> np.ndarray:
    """Add one delayed and scaled copy of a waveform."""

    if delay_samples < 0:
        raise ValueError("delay_samples must be nonnegative.")

    impaired = np.asarray(
        waveform,
        dtype=np.complex128,
    ).copy()

    if gain == 0 or delay_samples >= len(impaired):
        return impaired

    if delay_samples == 0:
        impaired += gain * waveform
    else:
        impaired[delay_samples:] += (
            gain * waveform[:-delay_samples]
        )

    return impaired


def apply_frequency_offset(
    waveform: np.ndarray,
    cfo_hz: float,
    sample_rate: float,
) -> np.ndarray:
    """Apply a controlled carrier-frequency offset."""

    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive.")

    sample_indices = np.arange(
        len(waveform),
        dtype=np.float64,
    )
    rotation = np.exp(
        1j
        * 2.0
        * np.pi
        * cfo_hz
        * sample_indices
        / sample_rate
    )
    return waveform * rotation


def apply_fractional_delay(
    waveform: np.ndarray,
    delay_samples: float,
) -> np.ndarray:
    """Delay a waveform by a fractional sample using sinc interpolation."""

    if abs(delay_samples) > 0.5:
        raise ValueError(
            "delay_samples must be between -0.5 and 0.5."
        )

    return correct_fractional_timing(
        samples=waveform,
        offset_samples=-delay_samples,
    )
