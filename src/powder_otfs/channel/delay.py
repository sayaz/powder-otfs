import numpy as np


def apply_delay(
    waveform: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    """Apply a fixed-length, zero-padded propagation delay."""

    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative.")

    if delay_samples >= len(waveform):
        return np.zeros_like(waveform)

    if delay_samples == 0:
        return waveform.copy()

    return np.concatenate(
        (
            np.zeros(delay_samples, dtype=waveform.dtype),
            waveform[:-delay_samples],
        )
    )


def apply_circular_delay(
    waveform: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    """Apply a circular delay to an OTFS frame."""

    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative.")

    return np.roll(
        waveform,
        shift=delay_samples,
    )