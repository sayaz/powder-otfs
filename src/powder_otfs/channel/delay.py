import numpy as np

def apply_delay(
    waveform: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    """Apply an integer propagation delay."""

    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative")

    delayed = np.concatenate(
        (
            np.zeros(delay_samples, dtype=waveform.dtype),
            waveform[:-delay_samples] if delay_samples > 0 else waveform,
        )
    )

    return delayed