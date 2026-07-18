import numpy as np

def apply_doppler(
    waveform: np.ndarray,
    doppler_hz: float,
    sample_rate: float,
) -> np.ndarray:
    """Apply a Doppler shift."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    sample_indices = np.arange(len(waveform))
    sample_times = sample_indices / sample_rate

    phase_rotation = np.exp(
        1j * 2 * np.pi * doppler_hz * sample_times
    )

    return waveform * phase_rotation
