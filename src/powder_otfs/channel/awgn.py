import numpy as np

def add_awgn(
        waveform: np.ndarray,
        snr_db: float,
) -> np.ndarray:
    """Add complex white Gaussian noise at requested SNR"""

    signal_power = np.mean(np.abs(waveform) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*waveform.shape)
        + 1j * np.random.randn(*waveform.shape)
    )

    return waveform + noise