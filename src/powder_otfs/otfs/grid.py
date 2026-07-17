import numpy as np

def create_delay_doppler_grid(
        symbols: np.ndarray,
        num_delay_bins: int,
        num_doppler_bins: int,
) -> np.ndarray:
    """Map symbols into an OTFS delay-Doppler grid."""

    expected = num_delay_bins * num_doppler_bins

    if len(symbols) != expected:
        raise ValueError(
            f"Expected {expected} symbols, got {len(symbols)}."
        )
    
    return symbols.reshape(num_delay_bins, num_doppler_bins)