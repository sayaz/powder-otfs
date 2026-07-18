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


def insert_pilot_and_guards(
    grid: np.ndarray,
    pilot_delay_bin: int,
    pilot_doppler_bin: int,
    pilot_amplitude: float,
    guard_delay: int,
    guard_doppler: int,
) -> np.ndarray:
    """Insert one pilot and zero-valued guard bins around it."""

    output_grid = grid.copy()

    delay_start = pilot_delay_bin - guard_delay
    delay_stop = pilot_delay_bin + guard_delay + 1

    doppler_start = pilot_doppler_bin - guard_doppler
    doppler_stop = pilot_doppler_bin + guard_doppler + 1

    output_grid[
        delay_start:delay_stop,
        doppler_start:doppler_stop,
    ] = 0.0 + 0.0j

    output_grid[
        pilot_delay_bin,
        pilot_doppler_bin,
    ] = pilot_amplitude + 0.0j

    return output_grid