import numpy as np

from powder_otfs.estimation.estimate import ChannelEstimate


def pilot_channel_estimate(
    received_pilot_grid: np.ndarray,
    pilot_position: tuple[int, int],
    pilot_value: complex,
    noise_variance: float,
    threshold: float = 0.0,
) -> ChannelEstimate:
    """Estimate a circular delay-Doppler channel from one pilot."""

    if pilot_value == 0:
        raise ValueError("pilot_value must be nonzero.")

    pilot_delay, pilot_doppler = pilot_position

    impulse_response = received_pilot_grid / pilot_value
    impulse_response = np.roll(
        impulse_response,
        shift=(-pilot_delay, -pilot_doppler),
        axis=(0, 1),
    )

    if threshold > 0:
        impulse_response[
            np.abs(impulse_response) < threshold
        ] = 0.0

    grid_shape = received_pilot_grid.shape
    num_symbols = received_pilot_grid.size

    channel_matrix = np.zeros(
        (num_symbols, num_symbols),
        dtype=np.complex128,
    )

    for column in range(num_symbols):
        delay_bin, doppler_bin = np.unravel_index(
            column,
            grid_shape,
        )

        shifted_response = np.roll(
            impulse_response,
            shift=(delay_bin, doppler_bin),
            axis=(0, 1),
        )

        channel_matrix[:, column] = shifted_response.reshape(-1)

    return ChannelEstimate(
        channel_response=channel_matrix,
        noise_variance=noise_variance,
        method="Embedded Pilot",
    )