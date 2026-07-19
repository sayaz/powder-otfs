import numpy as np

from powder_otfs.channel.delay import apply_circular_delay
from powder_otfs.channel.doppler import apply_doppler
from powder_otfs.channel.path import ChannelPath
from powder_otfs.estimation.estimate import ChannelEstimate
from powder_otfs.otfs.transforms import heisenberg, isfft, sfft, wigner


def _apply_paths(
    dd_grid: np.ndarray,
    paths: list[ChannelPath],
    sample_rate: float,
) -> np.ndarray:
    waveform = heisenberg(
        isfft(dd_grid)
    )

    received = np.zeros_like(
        waveform,
        dtype=np.complex128,
    )

    for path in paths:
        path_waveform = apply_circular_delay(
            waveform,
            path.delay_samples,
        )
        path_waveform = apply_doppler(
            path_waveform,
            path.doppler_hz,
            sample_rate,
        )
        received += path.gain * path_waveform

    tf_grid = wigner(
        received,
        num_subcarriers=dd_grid.shape[0],
        num_time_slots=dd_grid.shape[1],
    )

    return sfft(tf_grid)


def pilot_channel_estimate(
    received_pilot_grid: np.ndarray,
    pilot_position: tuple[int, int],
    pilot_value: complex,
    sample_rate: float,
    noise_variance: float,
    threshold: float,
) -> ChannelEstimate:
    """Estimate a grid-aligned delay-Doppler channel from one pilot."""

    if pilot_value == 0:
        raise ValueError("pilot_value must be nonzero.")

    grid_shape = received_pilot_grid.shape
    num_delay_bins, num_doppler_bins = grid_shape
    num_symbols = received_pilot_grid.size

    pilot_delay, pilot_doppler = pilot_position

    detected_bins = np.argwhere(
        np.abs(received_pilot_grid) >= threshold
    )

    estimated_paths: list[ChannelPath] = []

    for received_delay, received_doppler in detected_bins:
        delay_samples = (
            pilot_delay - int(received_delay)
        ) % num_delay_bins

        doppler_index = (
            pilot_doppler - int(received_doppler)
        ) % num_doppler_bins

        if doppler_index > num_doppler_bins // 2:
            doppler_index -= num_doppler_bins

        doppler_hz = (
            doppler_index
            * sample_rate
            / num_symbols
        )

        reference_pilot_grid = np.zeros(
            grid_shape,
            dtype=np.complex128,
        )
        reference_pilot_grid[pilot_position] = pilot_value

        reference_response = _apply_paths(
            reference_pilot_grid,
            [
                ChannelPath(
                    delay_samples=delay_samples,
                    doppler_hz=doppler_hz,
                    gain=1.0 + 0.0j,
                )
            ],
            sample_rate,
        )

        reference_value = reference_response[
            received_delay,
            received_doppler,
        ]

        if np.abs(reference_value) == 0:
            continue

        estimated_gain = (
            received_pilot_grid[
                received_delay,
                received_doppler,
            ]
            / reference_value
        )

        estimated_paths.append(
            ChannelPath(
                delay_samples=delay_samples,
                doppler_hz=doppler_hz,
                gain=estimated_gain,
            )
        )

    if not estimated_paths:
        raise ValueError("No channel paths detected from the pilot.")

    channel_matrix = np.zeros(
        (num_symbols, num_symbols),
        dtype=np.complex128,
    )

    for column in range(num_symbols):
        basis_grid = np.zeros(
            grid_shape,
            dtype=np.complex128,
        )
        basis_grid.reshape(-1)[column] = 1.0

        response_grid = _apply_paths(
            basis_grid,
            estimated_paths,
            sample_rate,
        )

        channel_matrix[:, column] = response_grid.reshape(-1)

    return ChannelEstimate(
        channel_response=channel_matrix,
        noise_variance=noise_variance,
        method="Embedded Pilot",
    )