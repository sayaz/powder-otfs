import numpy as np

from powder_otfs.channel.delay import apply_circular_delay
from powder_otfs.channel.doppler import apply_doppler
from powder_otfs.channel.result import ChannelResult
from powder_otfs.estimation.estimate import ChannelEstimate
from powder_otfs.otfs.transforms import heisenberg, isfft, sfft, wigner


def perfect_channel_estimate(
    channel: ChannelResult,
    grid_shape: tuple[int, int],
) -> ChannelEstimate:
    """Construct the exact delay-Doppler channel matrix."""

    num_delay_bins, num_doppler_bins = grid_shape
    num_symbols = num_delay_bins * num_doppler_bins

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

        basis_waveform = heisenberg(
            isfft(basis_grid)
        )

        received_waveform = np.zeros_like(
            basis_waveform,
            dtype=np.complex128,
        )

        for path in channel.paths:
            path_waveform = apply_circular_delay(
                basis_waveform,
                path.delay_samples,
            )
            path_waveform = apply_doppler(
                path_waveform,
                path.doppler_hz,
                channel.sample_rate,
            )
            received_waveform += path.gain * path_waveform

        received_tf_grid = wigner(
            received_waveform,
            num_subcarriers=num_delay_bins,
            num_time_slots=num_doppler_bins,
        )
        received_dd_grid = sfft(received_tf_grid)

        channel_matrix[:, column] = received_dd_grid.reshape(-1)

    return ChannelEstimate(
        channel_response=channel_matrix,
        noise_variance=channel.noise_variance,
        method="Perfect CSI",
    )