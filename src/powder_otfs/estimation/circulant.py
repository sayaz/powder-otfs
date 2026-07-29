import numpy as np

from powder_otfs.channel.path import ChannelPath


def circulant_from_first_column(
    column: np.ndarray,
) -> np.ndarray:
    """Construct a circulant matrix from its first column."""

    if column.ndim != 1:
        raise ValueError("column must be one-dimensional.")

    size = len(column)

    row_indices = np.arange(size)[:, None]
    column_indices = np.arange(size)[None, :]

    circulant_indices = (
        row_indices - column_indices
    ) % size

    return column[circulant_indices]


def block_circulant_from_impulse_response(
    impulse_response: np.ndarray,
) -> np.ndarray:
    """Construct the full DD channel matrix."""

    if impulse_response.ndim != 2:
        raise ValueError(
            "impulse_response must be two-dimensional."
        )

    num_delay_bins, num_doppler_bins = (
        impulse_response.shape
    )
    num_symbols = (
        num_delay_bins * num_doppler_bins
    )

    circulant_blocks = [
        circulant_from_first_column(
            impulse_response[delay_bin],
        )
        for delay_bin in range(num_delay_bins)
    ]

    channel_matrix = np.zeros(
        (num_symbols, num_symbols),
        dtype=np.complex128,
    )

    for output_delay in range(num_delay_bins):
        output_start = (
            output_delay * num_doppler_bins
        )
        output_stop = (
            output_start + num_doppler_bins
        )

        for input_delay in range(num_delay_bins):
            input_start = (
                input_delay * num_doppler_bins
            )
            input_stop = (
                input_start + num_doppler_bins
            )

            delay_offset = (
                output_delay - input_delay
            ) % num_delay_bins

            channel_matrix[
                output_start:output_stop,
                input_start:input_stop,
            ] = circulant_blocks[delay_offset]

    return channel_matrix


def rcp_channel_matrix_from_doppler_response(
    doppler_response: np.ndarray,
) -> np.ndarray:
    """Construct the RCP-OTFS DD channel matrix using Eqs. (7.10-7.11)."""

    if doppler_response.ndim != 2:
        raise ValueError(
            "doppler_response must be two-dimensional."
        )

    num_delay_bins, num_doppler_bins = (
        doppler_response.shape
    )
    num_symbols = num_delay_bins * num_doppler_bins

    doppler_indices = np.arange(num_doppler_bins)
    signed_doppler_indices = np.where(
        doppler_indices > num_doppler_bins // 2,
        doppler_indices - num_doppler_bins,
        doppler_indices,
    )

    phase_rotation = np.diag(
        np.exp(
            -1j
            * 2.0
            * np.pi
            * doppler_indices
            / num_doppler_bins
        )
    )

    channel_matrix = np.zeros(
        (num_symbols, num_symbols),
        dtype=np.complex128,
    )

    for output_delay in range(num_delay_bins):
        output_start = output_delay * num_doppler_bins
        output_stop = output_start + num_doppler_bins

        for delay_tap in range(num_delay_bins):
            if not np.any(doppler_response[delay_tap]):
                continue

            input_delay = (
                output_delay - delay_tap
            ) % num_delay_bins
            input_start = input_delay * num_doppler_bins
            input_stop = input_start + num_doppler_bins

            time_varying_response = (
                doppler_response[delay_tap]
                * np.exp(
                    1j
                    * 2.0
                    * np.pi
                    * signed_doppler_indices
                    * (output_delay - delay_tap)
                    / num_symbols
                )
            )

            channel_block = circulant_from_first_column(
                time_varying_response
            )

            if output_delay < delay_tap:
                channel_block = (
                    channel_block @ phase_rotation
                )

            channel_matrix[
                output_start:output_stop,
                input_start:input_stop,
            ] = channel_block

    return channel_matrix


def rectangular_pulse_channel_matrix(
    paths: list[ChannelPath],
    grid_shape: tuple[int, int],
    sample_rate: float,
) -> np.ndarray:
    """Construct the phase-aware DD channel matrix for rectangular pulses."""

    num_delay_bins, num_doppler_bins = grid_shape
    num_symbols = num_delay_bins * num_doppler_bins

    channel_matrix = np.zeros(
        (num_symbols, num_symbols),
        dtype=np.complex128,
    )

    for path in paths:
        delay_index = path.delay_samples % num_delay_bins
        doppler_index = int(
            np.rint(
                path.doppler_hz
                * num_symbols
                / sample_rate
            )
        )

        for input_delay in range(num_delay_bins):
            output_delay = (
                input_delay + delay_index
            ) % num_delay_bins
            wrapped_delay = (
                input_delay + delay_index
                >= num_delay_bins
            )

            for input_doppler in range(num_doppler_bins):
                output_doppler = (
                    input_doppler + doppler_index
                ) % num_doppler_bins

                phase = np.exp(
                    1j
                    * 2.0
                    * np.pi
                    * doppler_index
                    * output_delay
                    / num_symbols
                )

                if wrapped_delay:
                    phase *= np.exp(
                        -1j
                        * 2.0
                        * np.pi
                        * input_doppler
                        / num_doppler_bins
                    )

                output_index = (
                    output_delay * num_doppler_bins
                    + output_doppler
                )
                input_index = (
                    input_delay * num_doppler_bins
                    + input_doppler
                )

                channel_matrix[
                    output_index,
                    input_index,
                ] += path.gain * phase

    return channel_matrix
