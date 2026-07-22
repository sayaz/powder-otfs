import numpy as np


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