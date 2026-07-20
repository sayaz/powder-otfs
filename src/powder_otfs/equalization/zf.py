import numpy as np

from powder_otfs.equalization.equalized import EqualizedGrid
from powder_otfs.estimation.estimate import ChannelEstimate


def zero_forcing_equalizer(
    received_grid: np.ndarray,
    estimate: ChannelEstimate,
) -> EqualizedGrid:
    """Apply Zero-Forcing equalization using the channel matrix."""

    equalized_grids = zero_forcing_equalize_frames(
        received_grids=received_grid[
            np.newaxis,
            ...,
        ],
        estimate=estimate,
    )

    return EqualizedGrid(
        symbols=equalized_grids[0],
        method="Zero Forcing",
    )


def zero_forcing_equalize_frames(
    received_grids: np.ndarray,
    estimate: ChannelEstimate,
) -> np.ndarray:
    """ZF-equalize multiple grids using one channel estimate."""

    if received_grids.ndim != 3:
        raise ValueError(
            "received_grids must have shape "
            "(frames, delay_bins, doppler_bins)."
        )

    grid_shape = received_grids.shape[1:]
    received_matrix = (
        received_grids.reshape(
            received_grids.shape[0],
            -1,
        ).T
    )

    equalized_matrix = np.linalg.lstsq(
        estimate.channel_response,
        received_matrix,
        rcond=None,
    )[0]

    return equalized_matrix.T.reshape(
        received_grids.shape[0],
        *grid_shape,
    )
