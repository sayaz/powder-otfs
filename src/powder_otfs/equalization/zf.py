import numpy as np

from powder_otfs.equalization.equalized import EqualizedGrid
from powder_otfs.estimation.estimate import ChannelEstimate


def zero_forcing_equalizer(
    received_grid: np.ndarray,
    estimate: ChannelEstimate,
) -> EqualizedGrid:
    """Apply Zero-Forcing equalization using the channel matrix."""

    received_vector = received_grid.reshape(-1)

    equalized_vector = np.linalg.lstsq(
        estimate.channel_response,
        received_vector,
        rcond=None,
    )[0]

    equalized_grid = equalized_vector.reshape(
        received_grid.shape
    )

    return EqualizedGrid(
        symbols=equalized_grid,
        method="Zero Forcing",
    )