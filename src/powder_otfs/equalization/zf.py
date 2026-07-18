import numpy as np

from powder_otfs.equalization.equalized import EqualizedGrid
from powder_otfs.estimation.estimate import ChannelEstimate


def zero_forcing_equalizer(
    received_grid: np.ndarray,
    estimate: ChannelEstimate,
) -> EqualizedGrid:
    """
    Apply Zero-Forcing equalization.
    """

    equalized = received_grid / estimate.channel_response

    return EqualizedGrid(
        symbols=equalized,
        method="Zero Forcing",
    )