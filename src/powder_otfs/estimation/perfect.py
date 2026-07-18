import numpy as np

from powder_otfs.estimation.estimate import ChannelEstimate


def perfect_channel_estimate(
    channel_response: np.ndarray,
    noise_variance: float,
) -> ChannelEstimate:
    """
    Creates a perfect channel estimate.
    """

    return ChannelEstimate(
        channel_response=channel_response,
        noise_variance=noise_variance,
        method="Perfect CSI",
    )