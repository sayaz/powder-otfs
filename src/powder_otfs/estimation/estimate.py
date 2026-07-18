from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ChannelEstimate:
    """
    Stores the estimated channel information.
    """

    channel_response: np.ndarray
    noise_variance: float
    method: str