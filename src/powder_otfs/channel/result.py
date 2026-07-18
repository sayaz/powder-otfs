from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ChannelResult:
    """Stores the output of the channel model."""

    waveform: np.ndarray
    channel_response: np.ndarray