from dataclasses import dataclass

import numpy as np

from powder_otfs.channel.path import ChannelPath


@dataclass(slots=True)
class ChannelResult:
    """Stores the waveform and physical channel state."""

    waveform: np.ndarray
    paths: tuple[ChannelPath, ...]
    sample_rate: float
    noise_variance: float