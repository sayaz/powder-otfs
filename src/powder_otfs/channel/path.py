from dataclasses import dataclass

@dataclass(slots = True)
class ChannelPath:
    """One propagation path."""

    delay_samples: int
    doppler_hz: float
    gain: complex