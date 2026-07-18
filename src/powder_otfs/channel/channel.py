import numpy as np

from powder_otfs.channel.awgn import add_awgn
from powder_otfs.channel.delay import apply_delay
from powder_otfs.channel.doppler import apply_doppler
from powder_otfs.channel.path import ChannelPath


def apply_channel(
    waveform: np.ndarray,
    paths: list[ChannelPath],
    sample_rate: float,
    snr_db: float,
) -> np.ndarray:
    """Apply a multipath wireless channel."""

    received = np.zeros_like(waveform)

    for path in paths:
        path_waveform = apply_delay(
            waveform,
            path.delay_samples,
        )

        path_waveform = apply_doppler(
            path_waveform,
            path.doppler_hz,
            sample_rate,
        )

        path_waveform *= path.gain

        received += path_waveform

    received = add_awgn(
        received,
        snr_db,
    )

    return received