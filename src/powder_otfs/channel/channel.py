import numpy as np

from powder_otfs.channel.awgn import add_awgn
from powder_otfs.channel.delay import apply_delay
from powder_otfs.channel.doppler import apply_doppler
from powder_otfs.channel.path import ChannelPath
from powder_otfs.channel.result import ChannelResult


def apply_channel(
    waveform: np.ndarray,
    paths: list[ChannelPath],
    sample_rate: float,
    snr_db: float,
) -> ChannelResult:
    """
    Apply a multipath wireless channel and AWGN.
    """

    received = np.zeros_like(waveform, dtype=complex)

    #
    # Apply each propagation path
    #
    for path in paths:

        signal = apply_delay(
            waveform,
            path.delay_samples,
        )

        signal = apply_doppler(
            signal,
            path.doppler_hz,
            sample_rate,
        )

        signal *= path.gain

        received += signal

    #
    # Add AWGN
    #
    received = add_awgn(
        received,
        snr_db,
    )

    #
    # Temporary perfect channel response.
    # This will be replaced with the true channel response later.
    #
    channel_response = np.ones_like(received, dtype=complex)

    return ChannelResult(
        waveform=received,
        channel_response=channel_response,
    )