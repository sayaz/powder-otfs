import numpy as np

from powder_otfs.channel.awgn import add_awgn
from powder_otfs.channel.delay import apply_circular_delay
from powder_otfs.channel.doppler import apply_doppler
from powder_otfs.channel.path import ChannelPath
from powder_otfs.channel.result import ChannelResult


def apply_channel(
    waveform: np.ndarray,
    paths: list[ChannelPath],
    sample_rate: float,
    snr_db: float,
) -> ChannelResult:
    """Apply circular multipath propagation and AWGN."""

    if not paths:
        raise ValueError("At least one channel path is required.")

    received_without_noise = np.zeros_like(
        waveform,
        dtype=np.complex128,
    )

    for path in paths:
        path_waveform = apply_circular_delay(
            waveform,
            path.delay_samples,
        )

        path_waveform = apply_doppler(
            path_waveform,
            path.doppler_hz,
            sample_rate,
        )

        received_without_noise += path.gain * path_waveform

    signal_power = float(
        np.mean(np.abs(received_without_noise) ** 2)
    )
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance = signal_power / snr_linear

    received = add_awgn(
        received_without_noise,
        snr_db,
    )

    return ChannelResult(
        waveform=received,
        paths=tuple(paths),
        sample_rate=sample_rate,
        noise_variance=noise_variance,
    )