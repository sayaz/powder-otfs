import numpy as np

from powder_otfs.channel.path import ChannelPath
from powder_otfs.channel.result import ChannelResult
from powder_otfs.estimation.perfect import perfect_channel_estimate


def test_perfect_channel_estimate() -> None:
    channel = ChannelResult(
        waveform=np.zeros(4, dtype=np.complex128),
        paths=(
            ChannelPath(
                delay_samples=0,
                doppler_hz=0.0,
                gain=1.0 + 0.0j,
            ),
        ),
        sample_rate=1000.0,
        noise_variance=0.01,
    )

    estimate = perfect_channel_estimate(
        channel=channel,
        grid_shape=(2, 2),
    )

    np.testing.assert_allclose(
        estimate.channel_response,
        np.eye(4, dtype=np.complex128),
        atol=1e-12,
    )

    assert estimate.noise_variance == 0.01
    assert estimate.method == "Perfect CSI"