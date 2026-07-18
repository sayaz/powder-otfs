import numpy as np

from powder_otfs.channel.channel import apply_channel
from powder_otfs.channel.path import ChannelPath


def test_single_path_channel():
    waveform = np.array(
        [1, 2, 3, 4],
        dtype=np.complex128,
    )

    paths = [
        ChannelPath(
            delay_samples=0,
            doppler_hz=0.0,
            gain=1.0 + 0j,
        )
    ]

    channel = apply_channel(
        waveform,
        paths,
        sample_rate=1000.0,
        snr_db=1000.0,
    )

    np.testing.assert_allclose(
        channel.waveform,
        waveform,
        atol=1e-10,
    )


def test_two_path_channel():
    waveform = np.array(
        [1, 2, 3, 4],
        dtype=np.complex128,
    )

    paths = [
        ChannelPath(
            delay_samples=0,
            doppler_hz=0.0,
            gain=1.0 + 0j,
        ),
        ChannelPath(
            delay_samples=1,
            doppler_hz=0.0,
            gain=0.5 + 0j,
        ),
    ]

    channel = apply_channel(
        waveform,
        paths,
        sample_rate=1000.0,
        snr_db=1000.0,
    )

    expected = np.array(
        [
            1.0,
            2.5,
            4.0,
            5.5,
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        channel.waveform,
        expected,
        atol=1e-10,
    )