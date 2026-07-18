from powder_otfs.channel.path import ChannelPath


def test_channel_path():
    path = ChannelPath(
        delay_samples=3,
        doppler_hz=150.0,
        gain=0.5 + 0.2j,
    )

    assert path.delay_samples == 3
    assert path.doppler_hz == 150.0
    assert path.gain == 0.5 + 0.2j