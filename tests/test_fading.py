import numpy as np

from powder_otfs.channel.fading import apply_fading
from powder_otfs.channel.path import ChannelPath


def test_fixed_fading_preserves_path() -> None:
    paths = [
        ChannelPath(
            delay_samples=2,
            doppler_hz=100.0,
            gain=0.5 + 0.2j,
        )
    ]

    faded = apply_fading(
        paths=paths,
        model="fixed",
        rng=np.random.default_rng(1),
    )

    assert faded[0] == paths[0]


def test_rayleigh_fading_preserves_delay_and_doppler() -> None:
    path = ChannelPath(
        delay_samples=3,
        doppler_hz=-200.0,
        gain=1.0 + 0.0j,
    )

    faded = apply_fading(
        paths=[path],
        model="rayleigh",
        rng=np.random.default_rng(1),
    )

    assert faded[0].delay_samples == path.delay_samples
    assert faded[0].doppler_hz == path.doppler_hz
    assert faded[0].gain != path.gain


def test_rician_fading_preserves_delay_and_doppler() -> None:
    path = ChannelPath(
        delay_samples=1,
        doppler_hz=300.0,
        gain=1.0 + 0.0j,
    )

    faded = apply_fading(
        paths=[path],
        model="rician",
        rng=np.random.default_rng(1),
        rician_k=5.0,
    )

    assert faded[0].delay_samples == path.delay_samples
    assert faded[0].doppler_hz == path.doppler_hz