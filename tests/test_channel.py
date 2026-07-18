import numpy as np

from powder_otfs.channel.channel import apply_channel


def test_channel_without_impairments():
    waveform = np.array(
        [1, 2, 3, 4],
        dtype=np.complex128,
    )

    received = apply_channel(
        waveform,
        delay_samples=0,
        doppler_hz=0.0,
        sample_rate=1000.0,
        snr_db=1000.0,
    )

    np.testing.assert_allclose(
        received,
        waveform,
        atol=1e-10,
    )