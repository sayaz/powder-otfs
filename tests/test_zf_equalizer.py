import numpy as np

from powder_otfs.equalization.zf import zero_forcing_equalizer
from powder_otfs.estimation.perfect import perfect_channel_estimate


def test_zero_forcing_equalizer():

    transmitted = np.array(
        [[1 + 1j, 2 + 2j]],
        dtype=complex,
    )

    channel = np.array(
        [[2 + 0j, 2 + 0j]],
        dtype=complex,
    )

    received = transmitted * channel

    estimate = perfect_channel_estimate(
        channel_response=channel,
        noise_variance=0.0,
    )

    equalized = zero_forcing_equalizer(
        received,
        estimate,
    )

    assert np.allclose(
        equalized.symbols,
        transmitted,
    )

    assert equalized.method == "Zero Forcing"