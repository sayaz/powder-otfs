import numpy as np

from powder_otfs.estimation.perfect import perfect_channel_estimate


def test_perfect_channel_estimate():

    h = np.array([1 + 0j, 0.5 + 0.2j])

    estimate = perfect_channel_estimate(
        channel_response=h,
        noise_variance=0.01,
    )

    assert np.array_equal(
        estimate.channel_response,
        h,
    )

    assert estimate.noise_variance == 0.01
    assert estimate.method == "Perfect CSI"