import numpy as np

from powder_otfs.estimation.pilot import pilot_channel_estimate


def test_pilot_channel_estimate() -> None:
    grid_shape = (4, 3)
    pilot_position = (2, 1)
    pilot_value = 2.0 + 0.0j
    channel_gain = 0.5 + 0.2j

    received_pilot_grid = np.zeros(
        grid_shape,
        dtype=np.complex128,
    )
    received_pilot_grid[pilot_position] = (
        pilot_value * channel_gain
    )

    estimate = pilot_channel_estimate(
        received_pilot_grid=received_pilot_grid,
        pilot_position=pilot_position,
        pilot_value=pilot_value,
        sample_rate=1000.0,
        noise_variance=0.01,
        threshold=0.1,
    )

    expected = channel_gain * np.eye(
        np.prod(grid_shape),
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        estimate.channel_response,
        expected,
        atol=1e-12,
    )

    assert estimate.method == "Embedded Pilot"