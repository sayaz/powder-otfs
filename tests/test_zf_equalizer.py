import numpy as np

from powder_otfs.equalization.zf import (
    zero_forcing_equalize_frames,
    zero_forcing_equalizer,
)
from powder_otfs.estimation.estimate import ChannelEstimate


def test_zero_forcing_equalizer() -> None:
    transmitted_grid = np.array(
        [
            [1.0 + 1.0j, 2.0 + 2.0j],
            [3.0 + 3.0j, 4.0 + 4.0j],
        ],
        dtype=np.complex128,
    )

    channel_matrix = 2.0 * np.eye(
        transmitted_grid.size,
        dtype=np.complex128,
    )

    received_vector = (
        channel_matrix @ transmitted_grid.reshape(-1)
    )
    received_grid = received_vector.reshape(
        transmitted_grid.shape
    )

    estimate = ChannelEstimate(
        channel_response=channel_matrix,
        noise_variance=0.0,
        method="Perfect CSI",
    )

    equalized = zero_forcing_equalizer(
        received_grid=received_grid,
        estimate=estimate,
    )

    np.testing.assert_allclose(
        equalized.symbols,
        transmitted_grid,
        atol=1e-12,
    )

    assert equalized.method == "Zero Forcing"


def test_zf_equalizes_multiple_frames() -> None:
    transmitted = np.array(
        [
            [[1.0 + 1.0j]],
            [[-1.0 - 1.0j]],
        ]
    )
    channel = np.array(
        [[2.0 + 0.0j]]
    )
    estimate = ChannelEstimate(
        channel_response=channel,
        noise_variance=0.0,
        method="Perfect CSI",
    )

    equalized = zero_forcing_equalize_frames(
        received_grids=2.0 * transmitted,
        estimate=estimate,
    )

    np.testing.assert_allclose(
        equalized,
        transmitted,
    )
