import numpy as np

from powder_otfs.channel.path import ChannelPath
from powder_otfs.estimation.pilot import (
    _apply_paths,
    pilot_channel_estimate,
)


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


def test_multipath_estimate_matches_existing_channel_model() -> None:
    grid_shape = (4, 3)
    sample_rate = 1200.0
    pilot_position = (2, 1)
    pilot_value = 2.0 + 0.0j

    paths = [
        ChannelPath(
            delay_samples=0,
            doppler_hz=0.0,
            gain=0.8 + 0.1j,
        ),
        ChannelPath(
            delay_samples=1,
            doppler_hz=100.0,
            gain=0.4 - 0.2j,
        ),
    ]

    pilot_grid = np.zeros(
        grid_shape,
        dtype=np.complex128,
    )
    pilot_grid[pilot_position] = pilot_value

    received_pilot_grid = _apply_paths(
        pilot_grid,
        paths,
        sample_rate,
    )

    estimate = pilot_channel_estimate(
        received_pilot_grid=received_pilot_grid,
        pilot_position=pilot_position,
        pilot_value=pilot_value,
        sample_rate=sample_rate,
        noise_variance=0.0,
        threshold=1e-8,
    )

    test_grid = (
        np.arange(12).reshape(grid_shape)
        + 1j * np.arange(12)[::-1].reshape(grid_shape)
    ) / 10.0

    expected_grid = _apply_paths(
        test_grid,
        paths,
        sample_rate,
    )

    estimated_grid = (
        estimate.channel_response
        @ test_grid.reshape(-1)
    ).reshape(grid_shape)

    np.testing.assert_allclose(
        estimated_grid,
        expected_grid,
        atol=1e-12,
    )


def test_circulant_estimate_for_flat_channel() -> None:
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
        matrix_method="circulant",
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


def test_circulant_matches_basis_for_multipath() -> None:
    grid_shape = (4, 3)
    sample_rate = 1200.0
    pilot_position = (2, 1)
    pilot_value = 2.0 + 0.0j

    paths = [
        ChannelPath(
            delay_samples=0,
            doppler_hz=0.0,
            gain=0.8 + 0.1j,
        ),
        ChannelPath(
            delay_samples=1,
            doppler_hz=100.0,
            gain=0.4 - 0.2j,
        ),
    ]

    pilot_grid = np.zeros(
        grid_shape,
        dtype=np.complex128,
    )
    pilot_grid[pilot_position] = pilot_value

    received_pilot_grid = _apply_paths(
        pilot_grid,
        paths,
        sample_rate,
    )

    basis_estimate = pilot_channel_estimate(
        received_pilot_grid=received_pilot_grid,
        pilot_position=pilot_position,
        pilot_value=pilot_value,
        sample_rate=sample_rate,
        noise_variance=0.0,
        threshold=1e-8,
        matrix_method="basis",
    )

    circulant_estimate = pilot_channel_estimate(
        received_pilot_grid=received_pilot_grid,
        pilot_position=pilot_position,
        pilot_value=pilot_value,
        sample_rate=sample_rate,
        noise_variance=0.0,
        threshold=1e-8,
        matrix_method="circulant",
    )

    np.testing.assert_allclose(
        circulant_estimate.channel_response,
        basis_estimate.channel_response,
        atol=1e-12,
    )


def test_book_rcp_matrix_matches_basis_with_wrapped_paths() -> None:
    grid_shape = (5, 6)
    sample_rate = 3000.0
    pilot_position = (1, 3)
    pilot_value = 3.0 + 0.0j
    doppler_resolution = sample_rate / np.prod(grid_shape)

    paths = [
        ChannelPath(
            delay_samples=0,
            doppler_hz=0.0,
            gain=0.9 + 0.1j,
        ),
        ChannelPath(
            delay_samples=1,
            doppler_hz=doppler_resolution,
            gain=0.4 - 0.2j,
        ),
        ChannelPath(
            delay_samples=3,
            doppler_hz=-2.0 * doppler_resolution,
            gain=-0.1 + 0.3j,
        ),
    ]

    pilot_grid = np.zeros(
        grid_shape,
        dtype=np.complex128,
    )
    pilot_grid[pilot_position] = pilot_value

    received_pilot_grid = _apply_paths(
        pilot_grid,
        paths,
        sample_rate,
    )

    basis_estimate = pilot_channel_estimate(
        received_pilot_grid=received_pilot_grid,
        pilot_position=pilot_position,
        pilot_value=pilot_value,
        sample_rate=sample_rate,
        noise_variance=0.0,
        threshold=1e-8,
        matrix_method="basis",
    )

    circulant_estimate = pilot_channel_estimate(
        received_pilot_grid=received_pilot_grid,
        pilot_position=pilot_position,
        pilot_value=pilot_value,
        sample_rate=sample_rate,
        noise_variance=0.0,
        threshold=1e-8,
        matrix_method="circulant",
    )

    np.testing.assert_allclose(
        circulant_estimate.channel_response,
        basis_estimate.channel_response,
        atol=1e-12,
    )
