import numpy as np

from powder_otfs.equalization.equalized import EqualizedGrid
from powder_otfs.estimation.estimate import ChannelEstimate


def mmse_equalizer(
    received_grid: np.ndarray,
    estimate: ChannelEstimate,
    symbol_energy: float = 1.0,
) -> EqualizedGrid:
    """Apply linear MMSE equalization."""

    if symbol_energy <= 0:
        raise ValueError("symbol_energy must be positive.")

    equalized_grids = mmse_equalize_frames(
        received_grids=received_grid[
            np.newaxis,
            ...,
        ],
        estimate=estimate,
        symbol_energy=symbol_energy,
    )

    return EqualizedGrid(
        symbols=equalized_grids[0],
        method="MMSE",
    )


def mmse_equalize_frames(
    received_grids: np.ndarray,
    estimate: ChannelEstimate,
    symbol_energy: float = 1.0,
) -> np.ndarray:
    """MMSE-equalize multiple grids using one channel estimate."""

    if symbol_energy <= 0:
        raise ValueError("symbol_energy must be positive.")

    if received_grids.ndim != 3:
        raise ValueError(
            "received_grids must have shape "
            "(frames, delay_bins, doppler_bins)."
        )

    channel_matrix = estimate.channel_response
    grid_shape = received_grids.shape[1:]
    received_matrix = (
        received_grids.reshape(
            received_grids.shape[0],
            -1,
        ).T
    )

    hermitian = channel_matrix.conj().T

    regularization = (
        estimate.noise_variance
        / symbol_energy
    )

    system_matrix = (
        hermitian @ channel_matrix
        + regularization
        * np.eye(
            channel_matrix.shape[1],
            dtype=np.complex128,
        )
    )

    matched_received = hermitian @ received_matrix

    equalized_matrix = np.linalg.solve(
        system_matrix,
        matched_received,
    )

    return equalized_matrix.T.reshape(
        received_grids.shape[0],
        *grid_shape,
    )
