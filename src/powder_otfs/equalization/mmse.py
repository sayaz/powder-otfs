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

    channel_matrix = estimate.channel_response
    received_vector = received_grid.reshape(-1)

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

    matched_received = (
        hermitian @ received_vector
    )

    equalized_vector = np.linalg.solve(
        system_matrix,
        matched_received,
    )

    equalized_grid = equalized_vector.reshape(
        received_grid.shape
    )

    return EqualizedGrid(
        symbols=equalized_grid,
        method="MMSE",
    )