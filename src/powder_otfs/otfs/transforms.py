import numpy as np


def isfft(grid: np.ndarray) -> np.ndarray:
    """Convert a delay-Doppler grid to a time-frequency grid."""

    temp = np.fft.ifft(
        grid,
        axis=0,
        norm="ortho",
    )

    return np.fft.fft(
        temp,
        axis=1,
        norm="ortho",
    )


def heisenberg(tf_grid: np.ndarray) -> np.ndarray:
    """Convert a time-frequency grid to a time-domain waveform."""

    time_grid = np.fft.ifft(
        tf_grid,
        axis=0,
        norm="ortho",
    )

    return time_grid.reshape(
        -1,
        order="F",
    )


def wigner(
    waveform: np.ndarray,
    num_subcarriers: int,
    num_time_slots: int,
) -> np.ndarray:
    """Recover the time-frequency grid from a waveform."""

    time_grid = waveform.reshape(
        num_subcarriers,
        num_time_slots,
        order="F",
    )

    return np.fft.fft(
        time_grid,
        axis=0,
        norm="ortho",
    )


def sfft(tf_grid: np.ndarray) -> np.ndarray:
    """Convert a time-frequency grid to a delay-Doppler grid."""

    temp = np.fft.ifft(
        tf_grid,
        axis=1,
        norm="ortho",
    )

    return np.fft.fft(
        temp,
        axis=0,
        norm="ortho",
    )