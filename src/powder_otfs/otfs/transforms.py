import numpy as np

def isfft(grid: np.ndarray) -> np.ndarray:
    """Inverse Symplectic Finite Fourier Transform (ISFFT)."""

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
    """Convert the time-frequency grid to a time-domain waveform."""

    time_grid = np.fft.ifft(
        tf_grid,
        axis=0,
        norm="ortho",
    )

    return time_grid.reshape(-1)

def wigner(
        waveform: np.ndarray,
        num_subcarriers: int,
        num_time_slots: int,
) -> np.ndarray:
    """Apply the Wigner transforms to recover the TF grid"""

    tf_grid = waveform.reshape(
        num_subcarriers,
        num_time_slots,
    )

    return np.fft.fft(
        tf_grid,
        axis = 0,
        norm = "ortho",
    )

def sfft(
    tf_grid: np.ndarray,
) -> np.ndarray:
    """Apply the Symplectic FFT to recover the DD grid"""

    return np.fft.fft(
        np.fft.ifft(
            tf_grid,
            axis=1,
            norm="ortho",
        ),
        axis=0,
        norm="ortho",
    )