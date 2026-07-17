import numpy as np

def isfft(grid: np.ndarray) -> np.ndarray:
    """Inverse Symplectic Finite Fourier Transform (ISFFT)."""

    temp = np.fft.ifft(grid, axis=0)
    return np.fft.fft(temp, axis=1)

def heisenberg(tf_grid: np.ndarray) -> np.ndarray:
    """Convert the time-frequency grid to a time-domain waveform."""

    time_grid = np.fft.ifft(tf_grid, axis=0)
    return time_grid.reshape(-1)