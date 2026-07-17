import numpy as np

def isfft(grid: np.ndarray) -> np.ndarray:
    """Inverse Symplectic Finite Fourier Transform (ISFFT)."""

    temp = np.fft.ifft(grid, axis=0)
    return np.fft.fft(temp, axis=1)