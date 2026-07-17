import numpy as np

from powder_otfs.otfs.grid import create_delay_doppler_grid

def test_grid_shape() -> None:
    symbols = np.arange(32 * 16)
    grid = create_delay_doppler_grid(symbols, 32, 16)
    assert grid.shape == (32, 16)