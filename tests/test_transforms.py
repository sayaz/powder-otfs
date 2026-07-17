import numpy as np

from powder_otfs.otfs.transforms import isfft

def test_isfft_shape() -> None:
    grid = np.random.randn(32,16) + 1j * np.random.randn(32,16)

    tf_grid = isfft(grid)
    assert tf_grid.shape == (32,16)