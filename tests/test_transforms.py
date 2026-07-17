import numpy as np

from powder_otfs.otfs.transforms import isfft, heisenberg

def test_isfft_shape() -> None:
    grid = np.random.randn(32,16) + 1j * np.random.randn(32,16)

    tf_grid = isfft(grid)
    assert tf_grid.shape == (32,16)

def test_heisenberg_output_length() -> None:
    tf_grid = np.random.randn(32,16) + 1j * np.random.randn(32,16)
    
    waveform = heisenberg(tf_grid)
    assert len(waveform) == 32*16