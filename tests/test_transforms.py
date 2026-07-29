import numpy as np

from powder_otfs.otfs.transforms import heisenberg, isfft

def test_isfft_shape() -> None:
    grid = np.random.randn(32,16) + 1j * np.random.randn(32,16)

    tf_grid = isfft(grid)
    assert tf_grid.shape == (32,16)

def test_heisenberg_output_length() -> None:
    tf_grid = np.random.randn(32,16) + 1j * np.random.randn(32,16)
    
    waveform = heisenberg(tf_grid)
    assert len(waveform) == 32*16


def test_rectangular_pulse_otfs_waveform_convention() -> None:
    """Check the standard rectangular-pulse OTFS transmitter equation."""

    dd_grid = np.array(
        [
            [1.0 + 0.5j, -0.5 + 1.0j, 0.25 - 0.75j],
            [-1.0 + 0.25j, 0.5 - 0.5j, 0.75 + 1.0j],
            [0.25 + 0.5j, -0.75 - 0.25j, 1.0 - 1.0j],
            [0.5 + 0.75j, 1.0 + 0.25j, -0.25 + 0.5j],
        ],
        dtype=np.complex128,
    )

    waveform = heisenberg(isfft(dd_grid))

    expected_time_grid = np.fft.ifft(
        dd_grid,
        axis=1,
        norm="ortho",
    )
    expected_waveform = expected_time_grid.reshape(
        -1,
        order="F",
    )

    np.testing.assert_allclose(
        waveform,
        expected_waveform,
        atol=1e-12,
    )
