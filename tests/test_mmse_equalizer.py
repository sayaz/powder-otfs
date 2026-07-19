import numpy as np

from powder_otfs.equalization.mmse import mmse_equalizer
from powder_otfs.estimation.estimate import ChannelEstimate


def test_mmse_equalizer() -> None:
    received_grid = np.array(
        [[2.0 + 2.0j]],
        dtype=np.complex128,
    )

    channel_matrix = np.array(
        [[2.0 + 0.0j]],
        dtype=np.complex128,
    )

    estimate = ChannelEstimate(
        channel_response=channel_matrix,
        noise_variance=0.1,
        method="Test Channel",
    )

    equalized = mmse_equalizer(
        received_grid=received_grid,
        estimate=estimate,
        symbol_energy=1.0,
    )

    expected = np.array(
        [[
            (4.0 + 4.0j) / 4.1
        ]],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        equalized.symbols,
        expected,
        atol=1e-12,
    )

    assert equalized.method == "MMSE"