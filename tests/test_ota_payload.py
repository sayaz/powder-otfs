import numpy as np

from powder_otfs.ota.config import OTFSOTAConfig
from powder_otfs.ota.payload import create_otfs_payload


def test_ota_payload_contains_data_pilot_and_guards() -> None:
    config = OTFSOTAConfig()
    payload = create_otfs_payload(
        config
    )

    assert len(payload.bits) == config.bits_per_frame
    assert len(payload.data_symbols) == config.num_data_symbols
    assert payload.dd_grid.shape == config.grid_shape
    assert len(payload.waveform) == config.num_grid_symbols
    assert payload.dd_grid[
        config.pilot_position
    ] == config.pilot_value
    np.testing.assert_allclose(
        payload.dd_grid[
            config.data_mask
        ],
        payload.data_symbols,
    )
