import numpy as np

from powder_otfs.ota.config import OTFSOTAConfig
from powder_otfs.ota.payload import create_otfs_payload
from powder_otfs.ota.receiver import estimate_and_equalize_frames


def test_each_frame_uses_its_own_pilot_estimate() -> None:
    config = OTFSOTAConfig(
        bandwidth_mhz=1.0,
        maximum_supported_delay=1,
        maximum_supported_doppler=1,
    )
    transmitted = create_otfs_payload(config)
    gains = np.asarray([0.5 + 0.2j, -0.3 + 1.1j])
    received_grids = gains[:, np.newaxis, np.newaxis] * transmitted.dd_grid

    result = estimate_and_equalize_frames(
        received_grids=received_grids,
        config=config,
        channel_block_size=1,
    )

    assert result.rejected_frames == 0
    assert result.equalized_grids.shape == received_grids.shape
    np.testing.assert_allclose(
        result.equalized_grids[:, config.data_mask],
        np.broadcast_to(
            transmitted.data_symbols,
            (len(gains), config.num_data_symbols),
        ),
        atol=1e-8,
    )


def test_each_frame_block_uses_its_own_channel_estimate() -> None:
    config = OTFSOTAConfig(
        bandwidth_mhz=1.0,
        maximum_supported_delay=1,
        maximum_supported_doppler=1,
    )
    transmitted = create_otfs_payload(config)
    gains = np.asarray(
        [
            0.5 + 0.2j,
            0.5 + 0.2j,
            -0.3 + 1.1j,
            -0.3 + 1.1j,
        ]
    )
    received_grids = gains[:, np.newaxis, np.newaxis] * transmitted.dd_grid

    result = estimate_and_equalize_frames(
        received_grids=received_grids,
        config=config,
        channel_block_size=2,
    )

    assert result.rejected_frames == 0
    np.testing.assert_allclose(
        result.equalized_grids[:, config.data_mask],
        np.broadcast_to(
            transmitted.data_symbols,
            (len(gains), config.num_data_symbols),
        ),
        atol=1e-8,
    )
