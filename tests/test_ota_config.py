import numpy as np

from powder_otfs.ota.config import OTFSOTAConfig


def test_ota_config_data_mask() -> None:
    config = OTFSOTAConfig(
        num_delay_bins=32,
        num_doppler_bins=16,
        maximum_supported_delay=3,
        maximum_supported_doppler=2,
    )

    assert config.pilot_position == (16, 8)
    assert config.data_mask.shape == (32, 16)
    assert config.num_data_symbols == 449
    assert config.bits_per_frame == 898
    assert not config.data_mask[
        config.pilot_position
    ]
