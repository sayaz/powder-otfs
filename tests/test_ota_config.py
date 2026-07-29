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


def test_ota_bandwidth_scales_time_parameters() -> None:
    one_mhz = OTFSOTAConfig(
        bandwidth_mhz=1.0,
    )
    ten_mhz = OTFSOTAConfig(
        bandwidth_mhz=10.0,
    )

    assert one_mhz.sample_rate == 1e6
    assert ten_mhz.sample_rate == 10e6
    assert one_mhz.time_guard_samples == 128
    assert ten_mhz.time_guard_samples == 1280


def test_default_ota_bandwidth_is_20_mhz() -> None:
    config = OTFSOTAConfig()

    assert config.bandwidth_mhz == 20.0
    assert config.sample_rate == 20e6
