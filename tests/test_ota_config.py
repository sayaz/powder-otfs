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
    assert one_mhz.preamble_half_length == 64
    assert ten_mhz.preamble_half_length == 640
    assert one_mhz.time_guard_samples == 128
    assert ten_mhz.time_guard_samples == 1280
    assert (
        one_mhz.preamble_half_length
        / one_mhz.sample_rate
        == ten_mhz.preamble_half_length
        / ten_mhz.sample_rate
    )
