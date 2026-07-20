import numpy as np

from powder_otfs.equalization.mmse import mmse_equalizer
from powder_otfs.estimation.pilot import pilot_channel_estimate
from powder_otfs.modulation.qam import qam_demodulate
from powder_otfs.ota.config import OTFSOTAConfig
from powder_otfs.ota.payload import create_otfs_payload
from powder_otfs.otfs.transforms import sfft, wigner


def test_ota_payload_pilot_estimation_and_mmse() -> None:
    config = OTFSOTAConfig(
        num_delay_bins=8,
        num_doppler_bins=8,
        maximum_supported_delay=1,
        maximum_supported_doppler=1,
    )
    transmitted = create_otfs_payload(
        config
    )
    received_dd_grid = sfft(
        wigner(
            transmitted.waveform,
            num_subcarriers=config.num_delay_bins,
            num_time_slots=config.num_doppler_bins,
        )
    )

    pilot_observation = np.zeros_like(
        received_dd_grid
    )
    pilot_observation[
        config.observation_slices
    ] = received_dd_grid[
        config.observation_slices
    ]

    estimate = pilot_channel_estimate(
        received_pilot_grid=pilot_observation,
        pilot_position=config.pilot_position,
        pilot_value=config.pilot_value,
        sample_rate=config.sample_rate,
        noise_variance=0.0,
        threshold=1e-6,
    )
    equalized = mmse_equalizer(
        received_grid=received_dd_grid,
        estimate=estimate,
    )
    received_bits = qam_demodulate(
        equalized.symbols[
            config.data_mask
        ],
        order=config.qam_order,
    )

    np.testing.assert_array_equal(
        received_bits,
        transmitted.bits,
    )
