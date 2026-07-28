import numpy as np

from powder_otfs.channel.delay import apply_circular_delay
from powder_otfs.equalization.mmse import mmse_equalizer
from powder_otfs.estimation.pilot import pilot_channel_estimate
from powder_otfs.modulation.qam import qam_demodulate
from powder_otfs.ota.config import OTFSOTAConfig
from powder_otfs.ota.payload import create_otfs_payload
from powder_otfs.otfs.transforms import heisenberg, isfft, sfft, wigner


def test_pilot_observation_window_follows_positive_delay_region() -> None:
    config = OTFSOTAConfig(
        num_delay_bins=32,
        num_doppler_bins=16,
        maximum_supported_delay=4,
        maximum_supported_doppler=2,
    )

    delay_slice, doppler_slice = config.observation_slices

    assert delay_slice == slice(16, 21)
    assert doppler_slice == slice(6, 11)


def test_positive_delay_pilot_is_inside_observation_window() -> None:
    config = OTFSOTAConfig(
        num_delay_bins=32,
        num_doppler_bins=16,
        maximum_supported_delay=4,
        maximum_supported_doppler=2,
    )
    pilot_grid = np.zeros(config.grid_shape, dtype=np.complex128)
    pilot_grid[config.pilot_position] = config.pilot_value
    waveform = heisenberg(isfft(pilot_grid))
    delayed_waveform = apply_circular_delay(waveform, delay_samples=1)
    received_grid = sfft(
        wigner(
            delayed_waveform,
            num_subcarriers=config.num_delay_bins,
            num_time_slots=config.num_doppler_bins,
        )
    )

    peak_position = np.unravel_index(
        np.argmax(np.abs(received_grid)),
        received_grid.shape,
    )
    observed = np.zeros_like(received_grid)
    observed[config.observation_slices] = received_grid[
        config.observation_slices
    ]

    assert peak_position == (17, 8)
    assert observed[peak_position] != 0.0


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
