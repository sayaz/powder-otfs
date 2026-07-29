import argparse

import numpy as np

from powder_otfs.equalization.mmse import mmse_equalizer
from powder_otfs.equalization.zf import zero_forcing_equalizer
from powder_otfs.estimation.pilot import pilot_channel_estimate
from powder_otfs.modulation.qam import qam_demodulate
from powder_otfs.ota.config import (
    add_ota_config_arguments,
    ota_config_from_arguments,
)
from powder_otfs.ota.frequency_offset import (
    correct_cfo,
    estimate_repeated_symbol_cfo,
)
from powder_otfs.ota.framing import create_training_preamble
from powder_otfs.ota.payload import create_otfs_payload
from powder_otfs.ota.synchronization import (
    correct_fractional_timing,
    estimate_fractional_timing_offset,
    find_training_preamble_starts,
)
from powder_otfs.otfs.transforms import (
    sfft,
    wigner,
)
from powder_otfs.visualization.plots import (
    plot_otfs_debug_view,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode and visualize a saved X310 OTFS IQ capture."
        ),
    )
    parser.add_argument(
        "capture",
        help="Path to the rx_samples.npy capture.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Detected frame index to display.",
    )
    add_ota_config_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = ota_config_from_arguments(args)
    transmitted = create_otfs_payload(
        config
    )
    received = np.load(args.capture)
    training = create_training_preamble()
    preamble = training.samples
    frame_length = (
        config.time_guard_samples
        + len(preamble)
        + config.cyclic_prefix_samples
        + config.num_grid_symbols
        + config.time_guard_samples
    )

    preamble_starts = find_training_preamble_starts(
        received=received,
        preamble=training,
        stf_threshold=config.stf_detection_threshold,
        ltf_threshold=config.ltf_detection_threshold,
        minimum_separation=frame_length // 2,
    )

    if not 0 <= args.frame_index < len(preamble_starts):
        raise ValueError(
            f"frame-index must be between 0 and "
            f"{len(preamble_starts) - 1}."
        )

    preamble_start = int(
        preamble_starts[args.frame_index]
    )
    payload_start = (
        preamble_start
        + len(preamble)
        + config.cyclic_prefix_samples
    )
    payload_end = (
        payload_start
        + config.num_grid_symbols
    )

    received_frame = received[
        preamble_start:payload_end
    ]
    coarse_cfo_hz = estimate_repeated_symbol_cfo(
        repeated_symbols=received_frame[:len(training.stf)],
        symbol_length=len(training.stf_short_symbol),
        sample_rate=config.sample_rate,
    )
    coarse_corrected_frame = correct_cfo(
        samples=received_frame,
        cfo_hz=coarse_cfo_hz,
        sample_rate=config.sample_rate,
    )
    fractional_offset = estimate_fractional_timing_offset(
        received=coarse_corrected_frame,
        known_sequence=training.ltf_symbol,
        integer_start=training.ltf_symbol_offset,
    )
    timing_corrected_frame = correct_fractional_timing(
        samples=coarse_corrected_frame,
        offset_samples=fractional_offset,
    )
    ltf_symbol_start = training.ltf_symbol_offset
    repeated_ltf = timing_corrected_frame[
        ltf_symbol_start:
        ltf_symbol_start + 2 * len(training.ltf_symbol)
    ]
    fine_cfo_hz = estimate_repeated_symbol_cfo(
        repeated_symbols=repeated_ltf,
        symbol_length=len(training.ltf_symbol),
        sample_rate=config.sample_rate,
    )
    corrected_frame = correct_cfo(
        samples=timing_corrected_frame,
        cfo_hz=fine_cfo_hz,
        sample_rate=config.sample_rate,
    )
    corrected_ltf = corrected_frame[
        len(training.stf):len(preamble)
    ]
    channel_gain = np.vdot(
        training.ltf,
        corrected_ltf,
    ) / np.vdot(
        training.ltf,
        training.ltf,
    )
    cfo_hz = coarse_cfo_hz + fine_cfo_hz

    payload_offset = (
        len(preamble)
        + config.cyclic_prefix_samples
    )
    corrected_payload = (
        corrected_frame[payload_offset:]
        / channel_gain
    )
    rx_dd_grid = sfft(
        wigner(
            corrected_payload,
            num_subcarriers=config.num_delay_bins,
            num_time_slots=config.num_doppler_bins,
        )
    )

    pilot_observation = np.zeros_like(
        rx_dd_grid
    )
    observation_slices = (
        config.observation_slices
    )
    pilot_observation[
        observation_slices
    ] = rx_dd_grid[
        observation_slices
    ]

    observation_mask = np.zeros(
        config.grid_shape,
        dtype=bool,
    )
    observation_mask[
        observation_slices
    ] = True
    noise_mask = (
        ~config.data_mask
        & ~observation_mask
    )
    noise_variance = max(
        float(
            np.median(
                np.abs(
                    rx_dd_grid[noise_mask]
                ) ** 2
            )
            / np.log(2.0)
        ),
        1e-12,
    )
    threshold = (
        config.threshold_factor
        * np.sqrt(noise_variance)
    )

    estimate = pilot_channel_estimate(
        received_pilot_grid=pilot_observation,
        pilot_position=config.pilot_position,
        pilot_value=config.pilot_value,
        sample_rate=config.sample_rate,
        noise_variance=noise_variance,
        threshold=threshold,
    )

    if config.equalizer_name.lower() == "mmse":
        equalized = mmse_equalizer(
            received_grid=rx_dd_grid,
            estimate=estimate,
            symbol_energy=1.0,
        )
    elif config.equalizer_name.lower() == "zf":
        equalized = zero_forcing_equalizer(
            received_grid=rx_dd_grid,
            estimate=estimate,
        )
    else:
        raise ValueError(
            "equalizer_name must be 'zf' or 'mmse'."
        )

    received_symbols = equalized.symbols[
        config.data_mask
    ]
    received_bits = qam_demodulate(
        received_symbols,
        order=config.qam_order,
    )
    bit_errors = int(
        np.count_nonzero(
            received_bits
            != transmitted.bits
        )
    )
    ber = (
        bit_errors
        / config.bits_per_frame
    )

    print(
        "\n========== Offline OTA Debug =========="
    )
    print(f"Capture Samples      : {len(received)}")
    print(f"Detected Frames      : {len(preamble_starts)}")
    print(f"Displayed Frame      : {args.frame_index}")
    print(f"CFO Estimate         : {cfo_hz:.3f} Hz")
    print(f"Fractional Offset    : {fractional_offset:.4f} samples")
    print(f"Channel Gain         : {channel_gain}")
    print(f"Noise Variance       : {noise_variance:.6e}")
    print(f"Estimation Threshold : {threshold:.6e}")
    print(f"Channel Estimator    : {estimate.method}")
    print(f"Equalizer            : {equalized.method}")
    print(f"Frame Bit Errors     : {bit_errors}")
    print(f"Frame BER            : {ber:.6f}")
    print(
        "=======================================\n"
    )

    plot_otfs_debug_view(
        tx_dd_grid=transmitted.dd_grid,
        rx_dd_grid=rx_dd_grid,
        equalized_dd_grid=equalized.symbols,
        pilot_observation=pilot_observation,
        data_mask=config.data_mask,
        pilot_position=config.pilot_position,
    )


if __name__ == "__main__":
    main()
