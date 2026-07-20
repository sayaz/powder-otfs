import argparse

import numpy as np

from powder_otfs.equalization.mmse import mmse_equalizer
from powder_otfs.equalization.zf import zero_forcing_equalizer
from powder_otfs.estimation.pilot import pilot_channel_estimate
from powder_otfs.modulation.qam import qam_demodulate
from powder_otfs.ota.config import OTFSOTAConfig
from powder_otfs.ota.frequency_offset import (
    correct_cfo,
    estimate_cfo,
)
from powder_otfs.ota.framing import create_preamble
from powder_otfs.ota.payload import create_otfs_payload
from powder_otfs.ota.synchronization import (
    find_payload_starts,
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
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = OTFSOTAConfig()
    transmitted = create_otfs_payload(
        config
    )
    received = np.load(args.capture)
    preamble = create_preamble(
        half_length=config.preamble_half_length,
        seed=config.random_seed,
    )
    frame_length = (
        config.time_guard_samples
        + len(preamble)
        + config.cyclic_prefix_samples
        + config.num_grid_symbols
        + config.time_guard_samples
    )

    preamble_ends = find_payload_starts(
        received=received,
        preamble=preamble,
        threshold=config.synchronization_threshold,
        minimum_separation=frame_length // 2,
    )

    if not 0 <= args.frame_index < len(preamble_ends):
        raise ValueError(
            f"frame-index must be between 0 and "
            f"{len(preamble_ends) - 1}."
        )

    preamble_end = int(
        preamble_ends[args.frame_index]
    )
    preamble_start = (
        preamble_end
        - len(preamble)
    )
    payload_start = (
        preamble_end
        + config.cyclic_prefix_samples
    )
    payload_end = (
        payload_start
        + config.num_grid_symbols
    )

    received_frame = received[
        preamble_start:payload_end
    ]
    received_preamble = received_frame[
        :len(preamble)
    ]
    cfo_hz = estimate_cfo(
        repeated_preamble=received_preamble,
        sample_rate=config.sample_rate,
    )
    corrected_frame = correct_cfo(
        samples=received_frame,
        cfo_hz=cfo_hz,
        sample_rate=config.sample_rate,
    )
    corrected_preamble = corrected_frame[
        :len(preamble)
    ]
    channel_gain = np.vdot(
        preamble,
        corrected_preamble,
    ) / np.vdot(
        preamble,
        preamble,
    )

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
    print(f"Detected Frames      : {len(preamble_ends)}")
    print(f"Displayed Frame      : {args.frame_index}")
    print(f"CFO Estimate         : {cfo_hz:.3f} Hz")
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
