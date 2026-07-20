from pathlib import Path

import numpy as np

from powder_otfs.equalization.mmse import (
    mmse_equalize_frames,
)
from powder_otfs.equalization.zf import (
    zero_forcing_equalize_frames,
)
from powder_otfs.estimation.pilot import (
    pilot_channel_estimate,
)
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
from powder_otfs.ota.x310 import (
    configure_x310_rx,
    receive_samples,
)
from powder_otfs.otfs.transforms import (
    sfft,
    wigner,
)


def main() -> None:
    device_address = "192.168.40.2"
    center_frequency = 3.5e9
    rx_gain = 20.0
    channel = 0
    antenna = "RX2"
    capture_samples = 6_000_000
    save_received_samples = True
    received_samples_path = Path(
        "results/rx_samples.npy"
    )

    config = OTFSOTAConfig()
    transmitted = create_otfs_payload(
        config
    )
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

    print(
        "\n========== X310 OTFS Receiver Configuration =========="
    )
    print(f"Device Address       : {device_address}")
    print(f"Center Frequency     : {center_frequency / 1e9:.3f} GHz")
    print(f"Sample Rate          : {config.sample_rate:.0f} samples/s")
    print(f"RX Gain              : {rx_gain:.1f} dB")
    print(f"Channel              : {channel}")
    print(f"RX Antenna           : {antenna}")
    print(f"Capture Samples      : {capture_samples}")
    print(
        f"Capture Duration     : "
        f"{capture_samples / config.sample_rate:.3f} s"
    )
    print(f"Save IQ Samples      : {save_received_samples}")
    if save_received_samples:
        print(f"IQ Output File       : {received_samples_path}")
    print(f"Modulation           : {config.qam_order}-QAM")
    print(
        f"DD Grid              : "
        f"{config.num_delay_bins} x "
        f"{config.num_doppler_bins}"
    )
    print(f"Data Symbols         : {config.num_data_symbols}")
    print(f"Bits per Frame       : {config.bits_per_frame}")
    print(f"Pilot Position       : {config.pilot_position}")
    print(f"Pilot Value          : {config.pilot_value}")
    print(
        f"DD Guard Size        : "
        f"{2 * config.guard_delay + 1} x "
        f"{2 * config.guard_doppler + 1}"
    )
    print(
        f"Supported Delay      : "
        f"0 to {config.maximum_supported_delay} samples"
    )
    print(
        f"Supported Doppler    : "
        f"±{config.maximum_supported_doppler} bins"
    )
    print(
        f"Cyclic Prefix        : "
        f"{config.cyclic_prefix_samples} samples"
    )
    print(f"Preamble             : {len(preamble)} samples")
    print(
        f"Time Guard           : "
        f"{config.time_guard_samples} samples per side"
    )
    print(f"Complete Frame       : {frame_length} samples")
    print(
        f"Sync Threshold       : "
        f"{config.synchronization_threshold:.2f}"
    )
    print("CFO Correction       : Enabled")
    print(f"Channel Estimator    : Embedded Pilot")
    print(f"Equalizer            : {config.equalizer_name.upper()}")
    print(
        "=======================================================\n"
    )
    print("Waiting for samples...")

    usrp = configure_x310_rx(
        device_address=device_address,
        sample_rate=config.sample_rate,
        center_frequency=center_frequency,
        gain=rx_gain,
        channel=channel,
        antenna=antenna,
    )
    received = receive_samples(
        usrp=usrp,
        num_samples=capture_samples,
        channel=channel,
        timeout=5.0,
    )

    if save_received_samples:
        received_samples_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        np.save(
            received_samples_path,
            received,
        )
        print(
            f"Received IQ saved to "
            f"{received_samples_path.resolve()}"
        )

    print("Capture complete. Detecting frames...")

    preamble_ends = find_payload_starts(
        received=received,
        preamble=preamble,
        threshold=config.synchronization_threshold,
        minimum_separation=frame_length // 2,
    )

    received_dd_grids: list[np.ndarray] = []
    channel_gains: list[complex] = []
    cfo_estimates_hz: list[float] = []
    rejected_frames = 0

    for preamble_end in preamble_ends:
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

        if (
            preamble_start < 0
            or payload_end > len(received)
        ):
            rejected_frames += 1
            continue

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

        if abs(channel_gain) < 1e-12:
            rejected_frames += 1
            continue

        payload_offset = (
            len(preamble)
            + config.cyclic_prefix_samples
        )
        corrected_payload = (
            corrected_frame[payload_offset:]
            / channel_gain
        )

        received_tf_grid = wigner(
            corrected_payload,
            num_subcarriers=config.num_delay_bins,
            num_time_slots=config.num_doppler_bins,
        )
        received_dd_grid = sfft(
            received_tf_grid
        )

        received_dd_grids.append(
            received_dd_grid
        )
        channel_gains.append(channel_gain)
        cfo_estimates_hz.append(cfo_hz)

    if not received_dd_grids:
        raise RuntimeError(
            "No complete valid OTFS frames were decoded."
        )

    received_grids = np.stack(
        received_dd_grids
    )
    average_received_grid = np.mean(
        received_grids,
        axis=0,
    )

    pilot_observation = np.zeros_like(
        average_received_grid
    )
    observation_slices = (
        config.observation_slices
    )
    pilot_observation[
        observation_slices
    ] = average_received_grid[
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
    noise_samples = received_grids[
        :,
        noise_mask,
    ]
    noise_variance = max(
        float(
            np.median(
                np.abs(noise_samples) ** 2
            )
            / np.log(2.0)
        ),
        1e-12,
    )
    estimation_threshold = (
        config.threshold_factor
        * np.sqrt(noise_variance)
    )

    estimate = pilot_channel_estimate(
        received_pilot_grid=pilot_observation,
        pilot_position=config.pilot_position,
        pilot_value=config.pilot_value,
        sample_rate=config.sample_rate,
        noise_variance=noise_variance,
        threshold=estimation_threshold,
    )

    if config.equalizer_name.lower() == "mmse":
        equalized_grids = mmse_equalize_frames(
            received_grids=received_grids,
            estimate=estimate,
            symbol_energy=1.0,
        )
        equalizer_method = "MMSE"
    elif config.equalizer_name.lower() == "zf":
        equalized_grids = zero_forcing_equalize_frames(
            received_grids=received_grids,
            estimate=estimate,
        )
        equalizer_method = "Zero Forcing"
    else:
        raise ValueError(
            "equalizer_name must be 'zf' or 'mmse'."
        )

    frame_bers: list[float] = []
    symbol_mses: list[float] = []
    total_bit_errors = 0

    for equalized_grid in equalized_grids:
        received_symbols = equalized_grid[
            config.data_mask
        ]
        received_bits = qam_demodulate(
            received_symbols,
            order=config.qam_order,
        )
        bit_errors = int(
            np.count_nonzero(
                transmitted.bits
                != received_bits
            )
        )
        total_bit_errors += bit_errors
        frame_bers.append(
            bit_errors
            / config.bits_per_frame
        )
        symbol_mses.append(
            float(
                np.mean(
                    np.abs(
                        received_symbols
                        - transmitted.data_symbols
                    ) ** 2
                )
            )
        )

    processed_frames = len(
        equalized_grids
    )
    processed_bits = (
        processed_frames
        * config.bits_per_frame
    )
    aggregate_ber = (
        total_bit_errors
        / processed_bits
    )
    gain_magnitudes = np.abs(
        np.asarray(channel_gains)
    )
    cfo_estimates = np.asarray(
        cfo_estimates_hz
    )

    print(
        "\n========== X310 OTFS Multi-Frame Result =========="
    )
    print(f"Detected Frames       : {len(preamble_ends)}")
    print(f"Processed Frames      : {processed_frames}")
    print(f"Rejected Frames       : {rejected_frames}")
    print(f"Processed Bits        : {processed_bits}")
    print(f"Bit Errors            : {total_bit_errors}")
    print(f"Aggregate BER         : {aggregate_ber:.6f}")
    print(f"Mean Frame BER        : {np.mean(frame_bers):.6f}")
    print(f"Minimum Frame BER     : {np.min(frame_bers):.6f}")
    print(f"Maximum Frame BER     : {np.max(frame_bers):.6f}")
    print(f"Mean Symbol MSE       : {np.mean(symbol_mses):.6e}")
    print(f"Mean Channel Magnitude: {np.mean(gain_magnitudes):.6e}")
    print(f"Mean CFO Estimate     : {np.mean(cfo_estimates):.3f} Hz")
    print(f"CFO Standard Deviation: {np.std(cfo_estimates):.3f} Hz")
    print(f"Noise Variance        : {noise_variance:.6e}")
    print(f"Estimation Threshold  : {estimation_threshold:.6e}")
    print(f"Channel Estimator     : {estimate.method}")
    print(f"Equalizer             : {equalizer_method}")
    print(
        "===================================================\n"
    )


if __name__ == "__main__":
    main()
