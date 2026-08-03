import argparse
from pathlib import Path

import numpy as np

from powder_otfs.fec.qc_ldpc import create_qc_ldpc_code
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
from powder_otfs.ota.runtime import load_radio_runtime_config
from powder_otfs.ota.receiver import estimate_and_equalize_frames
from powder_otfs.ota.usrp import (
    configure_usrp_rx,
    receive_samples,
)
from powder_otfs.otfs.transforms import (
    sfft,
    wigner,
)


def parse_arguments() -> argparse.Namespace:
    """Parse receiver configuration options."""

    parser = argparse.ArgumentParser(
        description="Receive and decode OTFS frames using a POWDER USRP.",
    )
    add_ota_config_arguments(parser)
    parser.add_argument(
        "--rx-gain",
        type=float,
        default=20.0,
        help="USRP receive gain in dB (default: 20).",
    )
    parser.add_argument(
        "--capture-duration",
        type=float,
        default=6.0,
        help="Receive-capture duration in seconds (default: 6).",
    )
    parser.add_argument(
        "--channel-block-size",
        type=int,
        default=50,
        help=(
            "Frames sharing one channel estimate "
            "(default: 50; use 1 for per-frame estimation)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    runtime = load_radio_runtime_config()
    rx_gain = args.rx_gain
    channel = 0
    antenna = "RX2"
    if args.capture_duration <= 0.0:
        raise ValueError("capture_duration must be positive.")
    if args.channel_block_size <= 0:
        raise ValueError("channel_block_size must be positive.")

    config = ota_config_from_arguments(args)
    capture_samples = int(
        round(
            args.capture_duration
            * config.sample_rate
        )
    )
    save_received_samples = True
    received_samples_path = Path(
        "results/rx_samples.npy"
    )

    transmitted = create_otfs_payload(
        config
    )
    training = create_training_preamble()
    preamble = training.samples

    frame_length = (
        config.time_guard_samples
        + len(preamble)
        + config.cyclic_prefix_samples
        + config.num_grid_symbols
        + config.time_guard_samples
    )

    print(
        "\n========== USRP OTFS Receiver Configuration =========="
    )
    print(f"Radio Type           : {runtime.radio_type.upper()}")
    print(f"Device Arguments     : {runtime.device_args}")
    print(
        f"Center Frequency     : "
        f"{runtime.center_frequency / 1e9:.3f} GHz"
    )
    print(f"Sample Rate          : {config.sample_rate:.0f} samples/s")
    print(f"Bandwidth            : {config.bandwidth_mhz:.1f} MHz")
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
    print(f"Coded Capacity       : {config.bits_per_frame} bits")
    print(f"FEC                  : {config.fec_name}")
    if config.fec_name != "none":
        print(f"FEC Rate             : {config.fec_rate}")
        print(f"Information Bits     : {config.information_bits_per_frame}")
        print(f"LDPC Codeword        : {config.fec_codeword_length} bits")
        print(f"Filler Bits          : {config.fec_filler_bits}")
        print(f"LDPC Iterations      : {config.ldpc_iterations}")
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
        f"{config.cyclic_prefix_samples} samples "
        f"({config.cyclic_prefix_samples / config.sample_rate * 1e6:.3f} us)"
    )
    print(f"STF                  : {len(training.stf)} samples")
    print(f"LTF                  : {len(training.ltf)} samples")
    print(f"Complete Preamble    : {len(preamble)} samples")
    print(
        f"Time Guard           : "
        f"{config.time_guard_samples} samples per side"
    )
    print(f"Complete Frame       : {frame_length} samples")
    print(
        f"STF Threshold        : "
        f"{config.stf_detection_threshold:.2f}"
    )
    print(
        f"LTF Threshold        : "
        f"{config.ltf_detection_threshold:.2f}"
    )
    print("Fractional Timing    : Enabled")
    print("CFO Correction       : Enabled")
    print(f"Channel Estimator    : Embedded Pilot")
    print(f"Channel Block Size   : {args.channel_block_size} frames")
    print(f"Equalizer            : {config.equalizer_name.upper()}")
    print(
        "=======================================================\n"
    )
    print("Waiting for samples...")

    usrp = configure_usrp_rx(
        device_args=runtime.device_args,
        sample_rate=config.sample_rate,
        center_frequency=runtime.center_frequency,
        gain=rx_gain,
        channel=channel,
        antenna=antenna,
        clock_source=runtime.clock_source,
        time_source=runtime.time_source,
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

    preamble_starts = find_training_preamble_starts(
        received=received,
        preamble=training,
        stf_threshold=config.stf_detection_threshold,
        ltf_threshold=config.ltf_detection_threshold,
        minimum_separation=frame_length // 2,
    )

    received_dd_grids: list[np.ndarray] = []
    channel_gains: list[complex] = []
    cfo_estimates_hz: list[float] = []
    fractional_timing_offsets: list[float] = []
    rejected_frames = 0

    for preamble_start in preamble_starts:
        payload_start = (
            preamble_start
            + len(preamble)
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
        cfo_estimates_hz.append(
            coarse_cfo_hz + fine_cfo_hz
        )
        fractional_timing_offsets.append(fractional_offset)

    if not received_dd_grids:
        raise RuntimeError(
            "No complete valid OTFS frames were decoded."
        )

    received_grids = np.stack(
        received_dd_grids
    )
    frame_processing = estimate_and_equalize_frames(
        received_grids=received_grids,
        config=config,
        channel_block_size=args.channel_block_size,
    )
    equalized_grids = frame_processing.equalized_grids
    rejected_frames += frame_processing.rejected_frames

    frame_bers: list[float] = []
    symbol_mses: list[float] = []
    total_bit_errors = 0
    total_pre_fec_errors = 0

    for equalized_grid in equalized_grids:
        received_symbols = equalized_grid[
            config.data_mask
        ]
        hard_bits = qam_demodulate(
            received_symbols,
            order=config.qam_order,
        )
        if config.fec_name == "qc-ldpc":
            code = create_qc_ldpc_code(
                config.fec_rate,
                config.bits_per_frame,
            )
            interleaved_llrs = np.empty(config.bits_per_frame, dtype=float)
            interleaved_llrs[0::2] = received_symbols.real
            interleaved_llrs[1::2] = received_symbols.imag
            codeword_llrs = np.empty(config.fec_codeword_length, dtype=float)
            codeword_llrs[config.fec_interleaver] = interleaved_llrs[
                :config.fec_codeword_length
            ]
            received_bits = code.decode(
                codeword_llrs,
                maximum_iterations=config.ldpc_iterations,
            )
            hard_codeword = np.empty(config.fec_codeword_length, dtype=np.uint8)
            hard_codeword[config.fec_interleaver] = hard_bits[
                :config.fec_codeword_length
            ]
            total_pre_fec_errors += int(
                np.count_nonzero(hard_codeword != transmitted.codeword_bits)
            )
            expected_bits = transmitted.information_bits
        else:
            received_bits = hard_bits
            expected_bits = transmitted.information_bits
        bit_errors = int(
            np.count_nonzero(
                expected_bits
                != received_bits
            )
        )
        total_bit_errors += bit_errors
        frame_bers.append(
            bit_errors
            / config.information_bits_per_frame
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
        * config.information_bits_per_frame
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
    print(f"Detected Frames       : {len(preamble_starts)}")
    print(f"Processed Frames      : {processed_frames}")
    print(f"Rejected Frames       : {rejected_frames}")
    print(f"Processed Bits        : {processed_bits}")
    print(f"Bit Errors            : {total_bit_errors}")
    if config.fec_name != "none":
        pre_fec_bits = processed_frames * config.fec_codeword_length
        print(f"Pre-FEC Coded BER     : {total_pre_fec_errors / pre_fec_bits:.6f}")
    print(f"Aggregate BER         : {aggregate_ber:.6f}")
    print(f"Mean Frame BER        : {np.mean(frame_bers):.6f}")
    print(f"Minimum Frame BER     : {np.min(frame_bers):.6f}")
    print(f"Maximum Frame BER     : {np.max(frame_bers):.6f}")
    print(f"Mean Symbol MSE       : {np.mean(symbol_mses):.6e}")
    print(f"Mean Channel Magnitude: {np.mean(gain_magnitudes):.6e}")
    print(f"Mean CFO Estimate     : {np.mean(cfo_estimates):.3f} Hz")
    print(f"CFO Standard Deviation: {np.std(cfo_estimates):.3f} Hz")
    print(
        f"Mean Fractional Offset: "
        f"{np.mean(fractional_timing_offsets):.4f} samples"
    )
    print(
        f"Fractional Offset Std : "
        f"{np.std(fractional_timing_offsets):.4f} samples"
    )
    print(
        f"Mean Noise Variance   : "
        f"{np.mean(frame_processing.noise_variances):.6e}"
    )
    print(
        f"Mean Est. Threshold   : "
        f"{np.mean(frame_processing.estimation_thresholds):.6e}"
    )
    print(
        f"Channel Estimator     : "
        f"{frame_processing.estimator_method} "
        f"(blocks of {args.channel_block_size} frames)"
    )
    print(f"Equalizer             : {frame_processing.equalizer_method}")
    print(
        "===================================================\n"
    )


if __name__ == "__main__":
    main()
