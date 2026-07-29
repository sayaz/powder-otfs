import argparse

import numpy as np

from powder_otfs.ota.config import (
    add_ota_config_arguments,
    ota_config_from_arguments,
)
from powder_otfs.ota.framing import (
    build_ota_frame,
    create_training_preamble,
    normalize_waveform,
)
from powder_otfs.ota.impairments import (
    add_delayed_path,
    apply_fractional_delay,
    apply_frequency_offset,
)
from powder_otfs.ota.payload import create_otfs_payload
from powder_otfs.ota.runtime import load_radio_runtime_config
from powder_otfs.ota.usrp import (
    configure_usrp_tx,
    transmit_waveform,
)


def parse_arguments() -> argparse.Namespace:
    """Parse transmitter configuration options."""

    parser = argparse.ArgumentParser(
        description="Transmit an OTFS waveform using a POWDER USRP.",
    )
    add_ota_config_arguments(parser)
    parser.add_argument(
        "--tx-gain",
        type=float,
        default=0.0,
        help="USRP transmit gain in dB (default: 0).",
    )
    parser.add_argument(
        "--peak-amplitude",
        type=float,
        default=0.85,
        help=(
            "Peak magnitude of normalized complex baseband samples in "
            "(0, 1] (default: 0.85)."
        ),
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=5000,
        help="Number of repeated frames to transmit (default: 5000).",
    )
    parser.add_argument(
        "--extra-path-delay-samples",
        type=int,
        default=0,
        help="Delay of an optional second path in samples (default: 0).",
    )
    parser.add_argument(
        "--extra-path-gain",
        type=complex,
        default=0.0 + 0.0j,
        help="Complex gain of the optional second path (default: 0).",
    )
    parser.add_argument(
        "--cfo-hz",
        type=float,
        default=0.0,
        help="Controlled carrier-frequency offset in Hz (default: 0).",
    )
    parser.add_argument(
        "--fractional-delay-samples",
        type=float,
        default=0.0,
        help="Controlled fractional-sample delay in [-0.5, 0.5] (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    runtime = load_radio_runtime_config()
    tx_gain = args.tx_gain
    channel = 0
    antenna = "TX/RX"
    peak_amplitude = args.peak_amplitude
    repeat_count = args.repeat_count

    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive.")

    config = ota_config_from_arguments(args)
    payload = create_otfs_payload(
        config
    )
    training = create_training_preamble()
    preamble = training.samples

    tx_frame = build_ota_frame(
        payload=payload.waveform,
        preamble=preamble,
        guard_samples=config.time_guard_samples,
        cyclic_prefix_samples=(
            config.cyclic_prefix_samples
        ),
    )
    tx_frame = add_delayed_path(
        waveform=tx_frame,
        delay_samples=args.extra_path_delay_samples,
        gain=args.extra_path_gain,
    )
    tx_frame = apply_fractional_delay(
        waveform=tx_frame,
        delay_samples=args.fractional_delay_samples,
    )
    tx_frame = normalize_waveform(
        tx_frame,
        peak_amplitude=peak_amplitude,
    )
    repeated_waveform = np.tile(
        tx_frame,
        repeat_count,
    )
    repeated_waveform = apply_frequency_offset(
        waveform=repeated_waveform,
        cfo_hz=args.cfo_hz,
        sample_rate=config.sample_rate,
    ).astype(np.complex64)

    print(
        "\n========== USRP OTFS Transmitter =========="
    )
    print(f"Radio Type         : {runtime.radio_type.upper()}")
    print(f"Device Arguments   : {runtime.device_args}")
    print(
        f"Center Frequency   : "
        f"{runtime.center_frequency / 1e9:.3f} GHz"
    )
    print(f"Sample Rate        : {config.sample_rate:.0f} samples/s")
    print(f"Bandwidth          : {config.bandwidth_mhz:.1f} MHz")
    print(f"TX Gain            : {tx_gain:.1f} dB")
    print(f"Peak Amplitude     : {peak_amplitude:.2f}")
    print(
        f"Extra Delay Path   : "
        f"delay={args.extra_path_delay_samples} samples, "
        f"gain={args.extra_path_gain}"
    )
    print(f"Added CFO          : {args.cfo_hz:.3f} Hz")
    print(
        f"Fractional Delay   : "
        f"{args.fractional_delay_samples:.3f} samples"
    )
    print(f"Modulation         : {config.qam_order}-QAM")
    print(
        f"DD Grid            : "
        f"{config.num_delay_bins} x "
        f"{config.num_doppler_bins}"
    )
    print(f"Data Symbols       : {config.num_data_symbols}")
    print(f"Coded Capacity     : {config.bits_per_frame} bits")
    print(f"FEC                 : {config.fec_name}")
    if config.fec_name != "none":
        print(f"FEC Rate            : {config.fec_rate}")
        print(f"Information Bits    : {config.information_bits_per_frame}")
        print(f"LDPC Codeword       : {config.fec_codeword_length} bits")
        print(f"Filler Bits         : {config.fec_filler_bits}")
    print(f"Pilot Position     : {config.pilot_position}")
    print(f"Pilot Value        : {config.pilot_value}")
    print(
        f"DD Guard Size      : "
        f"{2 * config.guard_delay + 1} x "
        f"{2 * config.guard_doppler + 1}"
    )
    print(
        f"Cyclic Prefix      : "
        f"{config.cyclic_prefix_samples} samples "
        f"({config.cyclic_prefix_samples / config.sample_rate * 1e6:.3f} us)"
    )
    print(f"STF                : {len(training.stf)} samples")
    print(f"LTF                : {len(training.ltf)} samples")
    print(f"Complete Preamble  : {len(preamble)} samples")
    print(
        f"Time Guard         : "
        f"{config.time_guard_samples} samples per side"
    )
    print(f"Frame Length       : {len(tx_frame)} samples")
    print(f"Repeated Frames    : {repeat_count}")
    print(f"Total Samples      : {len(repeated_waveform)}")
    print(
        "===========================================\n"
    )

    usrp = configure_usrp_tx(
        device_args=runtime.device_args,
        sample_rate=config.sample_rate,
        center_frequency=runtime.center_frequency,
        gain=tx_gain,
        channel=channel,
        antenna=antenna,
    )

    sent = transmit_waveform(
        usrp=usrp,
        waveform=repeated_waveform,
        channel=channel,
    )

    print(f"Transmitted samples: {sent}")


if __name__ == "__main__":
    main()
