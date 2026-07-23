import argparse

import numpy as np

from powder_otfs.metrics.ber import bit_error_rate
from powder_otfs.modulation.qam import qam_demodulate
from powder_otfs.ota.config import (
    add_ota_config_arguments,
    ota_config_from_arguments,
)
from powder_otfs.ota.framing import (
    build_ota_frame,
    create_preamble,
    normalize_waveform,
)
from powder_otfs.ota.payload import create_otfs_payload
from powder_otfs.ota.synchronization import find_payload_start
from powder_otfs.otfs.transforms import sfft, wigner


def parse_arguments() -> argparse.Namespace:
    """Parse offline-loopback configuration options."""

    parser = argparse.ArgumentParser(
        description="Run the complete OTFS OTA framing loopback offline.",
    )
    add_ota_config_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = ota_config_from_arguments(args)
    transmitted = create_otfs_payload(config)

    preamble = create_preamble(
        half_length=config.preamble_half_length,
        seed=config.random_seed,
    )
    tx_frame = build_ota_frame(
        payload=transmitted.waveform,
        preamble=preamble,
        guard_samples=config.time_guard_samples,
        cyclic_prefix_samples=config.cyclic_prefix_samples,
    )
    tx_frame = normalize_waveform(
        tx_frame,
        peak_amplitude=0.8,
    )

    capture_padding = config.time_guard_samples
    rx_capture = np.concatenate(
        (
            np.zeros(capture_padding, dtype=np.complex64),
            tx_frame,
            np.zeros(capture_padding, dtype=np.complex64),
        )
    )

    preamble_end = find_payload_start(
        received=rx_capture,
        preamble=preamble,
    )
    payload_start = (
        preamble_end
        + config.cyclic_prefix_samples
    )
    rx_payload = rx_capture[
        payload_start:
        payload_start + config.num_grid_symbols
    ]

    rx_tf_grid = wigner(
        rx_payload,
        num_subcarriers=config.num_delay_bins,
        num_time_slots=config.num_doppler_bins,
    )
    rx_dd_grid = sfft(rx_tf_grid)
    rx_symbols = rx_dd_grid[config.data_mask]
    rx_bits = qam_demodulate(
        rx_symbols,
        order=config.qam_order,
    )
    ber = bit_error_rate(
        transmitted_bits=transmitted.bits,
        received_bits=rx_bits,
    )

    print("\n========== OTFS Offline Loopback ==========")
    print(f"Bandwidth          : {config.bandwidth_mhz:.1f} MHz")
    print(f"Sample Rate        : {config.sample_rate:.0f} samples/s")
    print(
        f"DD Grid            : "
        f"{config.num_delay_bins} x {config.num_doppler_bins}"
    )
    print(f"OTFS Payload       : {len(transmitted.waveform)} samples")
    print(f"Preamble           : {len(preamble)} samples")
    print(f"Cyclic Prefix      : {config.cyclic_prefix_samples} samples")
    print(
        f"Time Guard         : "
        f"{config.time_guard_samples} samples per side"
    )
    print(f"Transmitted Frame  : {len(tx_frame)} samples")
    print(f"Receiver Capture   : {len(rx_capture)} samples")
    print(f"Payload Start      : {payload_start}")
    print(f"Bit Error Rate     : {ber:.6f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
