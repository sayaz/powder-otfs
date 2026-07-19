from pathlib import Path

import numpy as np

from powder_otfs.modulation.qam import (
    qam_demodulate,
    qam_modulate,
)
from powder_otfs.ota.framing import create_preamble
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
    sample_rate = 1e6
    center_frequency = 3.5e9
    rx_gain = 20.0
    channel = 0
    antenna = "RX2"
    capture_samples = 6_000_000
    save_received_samples = True
    received_samples_path = Path("results/rx_samples.npy")

    num_delay_bins = 32
    num_doppler_bins = 16
    qam_order = 4

    guard_samples = 128
    preamble_half_length = 64
    synchronization_threshold = 0.50
    random_seed = 12345

    rng = np.random.default_rng(random_seed)

    num_symbols = (
        num_delay_bins
        * num_doppler_bins
    )
    bits_per_symbol = int(
        np.log2(qam_order)
    )
    bits_per_frame = (
        num_symbols
        * bits_per_symbol
    )
    payload_samples = num_symbols

    transmitted_bits = rng.integers(
        0,
        2,
        bits_per_frame,
        dtype=np.uint8,
    )

    transmitted_symbols = qam_modulate(
        transmitted_bits,
        order=qam_order,
    )

    preamble = create_preamble(
        half_length=preamble_half_length,
        seed=random_seed,
    )

    frame_length = (
        guard_samples
        + len(preamble)
        + payload_samples
        + guard_samples
    )

    minimum_separation = (
        frame_length // 2
    )

    print(
        "\n========== X310 OTFS Receiver Configuration =========="
    )
    print(f"Device Address       : {device_address}")
    print(f"Center Frequency     : {center_frequency / 1e9:.3f} GHz")
    print(f"Sample Rate          : {sample_rate:.0f} samples/s")
    print(f"RX Gain              : {rx_gain:.1f} dB")
    print(f"Channel              : {channel}")
    print(f"RX Antenna           : {antenna}")
    print(f"Capture Samples      : {capture_samples}")
    print(f"Capture Duration     : {capture_samples / sample_rate:.3f} s")
    print(f"Save IQ Samples      : {save_received_samples}")
    if save_received_samples:
        print(f"IQ Output File       : {received_samples_path}")
    print(f"Modulation           : {qam_order}-QAM")
    print(f"DD Grid              : {num_delay_bins} x {num_doppler_bins}")
    print(f"Payload              : {payload_samples} samples")
    print(f"Bits per Frame       : {bits_per_frame}")
    print(f"Preamble             : {len(preamble)} samples")
    print(f"Guard                : {guard_samples} samples per side")
    print(f"Complete Frame       : {frame_length} samples")
    print(f"Sync Threshold       : {synchronization_threshold:.2f}")
    print(
        "=======================================================\n"
    )
    print("Waiting for samples...")

    usrp = configure_x310_rx(
        device_address=device_address,
        sample_rate=sample_rate,
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

    payload_starts = find_payload_starts(
        received=received,
        preamble=preamble,
        threshold=synchronization_threshold,
        minimum_separation=minimum_separation,
    )

    total_bit_errors = 0
    processed_frames = 0
    rejected_frames = 0

    frame_bers: list[float] = []
    symbol_mses: list[float] = []
    channel_gains: list[complex] = []

    for payload_start in payload_starts:
        preamble_start = (
            payload_start
            - len(preamble)
        )
        payload_end = (
            payload_start
            + payload_samples
        )

        if (
            preamble_start < 0
            or payload_end > len(received)
        ):
            rejected_frames += 1
            continue

        received_preamble = received[
            preamble_start:payload_start
        ]

        channel_gain = np.vdot(
            preamble,
            received_preamble,
        ) / np.vdot(
            preamble,
            preamble,
        )

        if abs(channel_gain) < 1e-12:
            rejected_frames += 1
            continue

        received_payload = received[
            payload_start:payload_end
        ]

        corrected_payload = (
            received_payload
            / channel_gain
        )

        received_tf_grid = wigner(
            corrected_payload,
            num_subcarriers=num_delay_bins,
            num_time_slots=num_doppler_bins,
        )

        received_dd_grid = sfft(
            received_tf_grid
        )

        received_symbols = (
            received_dd_grid.reshape(-1)
        )

        received_bits = qam_demodulate(
            received_symbols,
            order=qam_order,
        )

        bit_errors = int(
            np.count_nonzero(
                transmitted_bits
                != received_bits
            )
        )

        frame_ber = (
            bit_errors
            / bits_per_frame
        )

        symbol_mse = float(
            np.mean(
                np.abs(
                    received_symbols
                    - transmitted_symbols
                ) ** 2
            )
        )

        total_bit_errors += bit_errors
        processed_frames += 1

        frame_bers.append(frame_ber)
        symbol_mses.append(symbol_mse)
        channel_gains.append(channel_gain)

    if processed_frames == 0:
        raise RuntimeError(
            "No complete valid OTFS frames were decoded."
        )

    processed_bits = (
        processed_frames
        * bits_per_frame
    )

    aggregate_ber = (
        total_bit_errors
        / processed_bits
    )

    gain_magnitudes = np.abs(
        np.asarray(channel_gains)
    )

    print(
        "\n========== X310 OTFS Multi-Frame Result =========="
    )
    print(f"Detected Frames       : {len(payload_starts)}")
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
    print(
        "===================================================\n"
    )


if __name__ == "__main__":
    main()
