import numpy as np

from powder_otfs.metrics.ber import bit_error_rate
from powder_otfs.modulation.qam import (
    qam_demodulate,
    qam_modulate,
)
from powder_otfs.ota.framing import (
    build_ota_frame,
    create_preamble,
    normalize_waveform,
)
from powder_otfs.ota.synchronization import find_payload_start
from powder_otfs.otfs.transforms import (
    heisenberg,
    isfft,
    sfft,
    wigner,
)


def main() -> None:
    num_delay_bins = 32
    num_doppler_bins = 16
    qam_order = 4
    guard_samples = 128
    preamble_half_length = 64
    peak_amplitude = 0.8
    random_seed = 12345

    rng = np.random.default_rng(random_seed)

    num_symbols = num_delay_bins * num_doppler_bins
    bits_per_symbol = int(np.log2(qam_order))
    num_bits = num_symbols * bits_per_symbol

    bits = rng.integers(
        0,
        2,
        num_bits,
        dtype=np.uint8,
    )

    tx_symbols = qam_modulate(
        bits,
        order=qam_order,
    )

    tx_dd_grid = tx_symbols.reshape(
        num_delay_bins,
        num_doppler_bins,
    )

    tx_tf_grid = isfft(tx_dd_grid)
    tx_payload = heisenberg(tx_tf_grid)

    preamble = create_preamble(
        half_length=preamble_half_length,
        seed=random_seed,
    )

    tx_frame = build_ota_frame(
        payload=tx_payload,
        preamble=preamble,
        guard_samples=guard_samples,
    )

    tx_frame = normalize_waveform(
        tx_frame,
        peak_amplitude=peak_amplitude,
    )

    capture_leading_samples = 300
    capture_trailing_samples = 300

    rx_capture = np.concatenate(
        (
            np.zeros(
                capture_leading_samples,
                dtype=np.complex64,
            ),
            tx_frame,
            np.zeros(
                capture_trailing_samples,
                dtype=np.complex64,
            ),
        )
    )

    payload_start = find_payload_start(
        received=rx_capture,
        preamble=preamble,
    )

    rx_payload = rx_capture[
        payload_start:
        payload_start + len(tx_payload)
    ]

    rx_tf_grid = wigner(
        rx_payload,
        num_subcarriers=num_delay_bins,
        num_time_slots=num_doppler_bins,
    )

    rx_dd_grid = sfft(rx_tf_grid)
    rx_symbols = rx_dd_grid.reshape(-1)

    rx_bits = qam_demodulate(
        rx_symbols,
        order=qam_order,
    )

    ber = bit_error_rate(
        transmitted_bits=bits,
        received_bits=rx_bits,
    )

    print("\n========== OTFS Offline Loopback ==========")
    print(f"DD Grid            : {num_delay_bins} x {num_doppler_bins}")
    print(f"OTFS Payload       : {len(tx_payload)} samples")
    print(f"Preamble           : {len(preamble)} samples")
    print(f"Guard              : {guard_samples} samples per side")
    print(f"Transmitted Frame  : {len(tx_frame)} samples")
    print(f"Receiver Capture   : {len(rx_capture)} samples")
    print(f"Payload Start      : {payload_start}")
    print(f"Bit Error Rate     : {ber:.6f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()