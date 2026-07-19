import numpy as np

from powder_otfs.metrics.ber import bit_error_rate
from powder_otfs.modulation.qam import (
    qam_demodulate,
    qam_modulate,
)
from powder_otfs.ota.framing import create_preamble
from powder_otfs.ota.synchronization import find_payload_start
from powder_otfs.ota.x310 import (
    configure_x310,
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
    antenna = "TX/RX"
    capture_samples = 6_000_000

    num_delay_bins = 32
    num_doppler_bins = 16
    qam_order = 4
    preamble_half_length = 64
    random_seed = 12345

    rng = np.random.default_rng(random_seed)

    num_symbols = num_delay_bins * num_doppler_bins
    bits_per_symbol = int(np.log2(qam_order))
    num_bits = num_symbols * bits_per_symbol
    payload_samples = num_symbols

    transmitted_bits = rng.integers(
        0,
        2,
        num_bits,
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

    print("\n========== X310 OTFS Receiver ==========")
    print(f"Device Address     : {device_address}")
    print(f"Center Frequency   : {center_frequency / 1e9:.3f} GHz")
    print(f"Sample Rate        : {sample_rate:.0f} samples/s")
    print(f"RX Gain            : {rx_gain:.1f} dB")
    print(f"Capture Samples    : {capture_samples}")
    print("Waiting for samples...")
    print("========================================\n")

    usrp = configure_x310(
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

    payload_start = find_payload_start(
        received=received,
        preamble=preamble,
    )

    preamble_start = payload_start - len(preamble)

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
        raise RuntimeError(
            "Estimated channel gain is too small."
        )

    received_payload = received[
        payload_start:
        payload_start + payload_samples
    ]

    if len(received_payload) != payload_samples:
        raise RuntimeError(
            "Capture ended before the complete payload arrived."
        )

    corrected_payload = (
        received_payload / channel_gain
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

    ber = bit_error_rate(
        transmitted_bits=transmitted_bits,
        received_bits=received_bits,
    )

    symbol_error = np.mean(
        np.abs(
            received_symbols
            - transmitted_symbols
        ) ** 2
    )

    print("\n========== X310 OTFS Result ==========")
    print(f"Payload Start      : {payload_start}")
    print(f"Channel Gain       : {channel_gain}")
    print(f"Symbol MSE         : {symbol_error:.6e}")
    print(f"Bit Error Rate     : {ber:.6f}")
    print("======================================\n")


if __name__ == "__main__":
    main()