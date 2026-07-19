import numpy as np

from powder_otfs.modulation.qam import qam_modulate
from powder_otfs.ota.framing import (
    build_ota_frame,
    create_preamble,
    normalize_waveform,
)
from powder_otfs.ota.x310 import (
    configure_x310_tx,
    transmit_waveform,
)
from powder_otfs.otfs.transforms import heisenberg, isfft


def main() -> None:
    device_address = "192.168.40.2"
    sample_rate = 1e6
    center_frequency = 3.5e9
    tx_gain = 0.0
    channel = 0
    antenna = "TX/RX"

    num_delay_bins = 32
    num_doppler_bins = 16
    qam_order = 4
    guard_samples = 128
    preamble_half_length = 64
    peak_amplitude = 0.5
    repeat_count = 5000
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

    repeated_waveform = np.tile(
        tx_frame,
        repeat_count,
    ).astype(np.complex64)

    print("\n========== X310 OTFS Transmitter ==========")
    print(f"Device Address     : {device_address}")
    print(f"Center Frequency   : {center_frequency / 1e9:.3f} GHz")
    print(f"Sample Rate        : {sample_rate:.0f} samples/s")
    print(f"TX Gain            : {tx_gain:.1f} dB")
    print(f"Frame Length       : {len(tx_frame)} samples")
    print(f"Repeated Frames    : {repeat_count}")
    print(f"Total Samples      : {len(repeated_waveform)}")
    print("===========================================\n")

    usrp = configure_x310_tx(
        device_address=device_address,
        sample_rate=sample_rate,
        center_frequency=center_frequency,
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