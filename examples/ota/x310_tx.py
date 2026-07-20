import numpy as np

from powder_otfs.ota.config import OTFSOTAConfig
from powder_otfs.ota.framing import (
    build_ota_frame,
    create_preamble,
    normalize_waveform,
)
from powder_otfs.ota.payload import create_otfs_payload
from powder_otfs.ota.x310 import (
    configure_x310_tx,
    transmit_waveform,
)


def main() -> None:
    device_address = "192.168.40.2"
    center_frequency = 3.5e9
    tx_gain = 0.0
    channel = 0
    antenna = "TX/RX"
    peak_amplitude = 0.5
    repeat_count = 5000

    config = OTFSOTAConfig()
    payload = create_otfs_payload(
        config
    )
    preamble = create_preamble(
        half_length=config.preamble_half_length,
        seed=config.random_seed,
    )

    tx_frame = build_ota_frame(
        payload=payload.waveform,
        preamble=preamble,
        guard_samples=config.time_guard_samples,
        cyclic_prefix_samples=(
            config.cyclic_prefix_samples
        ),
    )
    tx_frame = normalize_waveform(
        tx_frame,
        peak_amplitude=peak_amplitude,
    )
    repeated_waveform = np.tile(
        tx_frame,
        repeat_count,
    ).astype(np.complex64)

    print(
        "\n========== X310 OTFS Transmitter =========="
    )
    print(f"Device Address     : {device_address}")
    print(f"Center Frequency   : {center_frequency / 1e9:.3f} GHz")
    print(f"Sample Rate        : {config.sample_rate:.0f} samples/s")
    print(f"TX Gain            : {tx_gain:.1f} dB")
    print(f"Modulation         : {config.qam_order}-QAM")
    print(
        f"DD Grid            : "
        f"{config.num_delay_bins} x "
        f"{config.num_doppler_bins}"
    )
    print(f"Data Symbols       : {config.num_data_symbols}")
    print(f"Bits per Frame     : {config.bits_per_frame}")
    print(f"Pilot Position     : {config.pilot_position}")
    print(f"Pilot Value        : {config.pilot_value}")
    print(
        f"DD Guard Size      : "
        f"{2 * config.guard_delay + 1} x "
        f"{2 * config.guard_doppler + 1}"
    )
    print(
        f"Cyclic Prefix      : "
        f"{config.cyclic_prefix_samples} samples"
    )
    print(f"Frame Length       : {len(tx_frame)} samples")
    print(f"Repeated Frames    : {repeat_count}")
    print(f"Total Samples      : {len(repeated_waveform)}")
    print(
        "===========================================\n"
    )

    usrp = configure_x310_tx(
        device_address=device_address,
        sample_rate=config.sample_rate,
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
