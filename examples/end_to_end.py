import numpy as np

from powder_otfs.channel.channel import apply_channel
from powder_otfs.channel.path import ChannelPath
from powder_otfs.equalization.zf import zero_forcing_equalizer
from powder_otfs.estimation.perfect import perfect_channel_estimate
from powder_otfs.metrics.ber import bit_error_rate
from powder_otfs.modulation.qam import qam_demodulate, qam_modulate
from powder_otfs.otfs.transforms import (
    heisenberg,
    isfft,
    sfft,
    wigner,
)


def main():

    # ------------------------------------------------------------------
    # Simulation Parameters
    # ------------------------------------------------------------------
    num_subcarriers = 32
    num_time_slots = 16
    qam_order = 4

    sample_rate = 1e6
    snr_db = 30.0

    num_bits = (
        int(np.log2(qam_order))
        * num_subcarriers
        * num_time_slots
    )

    paths = [
        ChannelPath(
            delay_samples=0,
            doppler_hz=0.0,
            gain=1.0 + 0j,
        ),
    ]

    print("\n================ OTFS Simulation ================")
    print(f"Modulation        : {qam_order}-QAM")
    print(f"Grid Size         : {num_subcarriers} x {num_time_slots}")
    print(f"Total Bits        : {num_bits}")
    print(f"Sample Rate       : {sample_rate:.0f} Hz")
    print(f"SNR               : {snr_db:.1f} dB")
    print(f"Channel Paths     : {len(paths)}")

    for i, path in enumerate(paths, start=1):
        print(
            f"  Path {i}: "
            f"Delay={path.delay_samples} samples, "
            f"Doppler={path.doppler_hz} Hz, "
            f"Gain={path.gain}"
        )

    print("=================================================\n")

    # ------------------------------------------------------------------
    # Transmitter
    # ------------------------------------------------------------------
    bits = np.random.randint(
        0,
        2,
        num_bits,
        dtype=np.uint8,
    )

    tx_symbols = qam_modulate(
        bits,
        order=qam_order,
    )

    dd_grid = tx_symbols.reshape(
        num_subcarriers,
        num_time_slots,
    )

    tf_grid = isfft(dd_grid)

    tx_waveform = heisenberg(tf_grid)

    # ------------------------------------------------------------------
    # Channel
    # ------------------------------------------------------------------
    rx_waveform = apply_channel(
        tx_waveform,
        paths=paths,
        sample_rate=sample_rate,
        snr_db=snr_db,
    )

    # ------------------------------------------------------------------
    # Receiver
    # ------------------------------------------------------------------
    rx_tf_grid = wigner(
        rx_waveform,
        num_subcarriers=num_subcarriers,
        num_time_slots=num_time_slots,
    )

    rx_dd_grid = sfft(rx_tf_grid)

    #
    # Perfect CSI
    #
    channel_response = np.ones_like(rx_dd_grid)

    estimate = perfect_channel_estimate(
        channel_response=channel_response,
        noise_variance=0.0,
    )

    #
    # Zero-Forcing Equalization
    #
    equalized = zero_forcing_equalizer(
        rx_dd_grid,
        estimate,
    )

    rx_symbols = equalized.symbols.reshape(-1)

    rx_bits = qam_demodulate(
        rx_symbols,
        order=qam_order,
    )

    ber = bit_error_rate(
        bits,
        rx_bits,
    )

    print("Simulation Complete")
    print(f"Channel Estimator : {estimate.method}")
    print(f"Equalizer         : {equalized.method}")
    print(f"Bit Error Rate    : {ber:.6f}")


if __name__ == "__main__":
    main()