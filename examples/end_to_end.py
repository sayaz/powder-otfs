import numpy as np

from powder_otfs.channel.channel import apply_channel
from powder_otfs.channel.path import ChannelPath
from powder_otfs.equalization.zf import zero_forcing_equalizer
from powder_otfs.estimation.perfect import perfect_channel_estimate
from powder_otfs.metrics.ber import bit_error_rate
from powder_otfs.modulation.qam import qam_demodulate, qam_modulate
from powder_otfs.otfs.transforms import heisenberg, isfft, sfft, wigner


def main() -> None:
    num_delay_bins = 32
    num_doppler_bins = 16
    qam_order = 4
    sample_rate = 1e6
    snr_db = 30.0

    bits_per_symbol = int(np.log2(qam_order))
    num_bits = (
        bits_per_symbol
        * num_delay_bins
        * num_doppler_bins
    )

    paths = [
        ChannelPath(
            delay_samples=0,
            doppler_hz=1000.0,
            gain=1.0 + 0.0j,
        ),
    ]

    print("\n================ OTFS Simulation ================")
    print(f"Modulation        : {qam_order}-QAM")
    print(f"DD Grid Size      : {num_delay_bins} x {num_doppler_bins}")
    print(f"Total Bits        : {num_bits}")
    print(f"Sample Rate       : {sample_rate:.0f} Hz")
    print(f"SNR               : {snr_db:.1f} dB")
    print(f"Channel Paths     : {len(paths)}")

    for index, path in enumerate(paths, start=1):
        print(
            f"  Path {index}: "
            f"Delay={path.delay_samples} samples, "
            f"Doppler={path.doppler_hz} Hz, "
            f"Gain={path.gain}"
        )

    print("=================================================\n")

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

    tx_dd_grid = tx_symbols.reshape(
        num_delay_bins,
        num_doppler_bins,
    )

    tx_tf_grid = isfft(tx_dd_grid)
    tx_waveform = heisenberg(tx_tf_grid)

    channel = apply_channel(
        tx_waveform,
        paths=paths,
        sample_rate=sample_rate,
        snr_db=snr_db,
    )

    rx_tf_grid = wigner(
        channel.waveform,
        num_subcarriers=num_delay_bins,
        num_time_slots=num_doppler_bins,
    )
    rx_dd_grid = sfft(rx_tf_grid)

    estimate = perfect_channel_estimate(
        channel=channel,
        grid_shape=rx_dd_grid.shape,
    )

    equalized = zero_forcing_equalizer(
        received_grid=rx_dd_grid,
        estimate=estimate,
    )

    rx_bits = qam_demodulate(
        equalized.symbols.reshape(-1),
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