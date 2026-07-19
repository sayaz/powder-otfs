import numpy as np

from powder_otfs.channel.channel import apply_channel
from powder_otfs.channel.path import ChannelPath
from powder_otfs.equalization.zf import zero_forcing_equalizer
from powder_otfs.estimation.pilot import pilot_channel_estimate
from powder_otfs.metrics.ber import bit_error_rate
from powder_otfs.modulation.qam import qam_demodulate, qam_modulate
from powder_otfs.otfs.grid import insert_pilot_and_guards
from powder_otfs.otfs.transforms import heisenberg, isfft, sfft, wigner


def main() -> None:
    num_delay_bins = 32
    num_doppler_bins = 16
    qam_order = 4

    sample_rate = 1e6
    snr_db = 30.0

    pilot_position = (
        num_delay_bins // 2,
        num_doppler_bins // 2,
    )
    pilot_value = 2.0 + 0.0j
    guard_delay = 1
    guard_doppler = 1
    estimation_threshold = 0.1

    paths = [
        ChannelPath(
            delay_samples=0,
            doppler_hz=0.0,
            gain=0.8 + 0.2j,
        ),
    ]

    data_mask = np.ones(
        (num_delay_bins, num_doppler_bins),
        dtype=bool,
    )

    pilot_delay, pilot_doppler = pilot_position

    delay_start = pilot_delay - guard_delay
    delay_stop = pilot_delay + guard_delay + 1
    doppler_start = pilot_doppler - guard_doppler
    doppler_stop = pilot_doppler + guard_doppler + 1

    data_mask[
        delay_start:delay_stop,
        doppler_start:doppler_stop,
    ] = False

    bits_per_symbol = int(np.log2(qam_order))
    num_data_symbols = int(np.count_nonzero(data_mask))
    num_bits = bits_per_symbol * num_data_symbols

    print("\n================ OTFS Simulation ================")
    print(f"Modulation        : {qam_order}-QAM")
    print(f"DD Grid Size      : {num_delay_bins} x {num_doppler_bins}")
    print(f"Data Symbols      : {num_data_symbols}")
    print(f"Total Bits        : {num_bits}")
    print(f"Pilot Position    : {pilot_position}")
    print(f"Pilot Value       : {pilot_value}")
    print(f"Guard Size        : {2 * guard_delay + 1} x {2 * guard_doppler + 1}")
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

    tx_dd_grid = np.zeros(
        (num_delay_bins, num_doppler_bins),
        dtype=np.complex128,
    )
    tx_dd_grid[data_mask] = tx_symbols

    tx_dd_grid = insert_pilot_and_guards(
        grid=tx_dd_grid,
        pilot_delay_bin=pilot_delay,
        pilot_doppler_bin=pilot_doppler,
        pilot_amplitude=float(np.abs(pilot_value)),
        guard_delay=guard_delay,
        guard_doppler=guard_doppler,
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

    pilot_observation = np.zeros_like(rx_dd_grid)
    pilot_observation[
        delay_start:delay_stop,
        doppler_start:doppler_stop,
    ] = rx_dd_grid[
        delay_start:delay_stop,
        doppler_start:doppler_stop,
    ]

    estimate = pilot_channel_estimate(
        received_pilot_grid=pilot_observation,
        pilot_position=pilot_position,
        pilot_value=pilot_value,
        noise_variance=channel.noise_variance,
        threshold=estimation_threshold,
    )

    equalized = zero_forcing_equalizer(
        received_grid=rx_dd_grid,
        estimate=estimate,
    )

    rx_symbols = equalized.symbols[data_mask]

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