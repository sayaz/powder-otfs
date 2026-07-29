import argparse

import numpy as np

from powder_otfs.channel.channel import apply_channel
from powder_otfs.channel.fading import apply_fading
from powder_otfs.channel.path import ChannelPath
from powder_otfs.equalization.zf import zero_forcing_equalizer
from powder_otfs.equalization.mmse import mmse_equalizer
from powder_otfs.estimation.pilot import pilot_channel_estimate
from powder_otfs.modulation.qam import qam_demodulate, qam_modulate
from powder_otfs.otfs.grid import insert_pilot_and_guards
from powder_otfs.otfs.transforms import heisenberg, isfft, sfft, wigner
from powder_otfs.visualization.plots import (
    plot_channel_matrix_validation,
    plot_otfs_debug_view,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an OTFS simulation."
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=1,
        help="Number of predefined channel paths to use (1-10).",
    )
    parser.add_argument(
        "--bandwidth-mhz",
        type=float,
        choices=(1.0, 5.0, 10.0, 20.0),
        default=1.0,
        help=(
            "Baseband bandwidth in MHz. The simulation sample rate is "
            "set equal to this bandwidth (default: 1)."
        ),
    )
    parser.add_argument(
        "--delay-bins",
        type=int,
        default=32,
        help="Number of delay bins in the DD grid (default: 32).",
    )
    parser.add_argument(
        "--doppler-bins",
        type=int,
        default=16,
        help="Number of Doppler bins in the DD grid (default: 16).",
    )
    parser.add_argument(
        "--max-delay-samples",
        type=int,
        default=3,
        help="Maximum supported channel delay in samples (default: 3).",
    )
    parser.add_argument(
        "--max-doppler-bins",
        type=int,
        default=2,
        help="Maximum supported Doppler shift in bins (default: 2).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of OTFS frames to simulate (default: 100).",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=20.0,
        help="Channel SNR in dB (default: 20).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Simulation settings
    num_delay_bins = args.delay_bins
    num_doppler_bins = args.doppler_bins
    qam_order = 4

    bandwidth_hz = args.bandwidth_mhz * 1e6
    sample_rate = bandwidth_hz
    snr_db = args.snr_db
    num_frames = args.num_frames
    show_plots = True
    show_channel_matrix_validation = True
    fading_model = "rayleigh"
    rician_k = 5.0
    random_seed = 12345

    pilot_value = 4.0 + 0.0j
    threshold_factor = 5.0
    maximum_supported_delay = args.max_delay_samples
    maximum_supported_doppler = args.max_doppler_bins

    if num_delay_bins <= 0 or num_doppler_bins <= 0:
        raise ValueError("DD-grid dimensions must be positive.")
    if maximum_supported_delay < 0:
        raise ValueError("max_delay_samples must not be negative.")
    if maximum_supported_doppler < 0:
        raise ValueError("max_doppler_bins must not be negative.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    pilot_position = (
        num_delay_bins // 2,
        num_doppler_bins // 2,
    )

    doppler_resolution = (
        sample_rate
        / (num_delay_bins * num_doppler_bins)
    )

    equalizer_name = "mmse"
    channel_matrix_method = "circulant"
    rng = np.random.default_rng(random_seed)

    # delay samples, Doppler bins, complex gain
    path_definitions = [
        (0, 0, 1.00 + 0.00j),
        (1, 1, 0.70 + 0.20j),
        (2, -1, 0.50 - 0.15j),
        (3, 2, 0.40 + 0.10j),
        (1, -2, 0.35 - 0.20j),
        (4, 1, 0.30 + 0.05j),
        (2, 3, 0.25 - 0.10j),
        (5, -3, 0.22 + 0.08j),
        (3, -1, 0.20 - 0.05j),
        (4, 2, 0.18 + 0.04j),
    ]

    if not 1 <= args.num_paths <= len(path_definitions):
        raise ValueError(
            f"num_paths must be between 1 and "
            f"{len(path_definitions)}."
        )

    active_definitions = path_definitions[: args.num_paths]

    if any(
        delay_samples > maximum_supported_delay
        for delay_samples, _, _ in active_definitions
    ):
        raise ValueError(
            "An active channel path exceeds max_delay_samples."
        )
    if any(
        abs(doppler_bin) > maximum_supported_doppler
        for _, doppler_bin, _ in active_definitions
    ):
        raise ValueError(
            "An active channel path exceeds max_doppler_bins."
        )

    paths = [
        ChannelPath(
            delay_samples=delay_samples,
            doppler_hz=doppler_bin * doppler_resolution,
            gain=gain,
        )
        for delay_samples, doppler_bin, gain
        in active_definitions
    ]

    guard_delay = maximum_supported_delay
    guard_doppler = 2 * maximum_supported_doppler

    pilot_delay, pilot_doppler = pilot_position

    delay_start = pilot_delay - guard_delay
    delay_stop = pilot_delay + guard_delay + 1

    doppler_start = pilot_doppler - guard_doppler
    doppler_stop = pilot_doppler + guard_doppler + 1

    if (
        delay_start < 0
        or delay_stop > num_delay_bins
        or doppler_start < 0
        or doppler_stop > num_doppler_bins
    ):
        raise ValueError(
            "The DD grid is too small for the required guard region."
        )

    data_mask = np.ones(
        (num_delay_bins, num_doppler_bins),
        dtype=bool,
    )

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
    print(f"Bits per Frame    : {num_bits}")
    print(f"Number of Frames  : {num_frames}")
    print(f"Total Bits        : {num_bits * num_frames}")
    print(f"Pilot Position    : {pilot_position}")
    print(f"Pilot Value       : {pilot_value}")
    print(f"Supported Delay   : 0 to {maximum_supported_delay} samples")
    print(
        f"Supported Doppler : "
        f"±{maximum_supported_doppler} bins"
    )
    print(
        f"Guard Size        : "
        f"{2 * guard_delay + 1} x "
        f"{2 * guard_doppler + 1}"
    )
    print(f"Doppler Resolution: {doppler_resolution:.3f} Hz")
    print(f"Bandwidth         : {bandwidth_hz / 1e6:.1f} MHz")
    print(f"Sample Rate       : {sample_rate:.0f} Hz")
    print(f"SNR               : {snr_db:.1f} dB")
    print(f"Fading Model      : {fading_model}")
    print(f"Channel Matrix    : {channel_matrix_method}")
    if fading_model.lower() == "rician":
        print(f"Rician K-factor   : {rician_k}")
    print(f"Channel Paths     : {len(paths)}")

    for index, (
        delay_samples,
        doppler_bin,
        gain,
    ) in enumerate(active_definitions, start=1):
        print(
            f"  Path {index}: "
            f"Delay={delay_samples} samples, "
            f"Delay Time={delay_samples / sample_rate * 1e6:.3f} us, "
            f"Doppler Bin={doppler_bin}, "
            f"Doppler={doppler_bin * doppler_resolution:.3f} Hz, "
            f"Gain={gain}"
        )

    print("=================================================\n")

    observation_delay_start = (
        pilot_delay
    )
    observation_delay_stop = (
        pilot_delay + maximum_supported_delay + 1
    )

    observation_doppler_start = (
        pilot_doppler - maximum_supported_doppler
    )
    observation_doppler_stop = (
        pilot_doppler + maximum_supported_doppler + 1
    )

    estimate = None
    equalized = None
    estimation_threshold = 0.0
    total_bit_errors = 0
    total_bits = 0
    debug_plot_data = None
    matrix_validation_data = None

    for frame_index in range(num_frames):
        bits = rng.integers(
            0,
            2,
            num_bits,
            dtype=np.uint8,
        )

        frame_paths = apply_fading(
            paths=paths,
            model=fading_model,
            rng=rng,
            rician_k=rician_k,
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
            paths=frame_paths,
            sample_rate=sample_rate,
            snr_db=snr_db,
        )

        rx_tf_grid = wigner(
            channel.waveform,
            num_subcarriers=num_delay_bins,
            num_time_slots=num_doppler_bins,
        )
        rx_dd_grid = sfft(rx_tf_grid)

        if (
            estimate is None
            or fading_model.lower() != "fixed"
        ):
            estimation_threshold = (
                threshold_factor
                * np.sqrt(channel.noise_variance)
            )

            pilot_observation = np.zeros_like(rx_dd_grid)
            pilot_observation[
                observation_delay_start:observation_delay_stop,
                observation_doppler_start:observation_doppler_stop,
            ] = rx_dd_grid[
                observation_delay_start:observation_delay_stop,
                observation_doppler_start:observation_doppler_stop,
            ]

            estimate = pilot_channel_estimate(
                received_pilot_grid=pilot_observation,
                pilot_position=pilot_position,
                pilot_value=pilot_value,
                sample_rate=sample_rate,
                noise_variance=channel.noise_variance,
                threshold=estimation_threshold,
                matrix_method=channel_matrix_method,
            )

            if (
                show_channel_matrix_validation
                and frame_index == 0
            ):
                basis_estimate = pilot_channel_estimate(
                    received_pilot_grid=pilot_observation,
                    pilot_position=pilot_position,
                    pilot_value=pilot_value,
                    sample_rate=sample_rate,
                    noise_variance=channel.noise_variance,
                    threshold=estimation_threshold,
                    matrix_method="basis",
                )
                matrix_validation_data = (
                    estimate.channel_response.copy(),
                    basis_estimate.channel_response.copy(),
                )

        if equalizer_name.lower() == "zf":
            equalized = zero_forcing_equalizer(
                received_grid=rx_dd_grid,
                estimate=estimate,
            )
        elif equalizer_name.lower() == "mmse":
            equalized = mmse_equalizer(
                received_grid=rx_dd_grid,
                estimate=estimate,
                symbol_energy=1.0,
            )
        else:
            raise ValueError(
                "equalizer_name must be 'zf' or 'mmse'."
            )

        if show_plots and frame_index == 0:
            debug_plot_data = (
                tx_dd_grid.copy(),
                rx_dd_grid.copy(),
                equalized.symbols.copy(),
                pilot_observation.copy(),
            )

        rx_symbols = equalized.symbols[data_mask]
        rx_bits = qam_demodulate(
            rx_symbols,
            order=qam_order,
        )

        total_bit_errors += int(
            np.count_nonzero(bits != rx_bits)
        )
        total_bits += len(bits)

        if (
            frame_index == 0
            or (frame_index + 1) % 10 == 0
            or frame_index + 1 == num_frames
        ):
            print(
                f"Completed frame "
                f"{frame_index + 1}/{num_frames}"
            )

    ber = total_bit_errors / total_bits

    print("Simulation Complete")
    print(f"Channel Estimator : {estimate.method}")
    print(f"Equalizer         : {equalized.method}")
    print(f"Channel Matrix    : {channel_matrix_method}")
    print(f"Threshold         : {estimation_threshold:.6f}")
    print(f"Bit Errors        : {total_bit_errors}")
    print(f"Processed Bits    : {total_bits}")
    print(f"Bit Error Rate    : {ber:.6f}")

    if show_plots and debug_plot_data is not None:
        plot_otfs_debug_view(
            tx_dd_grid=debug_plot_data[0],
            rx_dd_grid=debug_plot_data[1],
            equalized_dd_grid=debug_plot_data[2],
            pilot_observation=debug_plot_data[3],
            data_mask=data_mask,
            pilot_position=pilot_position,
        )

    if (
        show_channel_matrix_validation
        and matrix_validation_data is not None
    ):
        plot_channel_matrix_validation(
            circulant_matrix=matrix_validation_data[0],
            basis_matrix=matrix_validation_data[1],
        )


if __name__ == "__main__":
    main()
