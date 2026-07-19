import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an X310 complex-IQ capture.",
    )
    parser.add_argument(
        "capture",
        help="Path to the rx_samples.npy capture.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First sample to display.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10_000,
        help="Number of samples to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    received = np.load(args.capture)
    stop = min(
        args.start + args.count,
        len(received),
    )
    samples = received[args.start:stop]
    sample_indices = np.arange(
        args.start,
        stop,
    )

    if len(samples) == 0:
        raise ValueError(
            "The selected sample range is empty."
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        sample_indices,
        samples.real,
        label="I",
        linewidth=0.8,
    )
    axes[0].plot(
        sample_indices,
        samples.imag,
        label="Q",
        linewidth=0.8,
        alpha=0.8,
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Received Complex-IQ Samples")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        sample_indices,
        np.abs(samples),
        color="#d55e00",
        linewidth=0.8,
    )
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("Received Signal Magnitude")
    axes[1].grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
