"""Plot average OTA BER versus USRP transmit gain."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator


def plot_ber_vs_tx_gain(
    tx_gain_db: list[float],
    ber: list[float],
) -> None:
    """Plot BER against transmit gain using a logarithmic BER axis."""

    if len(tx_gain_db) != len(ber):
        raise ValueError("tx_gain_db and ber must have the same length.")
    if not tx_gain_db:
        raise ValueError("At least one measurement is required.")
    if any(value < 0.0 or value > 1.0 for value in ber):
        raise ValueError("BER values must be between 0 and 1.")

    gain_values = np.asarray(tx_gain_db, dtype=float)
    ber_values = np.asarray(ber, dtype=float)

    positive_ber = ber_values[ber_values > 0.0]
    if positive_ber.size == 0:
        raise ValueError("At least one BER value must be greater than zero.")

    # Zero observed errors cannot be drawn on a logarithmic axis. Display them
    # one decade below the smallest positive BER measurement.
    plot_floor = positive_ber.min() / 10.0
    plotted_ber = np.where(ber_values > 0.0, ber_values, plot_floor)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.semilogy(
        gain_values,
        plotted_ber,
        marker="o",
        linewidth=2,
        markersize=7,
        label="Average BER",
    )

    axis.set_xlabel("TX Gain (dB)", fontsize=16)
    axis.set_ylabel("Bit Error Rate (BER)", fontsize=16)
    axis.set_title(
        "OTA BER vs. X310 Transmit Gain",
        fontsize=20,
    )
    axis.set_xticks(gain_values)
    axis.set_ylim(
        10.0 ** np.floor(np.log10(positive_ber.min())),
        10.0 ** np.ceil(np.log10(positive_ber.max())),
    )
    axis.yaxis.set_major_locator(LogLocator(base=10.0))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axis.tick_params(axis="both", which="major", labelsize=14)
    axis.grid(True, which="both", linestyle="--", alpha=0.5)
    axis.legend(fontsize=14)

    for gain, displayed, measured in zip(
        gain_values,
        plotted_ber,
        ber_values,
        strict=True,
    ):
        label = "0 observed" if measured == 0.0 else f"{measured:.2e}"
        axis.annotate(
            label,
            (gain, displayed),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=13,
        )

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Enter the measured TX gains and corresponding average BER values here.
    TX_GAIN_DB = [0, 5, 10, 15, 20]
    BER = [0.058404, 0.002807, 0.000244, 0.000074, 0.000035]

    plot_ber_vs_tx_gain(TX_GAIN_DB, BER)
