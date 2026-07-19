import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_delay_doppler(
    grid: np.ndarray,
    title: str = "Delay-Doppler Grid",
) -> None:
    """Display the magnitude of a delay-Doppler grid."""

    plt.figure(figsize=(7, 5))
    plt.imshow(
        np.abs(grid),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
    )
    plt.colorbar(label="Magnitude")
    plt.xlabel("Doppler Bin")
    plt.ylabel("Delay Bin")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_time_frequency(
    tf_grid: np.ndarray,
    title: str = "Time-Frequency Grid",
) -> None:
    """Display the magnitude of a time-frequency grid."""

    plt.figure(figsize=(7, 5))
    plt.imshow(
        np.abs(tf_grid),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
    )
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time Slot")
    plt.ylabel("Subcarrier")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_waveform(
    waveform: np.ndarray,
    title: str = "Time-Domain Waveform",
) -> None:
    """Display the real and imaginary parts of a waveform."""

    plt.figure(figsize=(10, 4))
    plt.plot(np.real(waveform), label="Real")
    plt.plot(np.imag(waveform), label="Imaginary", alpha=0.8)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_otfs_debug_view(
    tx_dd_grid: np.ndarray,
    rx_dd_grid: np.ndarray,
    equalized_dd_grid: np.ndarray,
    pilot_observation: np.ndarray,
    data_mask: np.ndarray,
    pilot_position: tuple[int, int],
) -> None:
    """Display the main OTFS channel and equalization diagnostics."""

    expected_shape = tx_dd_grid.shape
    grids = (
        rx_dd_grid,
        equalized_dd_grid,
        pilot_observation,
        data_mask,
    )

    if any(grid.shape != expected_shape for grid in grids):
        raise ValueError("All OTFS grids and masks must have the same shape.")

    category_grid = np.zeros(
        expected_shape,
        dtype=np.uint8,
    )
    category_grid[data_mask] = 1
    category_grid[pilot_position] = 2

    category_cmap = ListedColormap(
        [
            "#303030",
            "#2a78b8",
            "#ffd43b",
        ]
    )

    dd_vmax = max(
        float(np.percentile(np.abs(rx_dd_grid[data_mask]), 99)),
        float(np.percentile(np.abs(equalized_dd_grid[data_mask]), 99)),
        1e-12,
    )

    channel_vmax = max(
        float(np.max(np.abs(pilot_observation))),
        1e-12,
    )

    data_error = np.abs(
        equalized_dd_grid - tx_dd_grid
    )
    error_db = np.full(
        expected_shape,
        np.nan,
        dtype=float,
    )
    error_db[data_mask] = 20.0 * np.log10(
        np.maximum(
            data_error[data_mask],
            1e-8,
        )
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 9),
        constrained_layout=True,
    )

    tx_categories = axes[0, 0].imshow(
        category_grid,
        origin="lower",
        aspect="auto",
        cmap=category_cmap,
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
    )
    category_colorbar = fig.colorbar(
        tx_categories,
        ax=axes[0, 0],
        ticks=[0, 1, 2],
    )
    category_colorbar.ax.set_yticklabels(
        ["Guard", "Data", "Pilot"]
    )

    rx_dd = axes[0, 1].imshow(
        np.abs(rx_dd_grid),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=dd_vmax,
    )
    fig.colorbar(
        rx_dd,
        ax=axes[0, 1],
        label="Magnitude",
    )

    equalized_dd = axes[0, 2].imshow(
        np.abs(equalized_dd_grid),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=dd_vmax,
    )
    fig.colorbar(
        equalized_dd,
        ax=axes[0, 2],
        label="Magnitude",
    )

    channel_taps = axes[1, 0].imshow(
        np.abs(pilot_observation),
        origin="lower",
        aspect="auto",
        cmap="plasma",
        vmin=0.0,
        vmax=channel_vmax,
    )
    fig.colorbar(
        channel_taps,
        ax=axes[1, 0],
        label="Pilot-response magnitude",
    )

    error_image = axes[1, 1].imshow(
        error_db,
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )
    fig.colorbar(
        error_image,
        ax=axes[1, 1],
        label="Error magnitude (dB)",
    )

    tx_data = tx_dd_grid[data_mask]
    rx_data = rx_dd_grid[data_mask]
    equalized_data = equalized_dd_grid[data_mask]

    axes[1, 2].scatter(
        rx_data.real,
        rx_data.imag,
        s=12,
        alpha=0.25,
        color="#f28e2b",
        label="Received",
    )
    axes[1, 2].scatter(
        equalized_data.real,
        equalized_data.imag,
        s=14,
        alpha=0.55,
        color="#59a14f",
        label="Equalized",
    )
    axes[1, 2].scatter(
        tx_data.real,
        tx_data.imag,
        s=65,
        marker="x",
        linewidths=1.8,
        color="#1f77b4",
        label="Transmitted",
    )
    axes[1, 2].axhline(
        0.0,
        color="gray",
        linewidth=0.6,
    )
    axes[1, 2].axvline(
        0.0,
        color="gray",
        linewidth=0.6,
    )
    axes[1, 2].set_aspect(
        "equal",
        adjustable="datalim",
    )
    axes[1, 2].grid(
        True,
        alpha=0.25,
    )
    axes[1, 2].legend(
        loc="best",
        fontsize=8,
    )

    titles = (
        "Tx DD Structure",
        "Rx DD Magnitude",
        "Equalized DD Magnitude",
        "Estimated Delay-Doppler Channel Taps",
        "Equalization Error",
        "Data-Symbol Constellation",
    )

    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)

    for axis in axes[0]:
        axis.set_xlabel("Doppler Bin")
        axis.set_ylabel("Delay Bin")

    axes[1, 0].set_xlabel("Doppler Bin")
    axes[1, 0].set_ylabel("Delay Bin")
    axes[1, 1].set_xlabel("Doppler Bin")
    axes[1, 1].set_ylabel("Delay Bin")
    axes[1, 2].set_xlabel("In-phase")
    axes[1, 2].set_ylabel("Quadrature")

    fig.suptitle(
        "OTFS Transmit, Channel, and Equalization Debug View",
        fontsize=14,
    )
    plt.show()
