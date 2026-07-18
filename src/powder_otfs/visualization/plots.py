import matplotlib.pyplot as plt
import numpy as np

def plot_delay_doppler(
    grid: np.ndarray,
    title: str = "Delay-Doppler Grid",
) -> None:
    """Display the magnitue of the delay-Doppler grid."""

    plt.figure(figsize=(7,5))
    
    plt.imshow(
        np.abs(grid),
        origin = "lower",
        aspect = "auto",
        cmap = "viridis",
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
    """Display the magnitude of the time-frequency grid."""

    plt.figure(figsize=(7,5))
    
    plt.imshow(
        np.abs(tf_grid),
        origin = "lower",
        aspect = "auto",
        cmap = "viridis",
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
    """Display the real and imag parts of the time-domain waveform."""

    plt.figure(figsize=(10, 4))

    plt.plot(np.real(waveform), label="Real")
    plt.plot(np.imag(waveform), label="Imag", alpha=0.8)

    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_otfs_pipeline(
    tx_dd_grid: np.ndarray,
    tx_tf_grid: np.ndarray,
    tx_waveform: np.ndarray,
    rx_dd_grid: np.ndarray,
    rx_tf_grid: np.ndarray,
    rx_waveform: np.ndarray,
) -> None:
    """Display transmitter and receiver OTFS representations."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )

    # -------------------- Transmitter --------------------

    tx_dd = axes[0, 0].imshow(
        np.abs(tx_dd_grid),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=np.max(np.abs(tx_dd_grid)),
    )
    axes[0, 0].set_title("Tx Delay-Doppler")
    axes[0, 0].set_xlabel("Doppler Bin")
    axes[0, 0].set_ylabel("Delay Bin")
    fig.colorbar(tx_dd, ax=axes[0, 0])

    tx_tf = axes[0, 1].imshow(
        np.abs(tx_tf_grid),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[0, 1].set_title("Tx Time-Frequency")
    axes[0, 1].set_xlabel("Time Slot")
    axes[0, 1].set_ylabel("Subcarrier")
    fig.colorbar(tx_tf, ax=axes[0, 1])

    axes[0, 2].plot(np.real(tx_waveform), label="Real")
    axes[0, 2].plot(np.imag(tx_waveform), label="Imag")
    axes[0, 2].set_title("Tx Waveform")
    axes[0, 2].grid(True)
    axes[0, 2].legend()

    # -------------------- Receiver --------------------

    rx_dd = axes[1, 0].imshow(
        np.abs(rx_dd_grid),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=np.max(np.abs(tx_dd_grid)),
    )
    axes[1, 0].set_title("Rx Delay-Doppler")
    axes[1, 0].set_xlabel("Doppler Bin")
    axes[1, 0].set_ylabel("Delay Bin")
    fig.colorbar(rx_dd, ax=axes[1, 0])

    rx_tf = axes[1, 1].imshow(
        np.abs(rx_tf_grid),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[1, 1].set_title("Rx Time-Frequency")
    axes[1, 1].set_xlabel("Time Slot")
    axes[1, 1].set_ylabel("Subcarrier")
    fig.colorbar(rx_tf, ax=axes[1, 1])

    axes[1, 2].plot(np.real(rx_waveform), label="Real")
    axes[1, 2].plot(np.imag(rx_waveform), label="Imag")
    axes[1, 2].set_title("Rx Waveform")
    axes[1, 2].grid(True)
    axes[1, 2].legend()

    fig.suptitle("OTFS Transmitter / Receiver Pipeline")

    plt.tight_layout()
    plt.show()



