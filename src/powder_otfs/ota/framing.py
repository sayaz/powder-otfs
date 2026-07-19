import numpy as np


def create_preamble(
    half_length: int = 64,
    seed: int = 12345,
) -> np.ndarray:
    """Create a repeatable synchronization preamble."""

    if half_length <= 0:
        raise ValueError("half_length must be positive.")

    rng = np.random.default_rng(seed)

    half = (
        2 * rng.integers(0, 2, half_length) - 1
    ).astype(np.complex64)

    return np.concatenate((half, half))


def normalize_waveform(
    waveform: np.ndarray,
    peak_amplitude: float = 0.8,
) -> np.ndarray:
    """Scale a waveform to a selected peak amplitude."""

    if not 0.0 < peak_amplitude <= 1.0:
        raise ValueError(
            "peak_amplitude must be greater than 0 and at most 1."
        )

    maximum = float(np.max(np.abs(waveform)))

    if maximum == 0.0:
        return waveform.astype(np.complex64)

    normalized = waveform * (
        peak_amplitude / maximum
    )

    return normalized.astype(np.complex64)


def build_ota_frame(
    payload: np.ndarray,
    preamble: np.ndarray,
    guard_samples: int = 128,
) -> np.ndarray:
    """Build one OTA frame containing guards, preamble, and payload."""

    if guard_samples < 0:
        raise ValueError("guard_samples must be non-negative.")

    guard = np.zeros(
        guard_samples,
        dtype=np.complex64,
    )

    return np.concatenate(
        (
            guard,
            preamble.astype(np.complex64),
            payload.astype(np.complex64),
            guard,
        )
    )