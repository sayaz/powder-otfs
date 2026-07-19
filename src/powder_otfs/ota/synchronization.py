import numpy as np


def find_payload_start(
    received: np.ndarray,
    preamble: np.ndarray,
) -> int:
    """Locate a preamble and return the payload start index."""

    if len(preamble) == 0:
        raise ValueError("preamble must not be empty.")

    if len(received) < len(preamble):
        raise ValueError(
            "received must be at least as long as preamble."
        )

    preamble_energy = float(
        np.sum(np.abs(preamble) ** 2)
    )

    if preamble_energy == 0:
        raise ValueError("preamble must contain nonzero samples.")

    correlation = np.correlate(
        received,
        preamble,
        mode="valid",
    )

    window_energy = np.convolve(
        np.abs(received) ** 2,
        np.ones(len(preamble)),
        mode="valid",
    )

    metric = (
        np.abs(correlation) ** 2
        / np.maximum(
            preamble_energy * window_energy,
            1e-12,
        )
    )

    preamble_start = int(np.argmax(metric))

    return preamble_start + len(preamble)