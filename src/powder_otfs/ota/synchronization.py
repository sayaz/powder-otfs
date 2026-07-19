import numpy as np


def normalized_correlation_metric(
    received: np.ndarray,
    preamble: np.ndarray,
) -> np.ndarray:
    """Calculate the normalized preamble-correlation metric."""

    if len(preamble) == 0:
        raise ValueError("preamble must not be empty.")

    if len(received) < len(preamble):
        raise ValueError(
            "received must be at least as long as preamble."
        )

    preamble_energy = float(
        np.sum(np.abs(preamble) ** 2)
    )

    if preamble_energy == 0.0:
        raise ValueError(
            "preamble must contain nonzero samples."
        )

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

    return (
        np.abs(correlation) ** 2
        / np.maximum(
            preamble_energy * window_energy,
            1e-12,
        )
    )


def find_payload_start(
    received: np.ndarray,
    preamble: np.ndarray,
) -> int:
    """Locate the strongest preamble and return its payload start."""

    metric = normalized_correlation_metric(
        received=received,
        preamble=preamble,
    )

    preamble_start = int(np.argmax(metric))

    return preamble_start + len(preamble)


def find_payload_starts(
    received: np.ndarray,
    preamble: np.ndarray,
    threshold: float,
    minimum_separation: int,
) -> np.ndarray:
    """Locate all valid preambles and return their payload starts."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "threshold must be greater than 0 and at most 1."
        )

    if minimum_separation <= 0:
        raise ValueError(
            "minimum_separation must be positive."
        )

    metric = normalized_correlation_metric(
        received=received,
        preamble=preamble,
    )

    candidates = np.flatnonzero(
        metric >= threshold
    )

    if len(candidates) == 0:
        return np.empty(0, dtype=int)

    selected_preambles: list[int] = []
    group_start = 0

    for index in range(1, len(candidates)):
        separation = (
            candidates[index]
            - candidates[index - 1]
        )

        if separation >= minimum_separation:
            group = candidates[group_start:index]
            best = group[
                np.argmax(metric[group])
            ]
            selected_preambles.append(int(best))
            group_start = index

    final_group = candidates[group_start:]
    final_best = final_group[
        np.argmax(metric[final_group])
    ]
    selected_preambles.append(
        int(final_best)
    )

    return (
        np.asarray(selected_preambles, dtype=int)
        + len(preamble)
    )