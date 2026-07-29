import numpy as np

from powder_otfs.ota.framing import TrainingPreamble


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


def repeated_symbol_correlation_metric(
    received: np.ndarray,
    symbol_length: int,
) -> np.ndarray:
    """Calculate a normalized sliding STF autocorrelation metric."""

    if symbol_length <= 0:
        raise ValueError("symbol_length must be positive.")
    if len(received) < 2 * symbol_length:
        raise ValueError(
            "received must contain at least two short symbols."
        )

    products = (
        np.conj(received[:-symbol_length])
        * received[symbol_length:]
    )
    def rolling_sum(values: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate(
            (
                np.zeros(1, dtype=values.dtype),
                np.cumsum(values),
            )
        )
        return (
            cumulative[symbol_length:]
            - cumulative[:-symbol_length]
        )

    correlation = rolling_sum(products)
    first_energy = rolling_sum(
        np.abs(received[:-symbol_length]) ** 2
    )
    second_energy = rolling_sum(
        np.abs(received[symbol_length:]) ** 2
    )

    return (
        np.abs(correlation) ** 2
        / np.maximum(first_energy * second_energy, 1e-12)
    )


def find_training_preamble_starts(
    received: np.ndarray,
    preamble: TrainingPreamble,
    stf_threshold: float,
    ltf_threshold: float,
    minimum_separation: int,
) -> np.ndarray:
    """Detect frames using an STF plateau followed by an LTF peak."""

    if not 0.0 < stf_threshold <= 1.0:
        raise ValueError("stf_threshold must be in (0, 1].")

    stf_metric = repeated_symbol_correlation_metric(
        received,
        symbol_length=len(preamble.stf_short_symbol),
    )

    valid_starts: list[int] = []
    stf_length = len(preamble.stf)
    short_length = len(preamble.stf_short_symbol)
    candidates = np.flatnonzero(stf_metric >= stf_threshold)
    if len(candidates) == 0:
        return np.empty(0, dtype=int)

    group_boundaries = np.flatnonzero(
        np.diff(candidates) > max(1, short_length // 2)
    ) + 1
    candidate_groups = np.split(candidates, group_boundaries)
    ltf_search_radius = short_length

    for group in candidate_groups:
        if len(group) < 3 * short_length:
            continue

        coarse_start = int(group[0])
        expected_ltf_start = coarse_start + stf_length
        search_start = max(
            0,
            expected_ltf_start - ltf_search_radius,
        )
        search_stop = min(
            len(received),
            expected_ltf_start
            + ltf_search_radius
            + len(preamble.ltf),
        )
        if search_stop - search_start < len(preamble.ltf):
            continue

        local_metric = normalized_correlation_metric(
            received=received[search_start:search_stop],
            preamble=preamble.ltf,
        )
        best_local = int(np.argmax(local_metric))
        if local_metric[best_local] < ltf_threshold:
            continue

        ltf_start = search_start + best_local
        preamble_start = ltf_start - stf_length
        if (
            valid_starts
            and preamble_start - valid_starts[-1] < minimum_separation
        ):
            continue
        valid_starts.append(preamble_start)

    return np.asarray(valid_starts, dtype=int)


def estimate_fractional_timing_offset(
    received: np.ndarray,
    known_sequence: np.ndarray,
    integer_start: int,
    search_step: float = 0.02,
) -> float:
    """Estimate the fractional advance that maximizes LTF correlation."""

    if not 0.0 < search_step <= 0.5:
        raise ValueError("search_step must be in (0, 0.5].")

    sequence_length = len(known_sequence)
    margin = 16
    window_start = integer_start - margin
    window_stop = integer_start + sequence_length + margin
    if window_start < 0 or window_stop > len(received):
        raise ValueError(
            "The LTF and interpolation margins must be available."
        )

    window = received[window_start:window_stop]
    candidates = np.arange(
        -0.5,
        0.5 + 0.5 * search_step,
        search_step,
    )
    scores = np.empty(len(candidates), dtype=np.float64)
    known_energy = float(np.vdot(known_sequence, known_sequence).real)

    for index, candidate in enumerate(candidates):
        candidate = float(np.clip(candidate, -0.5, 0.5))
        shifted = correct_fractional_timing(
            window,
            offset_samples=candidate,
        )
        segment = shifted[margin:margin + sequence_length]
        segment_energy = float(np.vdot(segment, segment).real)
        scores[index] = (
            abs(np.vdot(known_sequence, segment)) ** 2
            / max(known_energy * segment_energy, 1e-12)
        )

    return float(candidates[int(np.argmax(scores))])


def correct_fractional_timing(
    samples: np.ndarray,
    offset_samples: float,
    half_length: int = 12,
) -> np.ndarray:
    """Advance samples by a fractional offset using a windowed-sinc FIR."""

    if half_length <= 0:
        raise ValueError("half_length must be positive.")
    if abs(offset_samples) > 0.5:
        raise ValueError("offset_samples must be between -0.5 and 0.5.")
    if abs(offset_samples) < 1e-12:
        return samples.copy()

    tap_indices = np.arange(-half_length, half_length + 1)
    taps = np.sinc(tap_indices + offset_samples)
    taps *= np.hamming(len(taps))
    taps /= np.sum(taps)

    return np.convolve(samples, taps, mode="same")
