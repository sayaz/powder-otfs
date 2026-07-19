import numpy as np


def estimate_cfo(
    repeated_preamble: np.ndarray,
    sample_rate: float,
) -> float:
    """Estimate carrier-frequency offset from two repeated halves."""

    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive.")

    if (
        len(repeated_preamble) == 0
        or len(repeated_preamble) % 2 != 0
    ):
        raise ValueError(
            "repeated_preamble must have a positive even length."
        )

    half_length = len(repeated_preamble) // 2
    first_half = repeated_preamble[:half_length]
    second_half = repeated_preamble[half_length:]

    phase_difference = np.angle(
        np.vdot(
            first_half,
            second_half,
        )
    )

    return float(
        phase_difference
        * sample_rate
        / (2.0 * np.pi * half_length)
    )


def correct_cfo(
    samples: np.ndarray,
    cfo_hz: float,
    sample_rate: float,
) -> np.ndarray:
    """Remove a carrier-frequency offset from complex samples."""

    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive.")

    sample_indices = np.arange(
        len(samples),
        dtype=np.float64,
    )

    correction = np.exp(
        -1j
        * 2.0
        * np.pi
        * cfo_hz
        * sample_indices
        / sample_rate
    )

    return samples * correction
