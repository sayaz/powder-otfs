import numpy as np


def bit_error_rate(
    transmitted_bits: np.ndarray,
    received_bits: np.ndarray,
) -> float:
    """
    Compute the bit error rate (BER).
    """

    if len(transmitted_bits) != len(received_bits):
        raise ValueError("Bit arrays must have the same length.")

    errors = np.count_nonzero(
        transmitted_bits != received_bits
    )

    return errors / len(transmitted_bits)