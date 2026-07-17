import numpy as np

def qam_modulate(bits: np.ndarray, order: int) -> np.ndarray:
    """Maps input bits to QAM symbols."""

    if len(bits) % 2 != 0:
        raise ValueError("QPSK requires an even number of bits")
    
    bit_pairs = bits.reshape(-1,2)
    symbols = (1 - 2 * bit_pairs[:,0]) + 1j * (1 - 2 * bit_pairs[:, 1])

    return symbols / np.sqrt(2)

def qam_demodulate(symbols: np.ndarray, order: int) -> np.ndarray:
    """Maps QAM symbols back to bits."""

    bits = np.empty(2 * len(symbols), dtype=int)

    bits[0::2] = (symbols.real < 0).astype(int)
    bits[1::2] = (symbols.imag < 0).astype(int)

    return bits
