import numpy as np
import pytest

from powder_otfs.modulation.qam import qam_modulate, qam_demodulate

def test_qpsk_output_length() -> None:
    bits = np.array([0,1,1,0])
    symbols = qam_modulate(bits, order=4)
    
    assert len(symbols) == 2


def test_qpsk_symbol_values() -> None:
    bits = np.array([0,0,0,1,1,1,1,0])
    symbols = qam_modulate(bits, order=4)

    expected = np.array([
        (1+1j) / np.sqrt(2),
        (1-1j) / np.sqrt(2),
        (-1-1j) / np.sqrt(2),
        (-1+1j) / np.sqrt(2),
    ])

    assert np.allclose(symbols, expected)

def test_qpsk_odd_number_of_bits() -> None:
    bits = np.array([0,1,1])

    with pytest.raises(ValueError):
        qam_modulate(bits, order=4)

def test_qpsk_round_trip() -> None:
    bits = np.array([0,0,0,1,1,1,1,0])
    symbols = qam_modulate(bits, order=4)
    recovered_bits = qam_demodulate(symbols, order=4)

    assert np.array_equal(bits, recovered_bits)