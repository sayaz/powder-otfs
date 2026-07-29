import numpy as np
import pytest

from powder_otfs.fec.qc_ldpc import create_qc_ldpc_code


@pytest.mark.parametrize("rate", ("1/2", "2/3", "3/4"))
def test_qc_ldpc_codeword_satisfies_all_checks(rate: str) -> None:
    code = create_qc_ldpc_code(rate, maximum_codeword_length=862)
    rng = np.random.default_rng(10)
    information = rng.integers(
        0,
        2,
        code.information_length,
        dtype=np.uint8,
    )

    codeword = code.encode(information)

    assert code.codeword_length == 828
    assert not np.any(code.parity_check(codeword))


@pytest.mark.parametrize("rate", ("1/2", "2/3", "3/4"))
def test_qc_ldpc_decoder_recovers_clean_codeword(rate: str) -> None:
    code = create_qc_ldpc_code(rate, maximum_codeword_length=862)
    rng = np.random.default_rng(20)
    information = rng.integers(
        0,
        2,
        code.information_length,
        dtype=np.uint8,
    )
    codeword = code.encode(information)
    llrs = 8.0 * (1.0 - 2.0 * codeword.astype(float))

    decoded = code.decode(llrs, maximum_iterations=30)

    np.testing.assert_array_equal(decoded, information)


def test_rate_half_qc_ldpc_corrects_sparse_errors() -> None:
    code = create_qc_ldpc_code("1/2", maximum_codeword_length=862)
    rng = np.random.default_rng(30)
    information = rng.integers(
        0,
        2,
        code.information_length,
        dtype=np.uint8,
    )
    codeword = code.encode(information)
    llrs = 8.0 * (1.0 - 2.0 * codeword.astype(float))
    llrs[[17, 208, 511]] *= -1.0

    decoded = code.decode(llrs, maximum_iterations=40)

    np.testing.assert_array_equal(decoded, information)
