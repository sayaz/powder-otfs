import numpy as np

from powder_otfs.estimation.circulant import (
    block_circulant_from_impulse_response,
    circulant_from_first_column,
)


def test_circulant_from_first_column() -> None:
    first_column = np.array(
        [
            1.0 + 0.0j,
            2.0 + 0.0j,
            3.0 + 0.0j,
        ],
        dtype=np.complex128,
    )

    result = circulant_from_first_column(
        first_column,
    )

    expected = np.array(
        [
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_block_circulant_from_impulse_response() -> None:
    impulse_response = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.complex128,
    )

    result = block_circulant_from_impulse_response(
        impulse_response,
    )

    first_block = np.array(
        [
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=np.complex128,
    )

    second_block = np.array(
        [
            [4.0, 6.0, 5.0],
            [5.0, 4.0, 6.0],
            [6.0, 5.0, 4.0],
        ],
        dtype=np.complex128,
    )

    expected = np.block(
        [
            [first_block, second_block],
            [second_block, first_block],
        ]
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )