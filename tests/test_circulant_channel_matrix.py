import numpy as np

from powder_otfs.estimation.circulant import (
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