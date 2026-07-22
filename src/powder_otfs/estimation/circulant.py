import numpy as np


def circulant_from_first_column(
    column: np.ndarray,
) -> np.ndarray:
    """Construct a circulant matrix from its first column."""

    if column.ndim != 1:
        raise ValueError("column must be one-dimensional.")

    size = len(column)

    row_indices = np.arange(size)[:, None]
    column_indices = np.arange(size)[None, :]

    circulant_indices = (
        row_indices - column_indices
    ) % size

    return column[circulant_indices]