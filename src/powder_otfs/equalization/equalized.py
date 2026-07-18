from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EqualizedGrid:
    """Stores the equalized delay-Doppler grid."""

    symbols: np.ndarray
    method: str