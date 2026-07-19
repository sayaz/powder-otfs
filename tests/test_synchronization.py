import numpy as np

from powder_otfs.ota.framing import create_preamble
from powder_otfs.ota.synchronization import find_payload_start


def test_find_payload_start() -> None:
    preamble = create_preamble(
        half_length=16,
        seed=1,
    )

    leading_samples = np.zeros(
        25,
        dtype=np.complex64,
    )
    payload = np.ones(
        20,
        dtype=np.complex64,
    )

    received = np.concatenate(
        (
            leading_samples,
            preamble,
            payload,
        )
    )

    payload_start = find_payload_start(
        received=received,
        preamble=preamble,
    )

    assert payload_start == (
        len(leading_samples) + len(preamble)
    )