import numpy as np

from powder_otfs.channel.awgn import add_awgn
from powder_otfs.channel.delay import apply_delay
from powder_otfs.channel.doppler import apply_doppler

def apply_channel(
    waveform: np.ndarray,
    delay_samples: int,
    doppler_hz: float,
    sample_rate: float,
    snr_db: float,
) -> np.ndarray:
    """
    Apply a wireless channel consisting of:
        1. Propagation delay
        2. Doppler shift
        3. AWGN
    """

    waveform = apply_delay(
        waveform, delay_samples,
    )

    waveform = apply_doppler(
        waveform,
        doppler_hz,
        sample_rate,
    )

    waveform = add_awgn(
        waveform,
        snr_db,
    )

    return waveform