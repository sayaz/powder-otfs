from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrainingPreamble:
    """802.11-style STF/LTF training fields used by the OTA receiver."""

    stf_short_symbol: np.ndarray
    ltf_symbol: np.ndarray
    stf_repetitions: int = 10
    ltf_cp_length: int = 32

    @property
    def stf(self) -> np.ndarray:
        """Return the repeated short training field."""

        return np.tile(
            self.stf_short_symbol,
            self.stf_repetitions,
        ).astype(np.complex64)

    @property
    def ltf(self) -> np.ndarray:
        """Return the LTF cyclic prefix followed by two LTF symbols."""

        cyclic_prefix = self.ltf_symbol[
            -self.ltf_cp_length:
        ]
        return np.concatenate(
            (cyclic_prefix, self.ltf_symbol, self.ltf_symbol)
        ).astype(np.complex64)

    @property
    def samples(self) -> np.ndarray:
        """Return the complete STF followed by the complete LTF."""

        return np.concatenate((self.stf, self.ltf))

    @property
    def ltf_symbol_offset(self) -> int:
        """Return the first LTF-symbol offset from the preamble start."""

        return len(self.stf) + self.ltf_cp_length


def create_training_preamble() -> TrainingPreamble:
    """Create the IEEE 802.11 legacy STF and LTF time sequences."""

    stf_subcarriers = np.sqrt(13.0 / 6.0) * np.asarray(
        [
            0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0,
            1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j,
            0, 0, 0, 1 + 1j, 0, 0, 0, 0, 0, 0, 0,
            -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j,
            0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0,
            1 + 1j, 0, 0,
        ],
        dtype=np.complex128,
    )
    ltf_subcarriers = np.asarray(
        [
            1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
            1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
            0,
            1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1,
            -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
        ],
        dtype=np.complex128,
    )
    subcarrier_indices = np.arange(-26, 27)

    stf_frequency = np.zeros(64, dtype=np.complex128)
    stf_frequency[subcarrier_indices % 64] = stf_subcarriers
    stf_symbol = np.fft.ifft(stf_frequency, norm="ortho")

    ltf_frequency = np.zeros(64, dtype=np.complex128)
    ltf_frequency[subcarrier_indices % 64] = ltf_subcarriers
    ltf_symbol = np.fft.ifft(ltf_frequency, norm="ortho")

    return TrainingPreamble(
        stf_short_symbol=stf_symbol[:16].astype(np.complex64),
        ltf_symbol=ltf_symbol.astype(np.complex64),
    )


def create_preamble(
    half_length: int = 64,
    seed: int = 12345,
) -> np.ndarray:
    """Create the legacy repeated preamble for saved-capture debugging."""

    if half_length <= 0:
        raise ValueError("half_length must be positive.")

    rng = np.random.default_rng(seed)
    half = (
        2 * rng.integers(0, 2, half_length) - 1
    ).astype(np.complex64)
    return np.concatenate((half, half))


def normalize_waveform(
    waveform: np.ndarray,
    peak_amplitude: float = 0.8,
) -> np.ndarray:
    """Scale a waveform to a selected peak amplitude."""

    if not 0.0 < peak_amplitude <= 1.0:
        raise ValueError(
            "peak_amplitude must be greater than 0 and at most 1."
        )

    maximum = float(np.max(np.abs(waveform)))

    if maximum == 0.0:
        return waveform.astype(np.complex64)

    normalized = waveform * (
        peak_amplitude / maximum
    )

    return normalized.astype(np.complex64)


def build_ota_frame(
    payload: np.ndarray,
    preamble: np.ndarray,
    guard_samples: int = 128,
    cyclic_prefix_samples: int = 0,
) -> np.ndarray:
    """Build one OTA frame with guards, preamble, CP, and payload."""

    if guard_samples < 0:
        raise ValueError("guard_samples must be non-negative.")

    if not 0 <= cyclic_prefix_samples <= len(payload):
        raise ValueError(
            "cyclic_prefix_samples must be between 0 "
            "and the payload length."
        )

    guard = np.zeros(
        guard_samples,
        dtype=np.complex64,
    )
    cyclic_prefix = payload[
        len(payload) - cyclic_prefix_samples:
    ] if cyclic_prefix_samples else np.empty(
        0,
        dtype=payload.dtype,
    )

    return np.concatenate(
        (
            guard,
            preamble.astype(np.complex64),
            cyclic_prefix.astype(np.complex64),
            payload.astype(np.complex64),
            guard,
        )
    )
