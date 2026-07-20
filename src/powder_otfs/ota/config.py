from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OTFSOTAConfig:
    """Shared OTFS waveform settings for OTA TX, RX, and offline analysis."""

    num_delay_bins: int = 32
    num_doppler_bins: int = 16
    qam_order: int = 4
    sample_rate: float = 1e6
    pilot_value: complex = 4.0 + 0.0j
    maximum_supported_delay: int = 3
    maximum_supported_doppler: int = 2
    threshold_factor: float = 5.0
    equalizer_name: str = "mmse"
    time_guard_samples: int = 128
    preamble_half_length: int = 64
    synchronization_threshold: float = 0.50
    random_seed: int = 12345

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (
            self.num_delay_bins,
            self.num_doppler_bins,
        )

    @property
    def num_grid_symbols(self) -> int:
        return (
            self.num_delay_bins
            * self.num_doppler_bins
        )

    @property
    def pilot_position(self) -> tuple[int, int]:
        return (
            self.num_delay_bins // 2,
            self.num_doppler_bins // 2,
        )

    @property
    def guard_delay(self) -> int:
        return self.maximum_supported_delay

    @property
    def guard_doppler(self) -> int:
        return 2 * self.maximum_supported_doppler

    @property
    def cyclic_prefix_samples(self) -> int:
        return self.maximum_supported_delay

    @property
    def data_mask(self) -> np.ndarray:
        pilot_delay, pilot_doppler = self.pilot_position

        delay_start = (
            pilot_delay
            - self.guard_delay
        )
        delay_stop = (
            pilot_delay
            + self.guard_delay
            + 1
        )
        doppler_start = (
            pilot_doppler
            - self.guard_doppler
        )
        doppler_stop = (
            pilot_doppler
            + self.guard_doppler
            + 1
        )

        if (
            delay_start < 0
            or delay_stop > self.num_delay_bins
            or doppler_start < 0
            or doppler_stop > self.num_doppler_bins
        ):
            raise ValueError(
                "The DD grid is too small for the selected guard region."
            )

        mask = np.ones(
            self.grid_shape,
            dtype=bool,
        )
        mask[
            delay_start:delay_stop,
            doppler_start:doppler_stop,
        ] = False
        return mask

    @property
    def num_data_symbols(self) -> int:
        return int(
            np.count_nonzero(
                self.data_mask
            )
        )

    @property
    def bits_per_frame(self) -> int:
        return (
            int(np.log2(self.qam_order))
            * self.num_data_symbols
        )

    @property
    def observation_slices(
        self,
    ) -> tuple[slice, slice]:
        pilot_delay, pilot_doppler = self.pilot_position
        return (
            slice(
                pilot_delay
                - self.maximum_supported_delay,
                pilot_delay + 1,
            ),
            slice(
                pilot_doppler
                - self.maximum_supported_doppler,
                pilot_doppler
                + self.maximum_supported_doppler
                + 1,
            ),
        )

    @property
    def doppler_resolution(self) -> float:
        return (
            self.sample_rate
            / self.num_grid_symbols
        )
