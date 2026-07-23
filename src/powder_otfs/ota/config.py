import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OTFSOTAConfig:
    """Shared OTFS waveform settings for OTA TX, RX, and offline analysis."""

    num_delay_bins: int = 32
    num_doppler_bins: int = 16
    qam_order: int = 4
    bandwidth_mhz: float = 1.0
    pilot_value: complex = 4.0 + 0.0j
    maximum_supported_delay: int = 4
    maximum_supported_doppler: int = 2
    threshold_factor: float = 5.0
    equalizer_name: str = "mmse"
    preamble_duration_seconds: float = 128e-6
    time_guard_duration_seconds: float = 128e-6
    synchronization_threshold: float = 0.50
    random_seed: int = 12345

    def __post_init__(self) -> None:
        if self.bandwidth_mhz not in (1.0, 5.0, 10.0, 20.0):
            raise ValueError(
                "bandwidth_mhz must be 1, 5, 10, or 20."
            )

    @property
    def sample_rate(self) -> float:
        """Return the sample rate used for the selected bandwidth."""

        return self.bandwidth_mhz * 1e6

    @property
    def preamble_half_length(self) -> int:
        """Return half of the repeated preamble in samples."""

        return max(
            1,
            int(
                round(
                    self.preamble_duration_seconds
                    * self.sample_rate
                    / 2.0
                )
            ),
        )

    @property
    def time_guard_samples(self) -> int:
        """Return one time-domain guard duration in samples."""

        return max(
            0,
            int(
                round(
                    self.time_guard_duration_seconds
                    * self.sample_rate
                )
            ),
        )

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


def add_ota_config_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add shared OTFS waveform options to a command-line parser."""

    parser.add_argument(
        "--bandwidth-mhz",
        type=float,
        choices=(1.0, 5.0, 10.0, 20.0),
        default=1.0,
        help="Baseband bandwidth and sample rate in MHz (default: 1).",
    )
    parser.add_argument(
        "--delay-bins",
        type=int,
        default=32,
        help="Number of DD delay bins (default: 32).",
    )
    parser.add_argument(
        "--doppler-bins",
        type=int,
        default=16,
        help="Number of DD Doppler bins (default: 16).",
    )
    parser.add_argument(
        "--max-delay-samples",
        type=int,
        default=4,
        help="Maximum supported delay and CP length in samples (default: 4).",
    )
    parser.add_argument(
        "--max-doppler-bins",
        type=int,
        default=2,
        help="Maximum supported Doppler shift in bins (default: 2).",
    )


def ota_config_from_arguments(
    arguments: argparse.Namespace,
) -> OTFSOTAConfig:
    """Create the shared OTA configuration from parsed arguments."""

    return OTFSOTAConfig(
        bandwidth_mhz=arguments.bandwidth_mhz,
        num_delay_bins=arguments.delay_bins,
        num_doppler_bins=arguments.doppler_bins,
        maximum_supported_delay=arguments.max_delay_samples,
        maximum_supported_doppler=arguments.max_doppler_bins,
    )
