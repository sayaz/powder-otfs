from dataclasses import dataclass

import numpy as np

from powder_otfs.modulation.qam import qam_modulate
from powder_otfs.ota.config import OTFSOTAConfig
from powder_otfs.otfs.grid import insert_pilot_and_guards
from powder_otfs.otfs.transforms import heisenberg, isfft


@dataclass(frozen=True, slots=True)
class OTFSPayload:
    """Known transmitted bits, symbols, DD grid, and time samples."""

    bits: np.ndarray
    data_symbols: np.ndarray
    dd_grid: np.ndarray
    waveform: np.ndarray


def create_otfs_payload(
    config: OTFSOTAConfig,
) -> OTFSPayload:
    """Create the deterministic pilot-bearing OTFS OTA payload."""

    rng = np.random.default_rng(
        config.random_seed
    )
    bits = rng.integers(
        0,
        2,
        config.bits_per_frame,
        dtype=np.uint8,
    )
    data_symbols = qam_modulate(
        bits,
        order=config.qam_order,
    )

    dd_grid = np.zeros(
        config.grid_shape,
        dtype=np.complex128,
    )
    dd_grid[config.data_mask] = (
        data_symbols
    )

    pilot_delay, pilot_doppler = (
        config.pilot_position
    )
    dd_grid = insert_pilot_and_guards(
        grid=dd_grid,
        pilot_delay_bin=pilot_delay,
        pilot_doppler_bin=pilot_doppler,
        pilot_amplitude=float(
            np.abs(config.pilot_value)
        ),
        guard_delay=config.guard_delay,
        guard_doppler=config.guard_doppler,
    )

    waveform = heisenberg(
        isfft(dd_grid)
    )

    return OTFSPayload(
        bits=bits,
        data_symbols=data_symbols,
        dd_grid=dd_grid,
        waveform=waveform,
    )
