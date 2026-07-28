from dataclasses import dataclass

import numpy as np

from powder_otfs.equalization.mmse import mmse_equalizer
from powder_otfs.equalization.zf import zero_forcing_equalizer
from powder_otfs.estimation.pilot import pilot_channel_estimate
from powder_otfs.ota.config import OTFSOTAConfig


@dataclass(frozen=True, slots=True)
class FrameEqualizationResult:
    """Store independently estimated and equalized OTFS frames."""

    equalized_grids: np.ndarray
    noise_variances: np.ndarray
    estimation_thresholds: np.ndarray
    rejected_frames: int
    estimator_method: str
    equalizer_method: str


def estimate_and_equalize_frames(
    received_grids: np.ndarray,
    config: OTFSOTAConfig,
) -> FrameEqualizationResult:
    """Estimate and equalize every received frame independently."""

    if received_grids.ndim != 3:
        raise ValueError(
            "received_grids must have shape "
            "(frames, delay_bins, doppler_bins)."
        )
    if received_grids.shape[1:] != config.grid_shape:
        raise ValueError(
            "received grid dimensions do not match the OTA configuration."
        )

    observation_mask = np.zeros(
        config.grid_shape,
        dtype=bool,
    )
    observation_mask[config.observation_slices] = True
    noise_mask = ~config.data_mask & ~observation_mask

    equalized_grids: list[np.ndarray] = []
    noise_variances: list[float] = []
    estimation_thresholds: list[float] = []
    rejected_frames = 0
    estimator_method = "Embedded Pilot"

    equalizer_name = config.equalizer_name.lower()
    if equalizer_name == "mmse":
        equalizer_method = "MMSE"
    elif equalizer_name == "zf":
        equalizer_method = "Zero Forcing"
    else:
        raise ValueError("equalizer_name must be 'zf' or 'mmse'.")

    for received_grid in received_grids:
        noise_samples = received_grid[noise_mask]
        noise_variance = max(
            float(
                np.median(np.abs(noise_samples) ** 2)
                / np.log(2.0)
            ),
            1e-12,
        )
        estimation_threshold = (
            config.threshold_factor
            * np.sqrt(noise_variance)
        )

        pilot_observation = np.zeros_like(received_grid)
        pilot_observation[config.observation_slices] = received_grid[
            config.observation_slices
        ]

        try:
            estimate = pilot_channel_estimate(
                received_pilot_grid=pilot_observation,
                pilot_position=config.pilot_position,
                pilot_value=config.pilot_value,
                sample_rate=config.sample_rate,
                noise_variance=noise_variance,
                threshold=estimation_threshold,
            )
        except ValueError as error:
            if "No channel paths detected" not in str(error):
                raise
            rejected_frames += 1
            continue

        if equalizer_name == "mmse":
            equalized_grid = mmse_equalizer(
                received_grid=received_grid,
                estimate=estimate,
                symbol_energy=1.0,
            )
        else:
            equalized_grid = zero_forcing_equalizer(
                received_grid=received_grid,
                estimate=estimate,
            )

        equalized_grids.append(equalized_grid.symbols)
        noise_variances.append(noise_variance)
        estimation_thresholds.append(estimation_threshold)
        estimator_method = estimate.method

    if not equalized_grids:
        raise RuntimeError(
            "No frames produced a valid embedded-pilot channel estimate."
        )

    return FrameEqualizationResult(
        equalized_grids=np.stack(equalized_grids),
        noise_variances=np.asarray(noise_variances),
        estimation_thresholds=np.asarray(estimation_thresholds),
        rejected_frames=rejected_frames,
        estimator_method=estimator_method,
        equalizer_method=equalizer_method,
    )
