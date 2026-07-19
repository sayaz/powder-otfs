import numpy as np

from powder_otfs.channel.path import ChannelPath


def apply_fading(
    paths: list[ChannelPath],
    model: str,
    rng: np.random.Generator,
    rician_k: float = 5.0,
) -> list[ChannelPath]:
    """Generate path gains for one fading realization."""

    model = model.lower()

    if model not in {"fixed", "rayleigh", "rician"}:
        raise ValueError(
            "model must be 'fixed', 'rayleigh', or 'rician'."
        )

    if rician_k < 0:
        raise ValueError("rician_k must be non-negative.")

    faded_paths: list[ChannelPath] = []

    for path in paths:
        if model == "fixed":
            fading_coefficient = 1.0 + 0.0j

        elif model == "rayleigh":
            fading_coefficient = (
                rng.standard_normal()
                + 1j * rng.standard_normal()
            ) / np.sqrt(2.0)

        else:
            line_of_sight = np.sqrt(
                rician_k / (rician_k + 1.0)
            )
            scattered = np.sqrt(
                1.0 / (rician_k + 1.0)
            ) * (
                rng.standard_normal()
                + 1j * rng.standard_normal()
            ) / np.sqrt(2.0)

            fading_coefficient = (
                line_of_sight + scattered
            )

        faded_paths.append(
            ChannelPath(
                delay_samples=path.delay_samples,
                doppler_hz=path.doppler_hz,
                gain=path.gain * fading_coefficient,
            )
        )

    return faded_paths