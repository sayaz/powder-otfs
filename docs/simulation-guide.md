# Simulation Guide

This guide explains how to install, configure, run, and debug the current OTFS
simulation. OTA instructions will be added after the POWDER radio implementation
has been developed and validated.

## Installation

Run the following commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

The project root is the directory containing:

```text
README.md
pyproject.toml
examples/
src/
tests/
```

## Verify the installation

From the project root:

```bash
python -m pytest
```

All tests should pass before running experiments or changing algorithms.

## Run the simulation

Run the default configuration:

```bash
python examples/end_to_end.py
```

Select the number of active predefined paths:

```bash
python examples/end_to_end.py --num-paths 5
```

The supported range is 1 through 10.

## Where to change parameters

Open:

```text
examples/end_to_end.py
```

The main user settings are at the beginning of `main()`.

### Grid and frames

```python
num_delay_bins = 32
num_doppler_bins = 16
qam_order = 4
num_frames = 100
```

The current modulation implementation supports only `qam_order = 4`.

Increasing the grid dimensions increases the number of symbols per frame, but
it also increases the channel-matrix size and equalization time substantially.

### Sampling, noise, and repeatability

```python
sample_rate = 1e6
snr_db = 30.0
random_seed = 12345
```

- `sample_rate` controls the sample timing and Doppler-bin resolution.
- `snr_db` controls AWGN strength.
- `random_seed` makes the generated bits and fading repeatable.

### Number of frames

```python
num_frames = 100
```

BER is calculated across all frames:

```text
BER = total bit errors / total processed bits
```

Processing multiple frames provides more transmitted bits than a single-frame
measurement and allows fading gains to change between frames.

### Pilot settings

```python
pilot_value = 4.0 + 0.0j
threshold_factor = 5.0
```

`pilot_value` controls the known embedded pilot. A stronger pilot is easier to
detect but consumes more energy.

The detection threshold is:

```text
threshold_factor × sqrt(channel noise variance)
```

If weak paths are missed, the threshold may be too high. If many false paths
are detected, it may be too low.

### Supported delay and Doppler

```python
maximum_supported_delay = 3
maximum_supported_doppler = 2
```

These are receiver design assumptions, not values discovered from the channel.
They determine the pilot guard and observation regions.

All active paths must fit within the selected support:

```text
0 <= path delay <= maximum_supported_delay
abs(path Doppler bin) <= maximum_supported_doppler
```

Larger support regions protect the pilot from a wider channel spread but leave
fewer bins for data.

### Channel paths

The built-in path table is:

```python
path_definitions = [
    (delay_samples, doppler_bin, complex_gain),
    ...
]
```

Edit these tuples to change the simulated paths. The `--num-paths` argument
selects the first requested number of entries.

The Doppler frequency is calculated automatically:

```text
doppler_resolution = sample_rate / (
    num_delay_bins * num_doppler_bins
)

doppler_hz = doppler_bin * doppler_resolution
```

Keep Doppler values on this grid while using the current embedded-pilot
estimator.

### Fading

```python
fading_model = "rayleigh"
rician_k = 5.0
```

Choose:

- `"fixed"`
- `"rayleigh"`
- `"rician"`

`rician_k` is used only for the Rician model.

### Equalizer

```python
equalizer_name = "mmse"
```

Choose `"zf"` or `"mmse"`. MMSE is normally more robust when the estimated
channel matrix is noisy or ill-conditioned.

### Plots

```python
show_plots = True
```

Set this to `False` for non-interactive runs.

## Understanding terminal output

The initial summary reports:

- modulation order;
- delay-Doppler grid dimensions;
- data symbols and bits per frame;
- total frames and bits;
- pilot value and position;
- assumed delay-Doppler support;
- guard dimensions;
- Doppler resolution;
- SNR and fading model;
- active physical paths.

The final result reports:

- estimator used;
- equalizer used;
- pilot-detection threshold;
- total bit errors;
- total processed bits;
- BER.

Zero BER means no error occurred in the processed bits. It does not prove that
the receiver will have zero BER under other random seeds, SNR values, channel
paths, fractional Doppler, synchronization errors, or OTA conditions.

## Understanding the debug view

When `show_plots` is enabled, the first processed frame is displayed.

### Tx DD Structure

This categorical plot shows:

- blue: QPSK data bins;
- black: zero-valued guard bins;
- yellow: embedded pilot.

Use it to confirm that the pilot and guard region are positioned correctly.

### Rx DD Magnitude

This shows the received delay-Doppler magnitude before equalization. Multipath,
Doppler, fading, and noise distort the transmitted grid.

### Equalized DD Magnitude

This shows the delay-Doppler grid after ZF or MMSE equalization. Data magnitudes
should return close to their transmitted QPSK magnitudes when estimation and
equalization work correctly.

### Estimated Delay-Doppler Channel Taps

This displays the pilot-observation region used by the embedded-pilot
estimator. Strong shifted pilot copies correspond to detected channel paths.

### Equalization Error

This displays:

```text
20 log10(|equalized grid - transmitted grid|)
```

over data bins. More-negative values indicate smaller errors.

### Data-Symbol Constellation

This overlays:

- transmitted QPSK points;
- received data symbols before equalization;
- data symbols after equalization.

The equalized symbols should cluster around the transmitted constellation
points.

## Debugging checklist

If BER is unexpectedly high:

1. Run `python -m pytest`.
2. Use `fading_model = "fixed"` to remove random fading.
3. Start with one path at zero delay and zero Doppler.
4. Confirm BER is near zero at high SNR.
5. Add one impairment at a time.
6. Confirm every active path lies inside the configured delay-Doppler support.
7. Inspect the pilot-response plot for the expected number of visible paths.
8. Compare ZF and MMSE.
9. Inspect the equalization-error and constellation plots.
10. Keep Doppler shifts grid-aligned with the current estimator.

If the estimator reports no detected paths, check:

- pilot amplitude;
- SNR;
- threshold factor;
- pilot observation region;
- active delay and Doppler values.

If ZF performs poorly while MMSE succeeds, the estimated channel matrix may be
ill-conditioned and ZF may be amplifying noise.

## Useful source files

| Task | File |
|---|---|
| End-to-end settings and experiment loop | `examples/end_to_end.py` |
| QPSK mapping and decisions | `src/powder_otfs/modulation/qam.py` |
| Pilot and delay-Doppler grid | `src/powder_otfs/otfs/grid.py` |
| OTFS transforms | `src/powder_otfs/otfs/transforms.py` |
| Multipath and AWGN channel | `src/powder_otfs/channel/channel.py` |
| Delay and Doppler | `src/powder_otfs/channel/delay.py`, `doppler.py` |
| Fading models | `src/powder_otfs/channel/fading.py` |
| Pilot estimator | `src/powder_otfs/estimation/pilot.py` |
| Perfect CSI reference | `src/powder_otfs/estimation/perfect.py` |
| ZF and MMSE | `src/powder_otfs/equalization/` |
| Debug plots | `src/powder_otfs/visualization/plots.py` |
| BER | `src/powder_otfs/metrics/ber.py` |

## OTA experiments

OTA support is not implemented yet. The future guide will cover:

- POWDER reservation and node selection;
- B210 and X310 setup;
- center frequency, sample rate, bandwidth, and gain;
- offline waveform transmission and IQ capture;
- frame detection and synchronization;
- frequency-offset correction;
- channel estimation and decoding;
- result collection and experiment reproducibility;
- transition from offline processing to real-time operation.

These instructions should be written only after the actual radio pipeline is
implemented and tested, so the documentation matches the working system.
