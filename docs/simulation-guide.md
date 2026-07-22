# Simulation Guide

## Install

From the project root:

```bash
python3 -m pip install .
```

## Run

Run the default simulation:

```bash
python3 examples/simulation/end_to_end.py
```

Select between one and ten predefined channel paths:

```bash
python3 examples/simulation/end_to_end.py --num-paths 5
```

## Configure

Edit the settings near the beginning of `main()` in:

```text
examples/simulation/end_to_end.py
```

The main settings are:

- `num_delay_bins` and `num_doppler_bins`: DD-grid dimensions;
- `qam_order`: modulation order (currently 4-QAM);
- `sample_rate`: waveform sample rate;
- `snr_db`: channel signal-to-noise ratio;
- `num_frames`: number of frames processed;
- `fading_model`: `"fixed"`, `"rayleigh"`, or `"rician"`;
- `equalizer_name`: `"zf"` or `"mmse"`;
- `channel_matrix_method`: channel-matrix construction method;
- `maximum_supported_delay` and `maximum_supported_doppler`: pilot observation and guard support;
- `show_plots`: enables or disables the debug view.

Edit `path_definitions` to change each path's delay, Doppler bin, and complex
gain. The `--num-paths` option selects entries from the start of this list.

The Doppler frequency is calculated automatically from the selected Doppler
bin:

```text
Doppler resolution = sample rate / (delay bins × Doppler bins)
```

Every active path must remain inside the configured delay and Doppler support.

## Results

The terminal reports the active configuration, detected channel-estimation
threshold, processed bits, bit errors, and aggregate BER.

The debug view shows:

- transmitted data, pilot, and DD guard positions;
- received DD-grid magnitude;
- equalized DD-grid magnitude;
- estimated delay-Doppler channel taps;
- equalization error;
- received and equalized constellations;
- optional channel-matrix validation.

If BER is unexpectedly high, first confirm that all paths fit inside the
configured pilot support. Then compare ZF with MMSE and inspect the estimated
channel taps and constellation.
