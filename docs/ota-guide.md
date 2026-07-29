# POWDER Offline OTA Guide

## Run the experiment

Wait until startup has finished on both nodes. The profile installs and
configures the project automatically at:

```bash
cd ~/powder-otfs
```

Start the receiver on the `rx` node:

```bash
python3 examples/ota/ota_rx.py
```

Then start the transmitter on the `tx` node:

```bash
python3 examples/ota/ota_tx.py
```

The receiver uses the STF for coarse detection and CFO estimation, the LTF for
fine frame alignment and CFO refinement, and the LTF correlation for
fractional-sample timing correction. It then converts each payload to the DD
domain, estimates the channel from the embedded pilot, applies ZF or MMSE
equalization, and reports aggregate BER.

The received complex-IQ samples are saved at:

```text
~/powder-otfs/results/rx_samples.npy
```

## Configure

The POWDER profile automatically supplies the selected radio type, UHD device
arguments, and the 3.370 GHz center frequency. The indoor profile requests the
reserved 3360-3380 MHz range.

The OTA commands accept bandwidth, gain, capture length, grid, and supported
delay/Doppler options. The default bandwidth and sample rate are 20 MHz and
20 MS/s. For example:

```bash
python3 examples/ota/ota_rx.py --bandwidth-mhz 20 --rx-gain 20
python3 examples/ota/ota_tx.py --bandwidth-mhz 20 --tx-gain 10 --peak-amplitude 0.85
```

The transmitter defaults to a peak complex-sample magnitude of `0.85`. Use
`--peak-amplitude` to test another value between zero and one.

FEC is disabled by default. Enable the selectable QC-LDPC implementation on
both nodes with matching settings:

```bash
python3 examples/ota/ota_rx.py --bandwidth-mhz 20 --rx-gain 20 --fec qc-ldpc --fec-rate 1/2
python3 examples/ota/ota_tx.py --bandwidth-mhz 20 --tx-gain 20 --peak-amplitude 0.85 --fec qc-ldpc --fec-rate 1/2
```

Available QC-LDPC rates are `1/2`, `2/3`, and `3/4`. Rate `1/2` provides the
strongest protection and should be tested first.

The transmitter and receiver entry points are:

```text
examples/ota/x310_tx.py
examples/ota/x310_rx.py
```

Shared OTFS settings—including grid size, pilot, DD guard, cyclic prefix,
sample rate, synchronization threshold, and equalizer—are in:

```text
src/powder_otfs/ota/config.py
```

The transmitter and receiver must use the same shared settings.

## Copy and inspect a capture

On your computer, copy the capture from the receiver node:

```bash
scp <username>@<rx-hostname>:/users/<username>/powder-otfs/results/rx_samples.npy .
```

From your local project directory, display the first detected frame:

```bash
python3 examples/ota/plot_rx_samples.py rx_samples.npy
```

Display another frame with:

```bash
python3 examples/ota/plot_rx_samples.py rx_samples.npy --frame-index 10
```

The debug view shows the transmitted DD structure, received and equalized DD
grids, estimated channel taps, equalization error, and data constellation.

If radio discovery fails, check the X310 connection with:

```bash
uhd_find_devices
uhd_usrp_probe --args "addr=192.168.40.2"
```
