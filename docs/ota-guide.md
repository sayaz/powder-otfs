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

The receiver detects frames, estimates and corrects CFO, converts each payload
to the DD domain, estimates the channel from the embedded pilot, applies ZF or
MMSE equalization, and reports aggregate BER.

The received complex-IQ samples are saved at:

```text
~/powder-otfs/results/rx_samples.npy
```

## Configure

The POWDER profile automatically supplies the selected radio type, UHD device
arguments, and the 3.560 GHz center frequency.

Radio settings such as gain, antenna, and capture length are near the beginning
of:

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
