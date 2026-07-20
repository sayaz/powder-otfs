# POWDER X310 Offline OTA Guide

## Start the experiment

Wait until startup finishes on both `tx` and `rx`, then log in to each compute
node. The project is installed at:

```bash
cd ~/powder-otfs
```

The profile configures the system Python and `PYTHONPATH` automatically. No
virtual environment or activation command is required.

X310 discovery can be checked when troubleshooting:

```bash
uhd_find_devices
uhd_usrp_probe --args "addr=192.168.40.2"
```

This check is optional during normal use.

## Run the link

Start the receiver first on the `rx` node:

```bash
cd ~/powder-otfs
python3 examples/ota/x310_rx.py
```

Then start the transmitter on the `tx` node:

```bash
cd ~/powder-otfs
python3 examples/ota/x310_tx.py
```

The OTA waveform contains:

- a time-domain preamble for synchronization and CFO correction;
- a cyclic prefix for the supported delay spread;
- QPSK data in the DD grid;
- one embedded DD pilot and a user-selected DD guard region.

The receiver detects multiple frames, corrects CFO, estimates the
delay-Doppler channel from the embedded pilot, applies ZF or MMSE equalization,
reports aggregate BER, and saves the complete complex-IQ capture at:

```text
~/powder-otfs/results/rx_samples.npy
```

## Copy the capture to your computer

Run this command on your computer, not on the POWDER node:

```bash
scp <username>@<rx-hostname>:/users/<username>/powder-otfs/results/rx_samples.npy .
```

For example, replace `<username>` with your POWDER username and
`<rx-hostname>` with the hostname shown on the experiment page.

## Plot the capture locally

From the local project directory:

```bash
python examples/ota/plot_rx_samples.py rx_samples.npy
```

The offline debug view shows:

- transmitted DD-grid magnitude;
- received DD grid before correction;
- received DD grid after CFO and channel-gain correction;
- DD-grid error;
- transmitted, raw received, and corrected constellations;
- the detected preamble and payload magnitude.

Select another detected frame with:

```bash
python examples/ota/plot_rx_samples.py rx_samples.npy --frame-index 10
```

The `.npy` file retains the original complex sample values and NumPy data type.
