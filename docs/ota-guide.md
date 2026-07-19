# POWDER X310 Offline OTA Guide

## Start the experiment

Wait until startup finishes on both `tx` and `rx`, then log in to each compute
node. The project is installed at:

```bash
cd ~/powder-otfs
```

The Python environment activates automatically when you log in.

Verify the X310 on both nodes:

```bash
uhd_find_devices
uhd_usrp_probe --args "addr=192.168.40.2"
```

## Run the link

Start the receiver first on the `rx` node:

```bash
cd ~/powder-otfs
python examples/ota/x310_rx.py
```

Then start the transmitter on the `tx` node:

```bash
cd ~/powder-otfs
python examples/ota/x310_tx.py
```

The receiver saves the complete complex-IQ capture at:

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

The plot shows the received IQ waveform and magnitude. The `.npy` file retains
the original complex sample values and NumPy data type.
