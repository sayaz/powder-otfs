#!/usr/bin/env python
"""POWDER profile for a two-node OTFS conducted-RF experiment."""

import geni.portal as portal
import geni.rspec.igext as ig
import geni.rspec.pg as rspec


DESCRIPTION = """
# POWDER-OTFS Paired Radio Workbench

This profile creates a two-node SISO OTFS experiment using one POWDER paired
radio workbench. Each server connects to one NI X310 through a dedicated
10 Gb/s Ethernet link.

The X310 radios are connected through fixed 30 dB attenuators and share an
external 10 MHz clock and PPS reference.

The profile automatically installs UHD, GNU Radio, and POWDER-OTFS under the
POWDER user's home directory. The Python environment activates automatically
when the user logs in.
"""

INSTRUCTIONS = """
Wait until the startup status for both `tx` and `rx` reports `Finished`.

The project is available on both compute nodes at:

```bash
cd ~/powder-otfs
```

The Python environment is activated automatically at login.

Verify X310 access on both nodes:

```bash
uhd_find_devices
uhd_usrp_probe --args "addr=192.168.40.2"
```

The SISO conducted-RF connection uses:

- TX node: channel 0, `TX/RX` port
- RX node: channel 0, `RX2` port
- Fixed path attenuation: 30 dB
- Default center frequency: 3.5 GHz
- Default sample rate: 1 MS/s

Start the receiver first on `rx`:

```bash
cd ~/powder-otfs
python examples/ota/x310_rx.py
```

Immediately afterward, start the transmitter on `tx`:

```bash
cd ~/powder-otfs
python examples/ota/x310_tx.py
```

The receiver reports multi-frame synchronization and BER results. It also saves
the complete complex-IQ capture as:

```text
~/powder-otfs/results/rx_samples.npy
```

See `docs/ota-guide.md` for instructions to copy the capture to another
computer for offline analysis and plotting.
"""

COMPONENT_MANAGER_ID = (
    "urn:publicid:IDN+emulab.net+authority+cm"
)
UBUNTU_IMAGE = (
    "urn:publicid:IDN+emulab.net+image+"
    "emulab-ops//UBUNTU22-64-STD"
)
SETUP_COMMAND = (
    "sudo bash /local/repository/"
    "scripts/powder/setup-rf-bench.sh"
)

WORKBENCH_RADIOS = {
    "bench_a": ("oai-wb-a1", "oai-wb-a2"),
    "bench_b": ("oai-wb-b1", "oai-wb-b2"),
}


context = portal.Context()

context.defineParameter(
    name="workbench",
    description="Paired radio workbench",
    typ=portal.ParameterType.STRING,
    defaultValue="bench_b",
    legalValues=[
        ("bench_a", "Paired Radio Workbench A"),
        ("bench_b", "Paired Radio Workbench B"),
    ],
)

context.defineParameter(
    name="compute_type",
    description="Compute-node type connected to each X310",
    typ=portal.ParameterType.STRING,
    defaultValue="d740",
    legalValues=[
        ("d430", "Emulab d430"),
        ("d740", "Emulab d740"),
    ],
)

parameters = context.bindParameters()
request = context.makeRequestRSpec()


def add_radio_pair(
    role,
    radio_component_id,
):
    """Add one compute node, one X310, and their dedicated link."""

    compute = request.RawPC(role)
    compute.component_manager_id = COMPONENT_MANAGER_ID
    compute.hardware_type = parameters.compute_type
    compute.disk_image = UBUNTU_IMAGE

    compute_interface = compute.addInterface(
        "{}-usrp-interface".format(role)
    )
    compute_interface.addAddress(
        rspec.IPv4Address(
            "192.168.40.1",
            "255.255.255.0",
        )
    )

    compute.addService(
        rspec.Execute(
            shell="bash",
            command=SETUP_COMMAND,
        )
    )

    radio = request.RawPC("{}-sdr".format(role))
    radio.component_manager_id = COMPONENT_MANAGER_ID
    radio.component_id = radio_component_id
    radio_interface = radio.addInterface(
        "{}-sdr-interface".format(role)
    )

    radio_link = request.Link(
        "{}-sdr-link".format(role)
    )
    radio_link.bandwidth = 10 * 1000 * 1000
    radio_link.addInterface(compute_interface)
    radio_link.addInterface(radio_interface)


tx_radio, rx_radio = WORKBENCH_RADIOS[
    parameters.workbench
]

add_radio_pair("tx", tx_radio)
add_radio_pair("rx", rx_radio)

tour = ig.Tour()
tour.Description(ig.Tour.MARKDOWN, DESCRIPTION)
tour.Instructions(ig.Tour.MARKDOWN, INSTRUCTIONS)
request.addTour(tour)

context.printRequestRSpec(request)
