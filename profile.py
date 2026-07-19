#!/usr/bin/env python
"""POWDER profile for a two-node OTFS conducted-RF experiment."""

import geni.portal as portal
import geni.rspec.igext as ig
import geni.rspec.pg as rspec


DESCRIPTION = """
# POWDER-OTFS Paired Radio Workbench

This profile allocates two server-class compute nodes and one POWDER paired
radio workbench. Each compute node connects to one NI X310 over a dedicated
10 Gb/s link. The X310 RF ports are connected through fixed 30 dB attenuators.

The profile installs UHD, GNU Radio, and the POWDER-OTFS Python package. It does
not install or start OAI.
"""

INSTRUCTIONS = """
Wait until the startup status for both `tx` and `rx` reports `Finished`.

Open a shell on each compute node and activate the project environment:

```bash
cd /local/repository
source .venv/bin/activate
```

Confirm radio access:

```bash
uhd_find_devices
uhd_usrp_probe --args "addr=192.168.40.2"
```

The intended roles are:

- `tx`: OTFS transmitter connected to `tx-sdr`
- `rx`: OTFS receiver connected to `rx-sdr`

Do not run transmitter code until the UHD checks succeed on both nodes.
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
