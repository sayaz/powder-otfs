#!/usr/bin/env python
"""POWDER profile for a selectable two-node indoor OTFS experiment."""

import geni.portal as portal
import geni.rspec.emulab.spectrum as spectrum
import geni.rspec.igext as ig
import geni.rspec.pg as rspec


DESCRIPTION = """
# POWDER-OTFS Indoor OTA

This profile creates a two-node SISO OTFS experiment in the POWDER indoor OTA
lab. The user selects one transmit radio and one receive radio from the four
indoor NI B210s and four indoor NI X310s.

Each B210 is USB-attached to its corresponding `ota-nuc` compute node. Each
X310 is connected through a dedicated 10 Gb/s link to a server-class compute
node.

The experiment requests the reserved 3550-3570 MHz spectrum range and installs
POWDER-OTFS and its dependencies automatically.
"""

INSTRUCTIONS = """
Wait until startup for both `tx` and `rx` reports `Finished`.

The project is available on both nodes at:

```bash
cd ~/powder-otfs
```

Start the receiver on `rx`:

```bash
cd ~/powder-otfs
python3 examples/ota/ota_rx.py
```

Then start the transmitter on `tx`:

```bash
cd ~/powder-otfs
python3 examples/ota/ota_tx.py
```

Each command automatically loads the radio type, device arguments, and center
frequency selected by this profile.
"""

COMPONENT_MANAGER_ID = (
    "urn:publicid:IDN+emulab.net+authority+cm"
)
UBUNTU_IMAGE = (
    "urn:publicid:IDN+emulab.net+image+"
    "emulab-ops//UBUNTU22-64-STD"
)
SETUP_SCRIPT = (
    "sudo bash /local/repository/"
    "scripts/powder/setup-rf-bench.sh"
)
CENTER_FREQUENCY_HZ = 3560000000

INDOOR_RADIOS = [
    ("ota-x310-1", "X310 #1"),
    ("ota-x310-2", "X310 #2"),
    ("ota-x310-3", "X310 #3"),
    ("ota-x310-4", "X310 #4"),
    ("ota-nuc1", "B210 #1 (ota-nuc1)"),
    ("ota-nuc2", "B210 #2 (ota-nuc2)"),
    ("ota-nuc3", "B210 #3 (ota-nuc3)"),
    ("ota-nuc4", "B210 #4 (ota-nuc4)"),
]

context = portal.Context()

context.defineParameter(
    name="tx_radio",
    description="Transmit USRP",
    typ=portal.ParameterType.STRING,
    defaultValue="ota-x310-1",
    legalValues=INDOOR_RADIOS,
)

context.defineParameter(
    name="rx_radio",
    description="Receive USRP",
    typ=portal.ParameterType.STRING,
    defaultValue="ota-x310-2",
    legalValues=INDOOR_RADIOS,
)

context.defineParameter(
    name="x310_compute_type",
    description="Compute-node type used with each selected X310",
    typ=portal.ParameterType.STRING,
    defaultValue="d740",
    legalValues=[
        ("d430", "Emulab d430"),
        ("d740", "Emulab d740"),
    ],
)

parameters = context.bindParameters()

if parameters.tx_radio == parameters.rx_radio:
    context.reportError(
        portal.ParameterError(
            "The TX and RX radios must be different devices.",
            ["tx_radio", "rx_radio"],
        )
    )

context.verifyParameters()
request = context.makeRequestRSpec()


def setup_command(role, radio_type):
    """Build the startup command for one selected radio."""

    return "{} {} {} {}".format(
        SETUP_SCRIPT,
        radio_type,
        role,
        CENTER_FREQUENCY_HZ,
    )


def add_x310(role, radio_component_id):
    """Add an X310 and its server-class compute node."""

    compute = request.RawPC(role)
    compute.component_manager_id = COMPONENT_MANAGER_ID
    compute.hardware_type = parameters.x310_compute_type
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
            command=setup_command(role, "x310"),
        )
    )

    radio = request.RawPC("{}-sdr".format(role))
    radio.component_manager_id = COMPONENT_MANAGER_ID
    radio.component_id = radio_component_id

    radio_link = request.Link(
        "{}-sdr-link".format(role)
    )
    radio_link.bandwidth = 10 * 1000 * 1000
    radio_link.addInterface(compute_interface)
    radio_link.addNode(radio)


def add_b210(role, nuc_component_id):
    """Add a NUC with its USB-attached B210."""

    nuc = request.RawPC(role)
    nuc.component_manager_id = COMPONENT_MANAGER_ID
    nuc.component_id = nuc_component_id
    nuc.disk_image = UBUNTU_IMAGE
    nuc.addService(
        rspec.Execute(
            shell="bash",
            command=setup_command(role, "b210"),
        )
    )


def add_radio(role, component_id):
    """Add the compute and radio resources selected for one role."""

    if component_id.startswith("ota-x310-"):
        add_x310(role, component_id)
    else:
        add_b210(role, component_id)


add_radio("tx", parameters.tx_radio)
add_radio("rx", parameters.rx_radio)

request.requestSpectrum(3550.0, 3570.0, 0)

tour = ig.Tour()
tour.Description(ig.Tour.MARKDOWN, DESCRIPTION)
tour.Instructions(ig.Tour.MARKDOWN, INSTRUCTIONS)
request.addTour(tour)

context.printRequestRSpec(request)
