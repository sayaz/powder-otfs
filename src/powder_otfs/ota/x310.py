import time

import numpy as np
import uhd


def configure_x310_tx(
    device_address: str,
    sample_rate: float,
    center_frequency: float,
    gain: float,
    channel: int = 0,
    antenna: str = "TX/RX",
) -> uhd.usrp.MultiUSRP:
    """Configure an X310 transmitter."""

    usrp = uhd.usrp.MultiUSRP(
        f"addr={device_address}"
    )

    usrp.set_clock_source("external")
    usrp.set_time_source("external")
    usrp.set_tx_rate(sample_rate, channel)
    usrp.set_tx_freq(
        uhd.types.TuneRequest(center_frequency),
        channel,
    )
    usrp.set_tx_gain(gain, channel)
    usrp.set_tx_antenna(antenna, channel)

    time.sleep(1.0)

    return usrp


def configure_x310_rx(
    device_address: str,
    sample_rate: float,
    center_frequency: float,
    gain: float,
    channel: int = 0,
    antenna: str = "RX2",
) -> uhd.usrp.MultiUSRP:
    """Configure an X310 receiver."""

    usrp = uhd.usrp.MultiUSRP(
        f"addr={device_address}"
    )

    usrp.set_clock_source("external")
    usrp.set_time_source("external")
    usrp.set_rx_rate(sample_rate, channel)
    usrp.set_rx_freq(
        uhd.types.TuneRequest(center_frequency),
        channel,
    )
    usrp.set_rx_gain(gain, channel)
    usrp.set_rx_antenna(antenna, channel)

    time.sleep(1.0)

    return usrp


def transmit_waveform(
    usrp: uhd.usrp.MultiUSRP,
    waveform: np.ndarray,
    channel: int = 0,
) -> int:
    """Transmit one complex waveform through an X310."""

    stream_args = uhd.usrp.StreamArgs(
        "fc32",
        "sc16",
    )
    stream_args.channels = [channel]

    tx_streamer = usrp.get_tx_stream(
        stream_args
    )

    samples = np.asarray(
        waveform,
        dtype=np.complex64,
    ).reshape(1, -1)

    metadata = uhd.types.TXMetadata()
    metadata.start_of_burst = True
    metadata.end_of_burst = False

    sent = tx_streamer.send(
        samples,
        metadata,
    )

    metadata.start_of_burst = False
    metadata.end_of_burst = True

    tx_streamer.send(
        np.empty((1, 0), dtype=np.complex64),
        metadata,
    )

    return int(sent)


def receive_samples(
    usrp: uhd.usrp.MultiUSRP,
    num_samples: int,
    channel: int = 0,
    timeout: float = 5.0,
) -> np.ndarray:
    """Capture a fixed number of complex samples from an X310."""

    stream_args = uhd.usrp.StreamArgs(
        "fc32",
        "sc16",
    )
    stream_args.channels = [channel]

    rx_streamer = usrp.get_rx_stream(
        stream_args
    )

    metadata = uhd.types.RXMetadata()

    stream_command = uhd.types.StreamCMD(
        uhd.types.StreamMode.num_done
    )
    stream_command.num_samps = num_samples
    stream_command.stream_now = True

    rx_streamer.issue_stream_cmd(
        stream_command
    )

    received = np.empty(
        num_samples,
        dtype=np.complex64,
    )

    offset = 0

    while offset < num_samples:
        buffer_size = min(
            rx_streamer.get_max_num_samps(),
            num_samples - offset,
        )

        buffer = np.empty(
            (1, buffer_size),
            dtype=np.complex64,
        )

        count = rx_streamer.recv(
            buffer,
            metadata,
            timeout,
        )

        if metadata.error_code != (
            uhd.types.RXMetadataErrorCode.none
        ):
            raise RuntimeError(
                f"X310 receive error: {metadata.strerror()}"
            )

        if count == 0:
            raise TimeoutError(
                "No samples were received from the X310."
            )

        received[offset:offset + count] = (
            buffer[0, :count]
        )
        offset += count

    return received