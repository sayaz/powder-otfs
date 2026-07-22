import json

from powder_otfs.ota.runtime import load_radio_runtime_config


def test_load_x310_runtime_config(tmp_path) -> None:
    path = tmp_path / "radio.json"
    path.write_text(
        json.dumps(
            {
                "role": "tx",
                "radio_type": "x310",
                "device_args": "addr=192.168.40.2",
                "center_frequency": 3.56e9,
            }
        ),
        encoding="utf-8",
    )

    config = load_radio_runtime_config(path)

    assert config.role == "tx"
    assert config.radio_type == "x310"
    assert config.device_args == "addr=192.168.40.2"
    assert config.center_frequency == 3.56e9


def test_load_b210_runtime_config(tmp_path) -> None:
    path = tmp_path / "radio.json"
    path.write_text(
        json.dumps(
            {
                "role": "rx",
                "radio_type": "b210",
                "device_args": "type=b200",
                "center_frequency": 3.56e9,
            }
        ),
        encoding="utf-8",
    )

    config = load_radio_runtime_config(path)

    assert config.role == "rx"
    assert config.radio_type == "b210"
    assert config.device_args == "type=b200"
    assert config.center_frequency == 3.56e9
