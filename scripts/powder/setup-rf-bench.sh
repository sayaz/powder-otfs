#!/usr/bin/env bash

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
    git \
    gnuradio \
    libuhd-dev \
    python3-pip \
    python3-uhd \
    python3-venv \
    uhd-host

sysctl -w net.core.wmem_max=24862979
sysctl -w net.core.rmem_max=24862979

sdr_interface="$(
    ip -o -4 addr show |
        awk '$4 ~ /^192[.]168[.]40[.]1[/]/ {print $2; exit}'
)"

if [[ -z "${sdr_interface}" ]]; then
    echo "Could not find the 192.168.40.1 SDR interface." >&2
    exit 1
fi

ip link set dev "${sdr_interface}" mtu 9000

cd /local/repository

python3 -m venv \
    --system-site-packages \
    .venv

.venv/bin/python -m pip install \
    --upgrade \
    pip \
    setuptools \
    wheel

.venv/bin/python -m pip install .