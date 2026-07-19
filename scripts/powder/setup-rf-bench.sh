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
    rsync \
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

project_user="$(
    stat -c '%U' /local/repository
)"
project_group="$(
    id -gn "${project_user}"
)"
project_home="/users/${project_user}"
project_dir="${project_home}/powder-otfs"

mkdir -p "${project_dir}"

rsync -a \
    --exclude ".venv/" \
    --exclude "results/" \
    /local/repository/ \
    "${project_dir}/"

chown -R \
    "${project_user}:${project_group}" \
    "${project_dir}"

sudo -u "${project_user}" \
    python3 -m venv \
    --system-site-packages \
    "${project_dir}/.venv"

sudo -u "${project_user}" \
    "${project_dir}/.venv/bin/python" -m pip install \
    --upgrade \
    pip \
    setuptools \
    wheel

sudo -u "${project_user}" \
    "${project_dir}/.venv/bin/python" -m pip install \
    "${project_dir}"

bashrc="${project_home}/.bashrc"
activation_marker="# POWDER-OTFS automatic environment"

if ! grep -Fq "${activation_marker}" "${bashrc}"; then
    {
        echo
        echo "${activation_marker}"
        echo "if [ -f \"${project_dir}/.venv/bin/activate\" ]; then"
        echo "    source \"${project_dir}/.venv/bin/activate\""
        echo "fi"
    } >> "${bashrc}"
fi

chown \
    "${project_user}:${project_group}" \
    "${bashrc}"
