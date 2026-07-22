#!/usr/bin/env bash

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

radio_type="${1:-x310}"
radio_role="${2:-unknown}"
center_frequency="${3:-3500000000}"

case "${radio_type}" in
    x310)
        device_args="addr=192.168.40.2"
        ;;
    b210)
        device_args="type=b200"
        ;;
    *)
        echo "Unsupported radio type: ${radio_type}" >&2
        exit 1
        ;;
esac

apt-get update
apt-get install -y \
    git \
    gnuradio \
    libuhd-dev \
    python3-matplotlib \
    python3-numpy \
    python3-pip \
    python3-uhd \
    rsync \
    uhd-host

sysctl -w net.core.wmem_max=24862979
sysctl -w net.core.rmem_max=24862979

sdr_interface="$(
    ip -o -4 addr show |
        awk '$4 ~ /^192[.]168[.]40[.]1[/]/ {print $2; exit}'
)"

if [[ -n "${sdr_interface}" ]]; then
    ip link set dev "${sdr_interface}" mtu 9000
    echo "Configured X310 network interface ${sdr_interface}."
else
    echo "No X310 network interface found; using USB-attached SDR."
fi

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

rm -rf "${project_dir}/.venv"

chown -R \
    "${project_user}:${project_group}" \
    "${project_dir}"

printf '%s\n' \
    '{' \
    "  \"role\": \"${radio_role}\"," \
    "  \"radio_type\": \"${radio_type}\"," \
    "  \"device_args\": \"${device_args}\"," \
    "  \"center_frequency\": ${center_frequency}" \
    '}' \
    > "${project_dir}/.powder-radio.json"

chown \
    "${project_user}:${project_group}" \
    "${project_dir}/.powder-radio.json"

bashrc="${project_home}/.bashrc"
old_activation_marker="# POWDER-OTFS automatic environment"
pythonpath_marker="# POWDER-OTFS Python path"

if grep -Fq "${old_activation_marker}" "${bashrc}"; then
    sed -i \
        "/${old_activation_marker}/,+3d" \
        "${bashrc}"
fi

if ! grep -Fq "${pythonpath_marker}" "${bashrc}"; then
    {
        echo
        echo "${pythonpath_marker}"
        echo "export PYTHONPATH=\"${project_dir}/src\${PYTHONPATH:+:\${PYTHONPATH}}\""
    } >> "${bashrc}"
fi

chown \
    "${project_user}:${project_group}" \
    "${bashrc}"

sudo -u "${project_user}" \
    env PYTHONPATH="${project_dir}/src" \
    python3 -c \
    "import numpy, powder_otfs, uhd"
