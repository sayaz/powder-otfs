#!/usr/bin/env bash

set -euo pipefail

uhd_version="4.7.0.0"
install_prefix="/opt/uhd-${uhd_version}"
source_root="/opt/src"
source_dir="${source_root}/uhd-${uhd_version}"
build_dir="${source_dir}/host/build"
environment_file="/etc/profile.d/powder-otfs-uhd-4.7.sh"

export DEBIAN_FRONTEND=noninteractive

apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libncurses-dev \
    libssl-dev \
    libudev-dev \
    libusb-1.0-0-dev \
    pkg-config \
    python3-dev \
    python3-mako \
    python3-numpy \
    python3-pybind11 \
    python3-requests \
    python3-ruamel.yaml \
    python3-setuptools

if [[ ! -x "${install_prefix}/bin/uhd_find_devices" ]]; then
    mkdir -p "${source_root}"
    rm -rf "${source_dir}"

    git clone \
        --branch "v${uhd_version}" \
        --depth 1 \
        https://github.com/EttusResearch/uhd.git \
        "${source_dir}"

    cmake \
        -S "${source_dir}/host" \
        -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
        -DENABLE_DPDK=OFF \
        -DENABLE_DOXYGEN=OFF \
        -DENABLE_EXAMPLES=ON \
        -DENABLE_MANUAL=OFF \
        -DENABLE_MAN_PAGES=OFF \
        -DENABLE_PYTHON_API=ON \
        -DENABLE_TESTS=OFF \
        -DENABLE_UTILS=ON \
        -DPYTHON_EXECUTABLE=/usr/bin/python3

    cmake \
        --build "${build_dir}" \
        --parallel "$(nproc)"

    cmake \
        --install "${build_dir}"
fi

python_directory="$(
    find "${install_prefix}" \
        -type f \
        -path '*/uhd/__init__.py' \
        -print \
        -quit |
        xargs -r dirname |
        xargs -r dirname
)"

if [[ -z "${python_directory}" ]]; then
    echo "UHD ${uhd_version} Python module was not installed." >&2
    exit 1
fi

cat > "${environment_file}" <<EOF
# Isolated UHD ${uhd_version} environment for POWDER N310 nodes.
export PATH="${install_prefix}/bin:\${PATH}"
export LD_LIBRARY_PATH="${install_prefix}/lib:${install_prefix}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export PYTHONPATH="${python_directory}\${PYTHONPATH:+:\${PYTHONPATH}}"
EOF

# shellcheck disable=SC1090
source "${environment_file}"

uhd_config_info --version
python3 -c \
    "import uhd; print('N310 UHD Python module:', uhd.__file__)"
