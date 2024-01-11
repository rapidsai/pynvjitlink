#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION

set -euo pipefail

rapids-logger "Install CUDA Toolkit"
source "$(dirname "$0")/install_latest_cuda_toolkit.sh"

rapids-logger "Build wheel"
mkdir -p ./wheel-build
pip wheel . --wheel-dir=./wheel-build -vvv

rapids-logger "Upload Wheel"
RAPIDS_PY_WHEEL_NAME="pynvjitlink-cu12" rapids-upload-wheels-to-s3 ./wheel-build
