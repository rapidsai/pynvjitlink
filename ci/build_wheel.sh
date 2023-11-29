#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION


rapids-logger "Install CUDA Toolkit"
yum update
yum install -y epel-release
yum install -y nvidia-driver-latest-dkms
yum install -y cuda-toolkit-12-3

rapids-logger "Check conda environment"
conda list

rapids-logger "Build wheel"
mkdir -p ./wheel-build
pip wheel . --wheel-dir=./wheel-build -vvv

WHEEL_PATH=$(find "./wheel-build" -type f -name "*.whl")
echo "WHEEL_PATH=${WHEEL_PATH}"

rapids-logger "Upload Wheel"
RAPIDS_BUILD_TYPE="branch" rapids-upload-to-s3 pynvjitlink-cu12 ${WHEEL_PATH}
