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

rapids-logger "Upload Wheel"
RAPIDS_PY_WHEEL_NAME="pynvjitlink-cu12" rapids-upload-wheels-to-s3 ./wheel-build
