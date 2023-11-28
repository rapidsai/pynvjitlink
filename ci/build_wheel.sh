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
export SCCACHE_S3_NO_CREDENTIALS=1
mkdir -p ./wheel-build
pip wheel . --wheel-dir=./wheel-build -vvv

rapids-logger "Upload Wheel"
RAPIDS_BUILD_TYPE="branch" RAPIDS_REPOSITORY=rapidsai/pynvjitlink rapids-upload-to-s3 pynvjitlink-cu12 ./wheel-build/
