#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION


rapids-logger "Install CUDA Toolkit"
yum update
yum install -y epel-release
yum install -y nvidia-driver-latest-dkms
yum install -y cuda-toolkit-12-3

rapids-logger "Install GCC"
yum install -y centos-release-scl
yum install -y devtoolset-9

source scl_source enable devtoolset-9

rapids-logger "Check conda environment"
conda list

rapids-logger "Build wheel"
export SCCACHE_S3_NO_CREDENTIALS=1
mkdir -p ./wheel-build
pip wheel . --wheel-dir=./wheel-build -vvv
