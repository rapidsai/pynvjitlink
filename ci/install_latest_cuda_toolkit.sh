#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION

# Installs the latest CUDA Toolkit.
# Supports Rocky Linux 8.

yum update -y
yum install -y epel-release

OS_ID=$(. /etc/os-release; echo $ID)
if [ "${OS_ID}" != "rocky" ]; then
    echo "Error: OS not detected as Rocky Linux. Exiting."
    exit 1
fi

CUDA_VERSION="$(cat pynvjitlink/CUDA_VERSION)"
export CUDA_VERSION
YUM_CUDA_VERSION="${CUDA_VERSION//./-}"
export YUM_CUDA_VERSION

yum install -y \
    cuda-nvcc-"$YUM_CUDA_VERSION" \
    cuda-cudart-devel-"$YUM_CUDA_VERSION" \
    cuda-driver-devel-"$YUM_CUDA_VERSION" \
    libnvjitlink-devel-"$YUM_CUDA_VERSION" \
;
