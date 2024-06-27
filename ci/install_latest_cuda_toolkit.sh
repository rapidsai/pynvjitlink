#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

# Installs the latest CUDA Toolkit.
# Supports Rocky Linux 8.

yum update -y
yum install -y epel-release

OS_ID=$(. /etc/os-release; echo $ID)
if [ "${OS_ID}" != "rocky" ]; then
    echo "Error: OS not detected as Rocky Linux. Exiting."
    exit 1
fi

yum install -y nvidia-driver cuda-nvcc-12-5 cuda-cudart-devel-12-5 cuda-driver-devel-12-5 libnvjitlink-devel-12-5
