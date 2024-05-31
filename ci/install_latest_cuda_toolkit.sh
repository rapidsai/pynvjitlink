#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

# Installs the latest CUDA Toolkit.
# Supports CentOS 7 and Rocky Linux 8.

yum update -y
yum install -y epel-release

OS_ID=$(. /etc/os-release; echo $ID)
if [ "${OS_ID}" == "rocky" ]; then
    yum install -y nvidia-driver
else
    echo "Error: OS not detected as Rocky Linux. Exiting."
    exit 1
fi

yum install -y cuda-toolkit-12-5
