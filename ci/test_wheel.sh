#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION

set -e

rapids-logger "Install CUDA Toolkit"
yum update
yum install -y epel-release
yum install -y nvidia-driver-latest-dkms
yum install -y cuda-toolkit-12-3


rapids-logger "Install Python"

python_version="${1:-3.10}"
conda install -y python="$python_version"

rapids-logger "Check conda environment"

conda list

rapids-logger "Install wheel"
pip install  --find-links ./artifact pynvjitlink

rapids-logger "Build Tests"
cd test_binary_generation && make

rapids-logger "Run Tests"
cd ..
conda install -y pytest
py.test pynvjitlink/tests
