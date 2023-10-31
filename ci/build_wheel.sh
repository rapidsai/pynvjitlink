#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION

# install CUDAToolKit version 12.2
apt-get update
apt-get -y install cuda-toolkit-12-2

rapids-logger "Check conda environment"

conda list

rapids-logger "Building wheel"

pip wheel .

rapids-logger "Installing wheel"

pip install pynvjitlink-0.1.0-cp310-cp310-manylinux_2_35_x86_64.whl

rapids-logger "Building tests"

cd test_binary_generation && make

rapids-logger "Running Tests"

cd ..
conda install -y pytest
py.test pynvjitlink/tests
