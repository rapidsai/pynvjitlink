#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION

rapids-logger "Install CUDA Toolkit"
apt-get update
apt-get -y install cuda-toolkit-12-3

rapids-logger "Check conda environment"

conda list

rapids-logger "Build wheel"

pip wheel .

rapids-logger "Install wheel"

pip install pynvjitlink-0.1.0-cp310-cp310-manylinux_2_35_x86_64.whl

rapids-logger "Build tests"

cd test_binary_generation && make

rapids-logger "Run tests"

cd ..
conda install -y pytest
py.test pynvjitlink/tests
