#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION

set -e

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
