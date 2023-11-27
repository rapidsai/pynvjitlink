#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION

set -e

rapids-logger "Search artifact directory"
ls ./artifact

rapids-logger "Install wheel"
pip install --find-links ./artifact pynvjitlink-cu12

rapids-logger "Build Tests"
cd test_binary_generation && make

rapids-logger "Run Tests"
cd ..
pip install pytest
pytest pynvjitlink/tests
