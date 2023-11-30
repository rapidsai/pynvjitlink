#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION

set -e

rapids-logger "Download Wheel"
rapids-download-wheels-from-s3 pynvjitlink ./wheel-build/

rapids-logger "Install wheel"
pip install --find-links ./wheel-build/ pynvjitlink-cu12

rapids-logger "Build Tests"
cd test_binary_generation && make

rapids-logger "Run Tests"
cd ..
pip install pytest
pytest pynvjitlink/tests
