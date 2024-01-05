#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION

set -euo pipefail

rapids-logger "Download Wheel"
RAPIDS_PY_WHEEL_NAME="pynvjitlink-cu12" rapids-download-wheels-from-s3 ./wheel-build/

rapids-logger "Install wheel"
pip install --find-links ./wheel-build pynvjitlink-cu12

rapids-logger "Build Tests"
cd test_binary_generation && make

rapids-logger "Run Tests"
cd ..
pip install pytest
pytest pynvjitlink/tests
