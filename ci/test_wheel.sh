#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION

set -euo pipefail

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
python -m pip install \
    "numba>=0.58" \
    psutil \
    pytest

rapids-logger "Download Wheel"
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pynvjitlink_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist/

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "Install wheel"
python -m pip install --find-links ./dist pynvjitlink${PACKAGE_CUDA_SUFFIX}

rapids-logger "Build Tests"
pushd test_binary_generation
export GPU_CC=`nvidia-smi --query-gpu=compute_cap --format=csv | grep -v compute_cap | sed 's/\.//'
make
popd

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

rapids-logger "Run Tests"
pushd pynvjitlink/tests
python -m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-pynvjitlink.xml" \
  -v
popd
