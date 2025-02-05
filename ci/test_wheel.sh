#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION

set -euo pipefail

rapids-logger "Download Wheel"
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pynvjitlink_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist/

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "Install wheel"
rapids-pip-retry install $(echo ./dist/pynvjitlink_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]

rapids-logger "Build Tests"
pushd test_binary_generation
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
