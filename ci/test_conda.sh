#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
rapids-mamba-retry create -n test \
    c-compiler \
    cxx-compiler \
    cuda-nvcc \
    cuda-version=${RAPIDS_CUDA_VERSION%.*} \
    "numba>=0.58" \
    make \
    pytest \
    python=${RAPIDS_PY_VERSION}

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${PYTHON_CHANNEL}" \
  pynvjitlink

rapids-logger "Build Tests"
pushd test_binary_generation
make
popd

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest pynvjitlink"
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-pynvjitlink.xml" \
  -v

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
