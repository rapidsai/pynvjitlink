#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
rapids-mamba-retry create -n test \
    cuda-nvcc \
    cuda-nvrtc \
    cuda-version=${RAPIDS_CUDA_VERSION%.*} \
    "numba>=0.58" \
    python=${RAPIDS_PY_VERSION}

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-print-env

rapids-mamba-retry install \
  --channel "${PYTHON_CHANNEL}" \
  pynvjitlink

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Test Numba with patch"
python ci/run_patched_numba_tests.py numba.cuda.tests -v -m

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
