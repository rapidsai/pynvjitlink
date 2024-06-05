#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Install testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

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

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Test Numba with patch"
python ci/run_patched_numba_tests.py numba.cuda.tests -v -m

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
