#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

export CUDA_VERSION="$(cat pynvjitlink/CUDA_VERSION)"

rapids-conda-retry build \
    conda/recipes/pynvjitlink \
    --variants "{cuda_compiler_version: ["'"'"${CUDA_VERSION}"'"'"]}" \
;

rapids-upload-conda-to-s3 python
