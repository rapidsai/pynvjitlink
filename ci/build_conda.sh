#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION

set -euo pipefail

conda config --set channel_priority strict

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

CUDA_VERSION="$(cat pynvjitlink/CUDA_VERSION)"
export CUDA_VERSION

cat > cuda_compiler_version.yaml << EOF
cuda_compiler_version:
  - "${CUDA_VERSION}"
EOF

sccache --zero-stats

rapids-conda-retry build \
    conda/recipes/pynvjitlink \
    -m cuda_compiler_version.yaml \
;

sccache --show-adv-stats

rapids-upload-conda-to-s3 python
