#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

rapids-conda-retry mambabuild conda/recipes/pynvjitlink

rapids-upload-conda-to-s3 python
