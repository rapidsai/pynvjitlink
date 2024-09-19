# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import os
import sys
from pathlib import Path

CONDA_PREFIX = os.environ["CONDA_PREFIX"]
NVJITLINK_DIR = Path(__file__).parent
PYTHON_VERSION = sys.version_info
PYTHON_INCLUDE_NAME = "python%d.%d" % PYTHON_VERSION

CONDA_INCLUDE_DIR = Path(CONDA_PREFIX, "include")
PYTHON_INCLUDE_DIR = Path(CONDA_INCLUDE_DIR, PYTHON_INCLUDE_NAME)
NVJITLINK_INCLUDE_DIR = Path(NVJITLINK_DIR, "include")
CUDA_INCLUDE_DIR = "/usr/local/cuda/include"

flags = [
    "--cuda-gpu-arch=sm_50",
    f"-I{CONDA_INCLUDE_DIR}",
    f"-I{CUDA_INCLUDE_DIR}",
    f"-I{NVJITLINK_INCLUDE_DIR}",
    f"-I{PYTHON_INCLUDE_DIR}",
]


def Settings(**kwargs):  # noqa: N802
    return {"flags": flags}


if __name__ == "__main__":
    print(Settings())
