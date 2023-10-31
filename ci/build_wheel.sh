#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION

# install CUDAToolKit version 12.2
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit --override
