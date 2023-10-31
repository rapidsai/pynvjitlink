#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION

# install CUDAToolKit version 12.2
apt-get update
apt-get -y install cuda-toolkit-12-2

conda list
pip wheel .
ls
