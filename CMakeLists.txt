# Copyright (c) 2023-2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
  pynvjitlink
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
