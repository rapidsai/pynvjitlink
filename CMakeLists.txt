cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
  pynvjitlink
  VERSION 0.1.6
  LANGUAGES CXX CUDA
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

Python_add_library(_nvjitlinklib MODULE pynvjitlink/_nvjitlinklib.cpp WITH_SOABI)

find_package(
  CUDAToolkit 12.3 REQUIRED
)
target_link_libraries(_nvjitlinklib PRIVATE CUDA::nvJitLink_static CUDA::nvptxcompiler_static)

target_compile_options(_nvjitlinklib PRIVATE -Werror -Wall)

target_compile_features(_nvjitlinklib PRIVATE cxx_std_11)

install(TARGETS _nvjitlinklib LIBRARY DESTINATION pynvjitlink)
