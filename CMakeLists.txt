cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
  pynvjitlink
  VERSION 0.1.0
  LANGUAGES CXX
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

Python_add_library(_nvjitlinklib MODULE pynvjitlink/_nvjitlinklib.cpp WITH_SOABI)

find_package(
  CUDAToolkit 12.0 REQUIRED
)
target_link_libraries(_nvjitlinklib PRIVATE CUDA::nvJitLink_static CUDA::nvptxcompiler_static)

set(opt_option -O0)
if(WIN32)
  set(opt_option -/Od)
endif()
target_compile_options(_nvjitlinklib PRIVATE -Werror -Wall ${opt_option})

install(TARGETS _nvjitlinklib LIBRARY DESTINATION pynvjitlink)
