cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
  nvjitlink
  VERSION 0.1.0
  LANGUAGES C CXX
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

Python_add_library(_nvjitlinklib MODULE nvjitlink/_nvjitlinklib.cpp WITH_SOABI)

find_package(
  CUDAToolkit 12.0 REQUIRED
)
target_link_libraries(_nvjitlinklib PRIVATE CUDA::nvJitLink_static CUDA::nvptxcompiler_static)

if(WIN32)
  target_compile_options(_nvjitlinklib PRIVATE -Werror -Wall -/Od)
else()
  target_compile_options(_nvjitlinklib PRIVATE -Werror -Wall -O0)
endif()

install(TARGETS _nvjitlinklib LIBRARY DESTINATION nvjitlink)
