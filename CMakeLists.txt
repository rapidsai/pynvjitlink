cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
  pynvjitlink
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

Python_add_library(_nvjitlinklib MODULE pynvjitlink/_nvjitlinklib.cpp WITH_SOABI)

find_package(
  # Require CUDA 12.2 Update 2 to avoid nvjitlink bugs
  CUDAToolkit 12.2.140 REQUIRED
)

#set_target_properties(_nvjitlinklib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#target_include_directories(_nvjitlinklib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(_nvjitlinklib PRIVATE CUDA::nvptxcompiler_static)
#target_link_libraries(_nvjitlinklib PRIVATE optimized CUDA::nvJitLink)
target_link_libraries(_nvjitlinklib PRIVATE CUDA::nvJitLink)

#get_target_property(loc CUDA::nvJitLink_static LOCATION)
#get_target_property(loc2 CUDA::nvJitLink LOCATION)

#include(CMakePrintHelpers)
#cmake_print_variables(loc)
#cmake_print_variables(loc2)
#cmake_print_variables(CUDAToolkit_INCLUDE_DIRS)


#target_compile_options(_nvjitlinklib PRIVATE -Werror -Wall)

target_compile_features(_nvjitlinklib PRIVATE cxx_std_11)

install(TARGETS _nvjitlinklib LIBRARY DESTINATION pynvjitlink)
