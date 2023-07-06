cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

enable_language(C CXX)

include(fetch_rapids.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-find)

rapids_cpm_init()

rapids_find_package(
  CUDAToolkit 12.2 REQUIRED
)

project(nvjitlink)

find_package(PythonExtensions REQUIRED)

add_library(_nvjitlinklib MODULE nvjitlink/_nvjitlinklib.cpp)
python_extension_module(_nvjitlinklib)

target_link_libraries(_nvjitlinklib CUDA::nvJitLink_static CUDA::nvptxcompiler_static)

if(WIN32)
  target_compile_options(_nvjitlinklib PRIVATE -Werror -Wall -/Od)
else()
  target_compile_options(_nvjitlinklib PRIVATE -Werror -Wall -O0)
endif()

install(TARGETS _nvjitlinklib LIBRARY DESTINATION nvjitlink)
