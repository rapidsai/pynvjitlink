# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.

import sys

import pytest
from pynvjitlink import NvJitLinker, NvJitLinkError


def test_create_no_arch_error():
    # nvlink expects at least the architecture to be specified.
    with pytest.raises(NvJitLinkError, match="NVJITLINK_ERROR_MISSING_ARCH error"):
        NvJitLinker()


def test_invalid_arch_error():
    # sm_XX is not a valid architecture
    with pytest.raises(
        NvJitLinkError, match="NVJITLINK_ERROR_UNRECOGNIZED_OPTION error"
    ):
        NvJitLinker("-arch=sm_XX")


def test_invalid_option_type_error():
    with pytest.raises(TypeError, match="Expecting only strings"):
        NvJitLinker("-arch", 53)


def test_create_and_destroy():
    nvjitlinker = NvJitLinker("-arch=sm_53")
    assert nvjitlinker.handle != 0


def test_add_cubin(device_functions_cubin, gpu_arch_flag):
    nvjitlinker = NvJitLinker(gpu_arch_flag)
    name, cubin = device_functions_cubin
    nvjitlinker.add_cubin(cubin, name)


def test_add_incompatible_cubin_arch_error(device_functions_cubin, alt_gpu_arch_flag):
    nvjitlinker = NvJitLinker(alt_gpu_arch_flag)
    name, cubin = device_functions_cubin
    with pytest.raises(NvJitLinkError, match="NVJITLINK_ERROR_INVALID_INPUT error"):
        nvjitlinker.add_cubin(cubin, name)


def test_add_fatbin_arch_1(device_functions_fatbin, gpu_arch_flag):
    nvjitlinker = NvJitLinker(gpu_arch_flag)
    name, fatbin = device_functions_fatbin
    nvjitlinker.add_fatbin(fatbin, name)


def test_add_fatbin_arch_2(device_functions_fatbin, alt_gpu_arch_flag):
    nvjitlinker = NvJitLinker(alt_gpu_arch_flag)
    name, fatbin = device_functions_fatbin
    nvjitlinker.add_fatbin(fatbin, name)


def test_add_nonexistent_fatbin_arch_error(
    device_functions_fatbin, absent_gpu_arch_flag
):
    nvjitlinker = NvJitLinker(absent_gpu_arch_flag)
    name, fatbin = device_functions_fatbin
    with pytest.raises(NvJitLinkError, match="NVJITLINK_ERROR_INVALID_INPUT error"):
        nvjitlinker.add_fatbin(fatbin, name)


def test_add_cubin_with_fatbin_error(device_functions_fatbin, gpu_arch_flag):
    nvjitlinker = NvJitLinker(gpu_arch_flag)
    name, fatbin = device_functions_fatbin
    with pytest.raises(NvJitLinkError, match="NVJITLINK_ERROR_INVALID_INPUT error"):
        nvjitlinker.add_cubin(fatbin, name)


def test_add_fatbin_with_cubin_error(device_functions_cubin, gpu_arch_flag):
    nvjitlinker = NvJitLinker(gpu_arch_flag)
    name, cubin = device_functions_cubin
    with pytest.raises(NvJitLinkError, match="NVJITLINK_ERROR_INVALID_INPUT error"):
        nvjitlinker.add_fatbin(cubin, name)


@pytest.mark.skip(
    reason="CUDA 12.9 nvjitlink performs invalid reads under this error scenario"
)
def test_duplicate_symbols_cubin_and_fatbin(
    device_functions_cubin, device_functions_fatbin, gpu_arch_flag
):
    # This link errors because the cubin and the fatbin contain the same
    # symbols.
    nvjitlinker = NvJitLinker(gpu_arch_flag)
    name, cubin = device_functions_cubin
    nvjitlinker.add_cubin(cubin, name)
    name, fatbin = device_functions_fatbin
    with pytest.raises(NvJitLinkError, match="NVJITLINK_ERROR_INVALID_INPUT error"):
        nvjitlinker.add_fatbin(fatbin, name)


def test_get_linked_cubin_complete_empty_error():
    nvjitlinker = NvJitLinker("-arch=sm_75")
    cubin = nvjitlinker.get_linked_cubin()

    # Linking nothing still gives us an empty ELF back
    assert cubin[:4] == b"\x7fELF"


def test_get_linked_cubin(device_functions_cubin, gpu_arch_flag):
    nvjitlinker = NvJitLinker(gpu_arch_flag)
    name, cubin = device_functions_cubin
    nvjitlinker.add_cubin(cubin, name)
    cubin = nvjitlinker.get_linked_cubin()

    # Just check we got something that looks like an ELF
    assert cubin[:4] == b"\x7fELF"


def test_get_error_log(undefined_extern_cubin, gpu_arch_flag):
    nvjitlinker = NvJitLinker(gpu_arch_flag)
    name, cubin = undefined_extern_cubin
    nvjitlinker.add_cubin(cubin, name)
    with pytest.raises(NvJitLinkError):
        nvjitlinker.get_linked_cubin()
    error_log = nvjitlinker.error_log
    assert "Undefined reference to '_Z5undefff'" in error_log


def test_get_info_log(device_functions_cubin, gpu_arch_flag, gpu_compute_capability):
    if gpu_compute_capability < (7, 5):
        pytest.skip(
            "CUDA 12.8 shows deprecations for devices older than compute capability 7.5"
        )
    nvjitlinker = NvJitLinker(gpu_arch_flag)
    name, cubin = device_functions_cubin
    nvjitlinker.add_cubin(cubin, name)
    nvjitlinker.get_linked_cubin()
    info_log = nvjitlinker.info_log
    # Info log is empty
    assert "" == info_log


if __name__ == "__main__":
    sys.exit(pytest.main())
