# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import sys
import os

from nvjitlink import NvJitLinker, NvJitLinkError


@pytest.fixture
def device_functions_cubin():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    cubin_path = os.path.join(test_dir, 'test_device_functions.cubin')
    with open(cubin_path, 'rb') as f:
        return f.read()


@pytest.fixture
def device_functions_fatbin():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fatbin_path = os.path.join(test_dir, 'test_device_functions.fatbin')
    with open(fatbin_path, 'rb') as f:
        return f.read()


@pytest.fixture
def undefined_extern_cubin():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fatbin_path = os.path.join(test_dir, 'undefined_extern.cubin')
    with open(fatbin_path, 'rb') as f:
        return f.read()

@pytest.mark.skip
def test_create_no_arch_error():
    # nvlink expects at least the architecture to be specified.
    with pytest.raises(NvJitLinkError,
                       match='NVJITLINK_ERROR_MISSING_ARCH error'):
        NvJitLinker()

@pytest.mark.skip
def test_invalid_arch_error():
    # sm_XX is not a valid architecture
    with pytest.raises(NvJitLinkError,
                       match='NVJITLINK_ERROR_UNRECOGNIZED_OPTION error'):
        NvJitLinker('-arch=sm_XX')

@pytest.mark.skip
def test_invalid_option_type_error():
    with pytest.raises(TypeError,
                       match='Expecting only strings'):
        NvJitLinker('-arch', 53)


def test_create_and_destroy():
    nvjitlinker = NvJitLinker('-arch=sm_53')
    assert nvjitlinker.handle != 0


def test_add_cubin(device_functions_cubin):
    nvjitlinker = NvJitLinker('-arch=sm_75')
    name = 'test_device_functions.cubin'
    nvjitlinker.add_cubin(device_functions_cubin, name)

@pytest.mark.skip
def test_add_incompatible_cubin_arch_error(device_functions_cubin):
    nvjitlinker = NvJitLinker('-arch=sm_70')
    name = 'test_device_functions.cubin'
    with pytest.raises(NvJitLinkError,
                       match='NVJITLINK_ERROR_INVALID_INPUT error'):
        nvjitlinker.add_cubin(device_functions_cubin, name)


def test_add_fatbin_sm75(device_functions_fatbin):
    nvjitlinker = NvJitLinker('-arch=sm_75')
    name = 'test_device_functions.fatbin'
    nvjitlinker.add_fatbin(device_functions_fatbin, name)


def test_add_fatbin_sm70(device_functions_fatbin):
    nvjitlinker = NvJitLinker('-arch=sm_70')
    name = 'test_device_functions.fatbin'
    nvjitlinker.add_fatbin(device_functions_fatbin, name)

@pytest.mark.skip
def test_add_incompatible_fatbin_arch_error(device_functions_fatbin):
    nvjitlinker = NvJitLinker('-arch=sm_80')
    name = 'test_device_functions.fatbin'
    with pytest.raises(NvJitLinkError,
                       match='NVJITLINK_ERROR_INVALID_INPUT error'):
        nvjitlinker.add_fatbin(device_functions_fatbin, name)

@pytest.mark.skip
def test_add_cubin_with_fatbin_error(device_functions_fatbin):
    nvjitlinker = NvJitLinker('-arch=sm_75')
    name = 'test_device_functions.fatbin'
    with pytest.raises(NvJitLinkError,
                       match='NVJITLINK_ERROR_INVALID_INPUT error'):
        nvjitlinker.add_cubin(device_functions_fatbin, name)


def test_add_fatbin_with_cubin(device_functions_cubin):
    # Adding a cubin with add_fatbin seems to work - this may be expected
    # behaviour.
    nvjitlinker = NvJitLinker('-arch=sm_75')
    name = 'test_device_functions.cubin'
    nvjitlinker.add_fatbin(device_functions_cubin, name)

@pytest.mark.skip
def test_duplicate_symbols_cubin_and_fatbin(device_functions_cubin,
                                            device_functions_fatbin):
    # This link errors because the cubin and the fatbin contain the same
    # symbols.
    nvjitlinker = NvJitLinker('-arch=sm_75')
    name = 'test_device_functions.cubin'
    nvjitlinker.add_cubin(device_functions_cubin, name)
    name = 'test_device_functions.fatbin'
    with pytest.raises(NvJitLinkError,
                       match="NVJITLINK_ERROR_INVALID_INPUT error"):
        nvjitlinker.add_fatbin(device_functions_fatbin, name)


def test_get_linked_cubin_complete_empty_error():
    nvjitlinker = NvJitLinker('-arch=sm_75')
    cubin = nvjitlinker.get_linked_cubin()

    # Linking nothing still gives us an empty ELF back
    assert cubin[:4] == b'\x7fELF'


def test_get_linked_cubin(device_functions_cubin):
    nvjitlinker = NvJitLinker('-arch=sm_75')
    name = 'test_device_functions.cubin'
    nvjitlinker.add_cubin(device_functions_cubin, name)
    cubin = nvjitlinker.get_linked_cubin()

    # Just check we got something that looks like an ELF
    assert cubin[:4] == b'\x7fELF'

@pytest.mark.skip
def test_get_error_log(undefined_extern_cubin):
    nvjitlinker = NvJitLinker('-arch=sm_75')
    name = 'undefined_extern.cubin'
    nvjitlinker.add_cubin(undefined_extern_cubin, name)
    with pytest.raises(NvJitLinkError):
        nvjitlinker.get_linked_cubin()
    error_log = nvjitlinker.error_log
    assert "Undefined reference to '_Z5undefff'" in error_log


def test_get_info_log(device_functions_cubin):
    nvjitlinker = NvJitLinker('-arch=sm_75')
    name = 'test_device_functions.cubin'
    nvjitlinker.add_cubin(device_functions_cubin, name)
    nvjitlinker.get_linked_cubin()
    info_log = nvjitlinker.info_log
    # Info log is empty
    assert "" == info_log


if __name__ == '__main__':
    sys.exit(pytest.main())
