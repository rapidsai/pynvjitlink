# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import pytest

from nvjitlink import _nvjitlinklib
from nvjitlink.api import InputType


@pytest.fixture
def device_functions_cubin():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    cubin_path = os.path.join(test_dir, 'test_device_functions.cubin')
    with open(cubin_path, 'rb') as f:
        return f.read()


def test_create_no_arch_error():
    # nvjitlink expects at least the architecture to be specified.
    with pytest.raises(RuntimeError,
                       match='NVJITLINK_ERROR_MISSING_ARCH error'):
        _nvjitlinklib.create()


@pytest.mark.skip('Causes fatal error and exit(1)')
def test_invalid_arch_error():
    # sm_XX is not a valid architecture
    with pytest.raises(RuntimeError,
                       match='NVJITLINK_ERROR_UNRECOGNIZED_OPTION error'):
        _nvjitlinklib.create('-arch=sm_XX')


def test_unrecognized_option_error():
    with pytest.raises(RuntimeError,
                       match='NVJITLINK_ERROR_UNRECOGNIZED_OPTION error'):
        _nvjitlinklib.create('-fictitious_option')


def test_invalid_option_type_error():
    with pytest.raises(TypeError,
                       match='Expecting only strings'):
        _nvjitlinklib.create('-arch', 53)


def test_create_and_destroy():
    handle = _nvjitlinklib.create('-arch=sm_53')
    assert handle != 0
    _nvjitlinklib.destroy(handle)


def test_complete_empty():
    handle = _nvjitlinklib.create('-arch=sm_75')
    _nvjitlinklib.complete(handle)
    _nvjitlinklib.destroy(handle)


def test_add_file_cubin(device_functions_cubin):
    handle = _nvjitlinklib.create('-arch=sm_75')
    name = 'test_device_functions.cubin'
    _nvjitlinklib.add_data(handle, InputType.CUBIN.value,
                           device_functions_cubin, name)
    _nvjitlinklib.destroy(handle)
