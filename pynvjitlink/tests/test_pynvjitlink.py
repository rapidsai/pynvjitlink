# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.

import pynvjitlink
import pytest
from pynvjitlink import _nvjitlinklib
from pynvjitlink.api import InputType


def test_create_no_arch_error():
    # nvjitlink expects at least the architecture to be specified.
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_MISSING_ARCH error"):
        _nvjitlinklib.create()


def test_invalid_arch_error():
    # sm_XX is not a valid architecture
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_UNRECOGNIZED_OPTION error"):
        _nvjitlinklib.create("-arch=sm_XX")


def test_unrecognized_option_error():
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_UNRECOGNIZED_OPTION error"):
        _nvjitlinklib.create("-fictitious_option")


def test_invalid_option_type_error():
    with pytest.raises(TypeError, match="Expecting only strings"):
        _nvjitlinklib.create("-arch", 53)


def test_create_and_destroy():
    handle = _nvjitlinklib.create("-arch=sm_53")
    assert handle != 0
    _nvjitlinklib.destroy(handle)


def test_complete_empty():
    handle = _nvjitlinklib.create("-arch=sm_75")
    _nvjitlinklib.complete(handle)
    _nvjitlinklib.destroy(handle)


@pytest.mark.parametrize(
    "input_file,input_type",
    [
        ("device_functions_cubin", InputType.CUBIN),
        ("device_functions_fatbin", InputType.FATBIN),
        ("device_functions_ptx", InputType.PTX),
        ("device_functions_object", InputType.OBJECT),
        ("device_functions_archive", InputType.LIBRARY),
    ],
)
def test_add_file(input_file, input_type, gpu_arch_flag, request):
    filename, data = request.getfixturevalue(input_file)

    handle = _nvjitlinklib.create(gpu_arch_flag)
    _nvjitlinklib.add_data(handle, input_type.value, data, filename)
    _nvjitlinklib.destroy(handle)


# We test the LTO input case separately as it requires the `-lto` flag. The
# OBJECT input type is used because the LTO-IR container is packaged in an ELF
# object when produced by NVCC.
def test_add_file_lto(device_functions_ltoir_object, gpu_arch_flag):
    filename, data = device_functions_ltoir_object

    handle = _nvjitlinklib.create(gpu_arch_flag, "-lto")
    _nvjitlinklib.add_data(handle, InputType.OBJECT.value, data, filename)
    _nvjitlinklib.destroy(handle)


def test_get_error_log(undefined_extern_cubin, gpu_arch_flag):
    handle = _nvjitlinklib.create(gpu_arch_flag)
    filename, data = undefined_extern_cubin
    input_type = InputType.CUBIN.value
    _nvjitlinklib.add_data(handle, input_type, data, filename)
    with pytest.raises(RuntimeError):
        _nvjitlinklib.complete(handle)
    error_log = _nvjitlinklib.get_error_log(handle)
    _nvjitlinklib.destroy(handle)
    assert (
        "Undefined reference to '_Z5undefff' "
        "in 'undefined_extern.cubin'" in error_log
    )


def test_get_info_log(device_functions_cubin, gpu_arch_flag, gpu_compute_capability):
    if gpu_compute_capability < (7, 5):
        pytest.skip(
            "CUDA 12.8 shows deprecations for devices older than compute capability 7.5"
        )
    handle = _nvjitlinklib.create(gpu_arch_flag)
    filename, data = device_functions_cubin
    input_type = InputType.CUBIN.value
    _nvjitlinklib.add_data(handle, input_type, data, filename)
    _nvjitlinklib.complete(handle)
    info_log = _nvjitlinklib.get_info_log(handle)
    _nvjitlinklib.destroy(handle)
    # Info log is empty
    assert "" == info_log


def test_get_linked_cubin(device_functions_cubin, gpu_arch_flag):
    handle = _nvjitlinklib.create(gpu_arch_flag)
    filename, data = device_functions_cubin
    input_type = InputType.CUBIN.value
    _nvjitlinklib.add_data(handle, input_type, data, filename)
    _nvjitlinklib.complete(handle)
    cubin = _nvjitlinklib.get_linked_cubin(handle)
    _nvjitlinklib.destroy(handle)

    # Just check we got something that looks like an ELF
    assert cubin[:4] == b"\x7fELF"


def test_get_linked_cubin_link_not_complete_error(
    device_functions_cubin, gpu_arch_flag
):
    handle = _nvjitlinklib.create(gpu_arch_flag)
    filename, data = device_functions_cubin
    input_type = InputType.CUBIN.value
    _nvjitlinklib.add_data(handle, input_type, data, filename)
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_INTERNAL error"):
        _nvjitlinklib.get_linked_cubin(handle)
    _nvjitlinklib.destroy(handle)


def test_get_linked_cubin_from_lto(device_functions_ltoir_object, gpu_arch_flag):
    filename, data = device_functions_ltoir_object
    # device_functions_ltoir_object is a host object containing a fatbin
    # containing an LTOIR container, because that is what NVCC produces when
    # LTO is requested. So we need to use the OBJECT input type, and the linker
    # retrieves the LTO IR from it because we passed the -lto flag.
    input_type = InputType.OBJECT.value
    handle = _nvjitlinklib.create(gpu_arch_flag, "-lto")
    _nvjitlinklib.add_data(handle, input_type, data, filename)
    _nvjitlinklib.complete(handle)
    cubin = _nvjitlinklib.get_linked_cubin(handle)
    _nvjitlinklib.destroy(handle)

    # Just check we got something that looks like an ELF
    assert cubin[:4] == b"\x7fELF"


def test_get_linked_ptx_from_lto(device_functions_ltoir_object, gpu_arch_flag):
    filename, data = device_functions_ltoir_object
    # device_functions_ltoir_object is a host object containing a fatbin
    # containing an LTOIR container, because that is what NVCC produces when
    # LTO is requested. So we need to use the OBJECT input type, and the linker
    # retrieves the LTO IR from it because we passed the -lto flag.
    input_type = InputType.OBJECT.value
    handle = _nvjitlinklib.create(gpu_arch_flag, "-lto", "-ptx")
    _nvjitlinklib.add_data(handle, input_type, data, filename)
    _nvjitlinklib.complete(handle)
    _nvjitlinklib.get_linked_ptx(handle)
    _nvjitlinklib.destroy(handle)


def test_get_linked_ptx_link_not_complete_error(
    device_functions_ltoir_object, gpu_arch_flag
):
    handle = _nvjitlinklib.create(gpu_arch_flag, "-lto", "-ptx")
    filename, data = device_functions_ltoir_object
    input_type = InputType.OBJECT.value
    _nvjitlinklib.add_data(handle, input_type, data, filename)
    with pytest.raises(RuntimeError, match="NVJITLINK_ERROR_INTERNAL error"):
        _nvjitlinklib.get_linked_ptx(handle)
    _nvjitlinklib.destroy(handle)


def test_package_version():
    assert pynvjitlink.__version__ is not None
    assert len(str(pynvjitlink.__version__)) > 0
