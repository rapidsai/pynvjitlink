# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import os

import pytest
from numba import cuda
from pynvjitlink.patch import LTOIR, Archive, Cubin, CUSource, Fatbin, Object, PTXSource


@pytest.fixture(scope="session")
def gpu_compute_capability():
    """Compute capability of the current GPU"""
    return cuda.get_current_device().compute_capability


@pytest.fixture(scope="session")
def alt_gpu_compute_capability(gpu_compute_capability):
    """A compute capability that does not match the current GPU"""
    # For sufficient incompatibility for the test suites, the major number of
    # the compute capabilities must differ (for example one can load a 7.0
    # cubin when linking for 7.5)
    if gpu_compute_capability[0] == 7:
        return (8, 0)
    else:
        return (7, 0)


@pytest.fixture(scope="session")
def absent_gpu_compute_capability(gpu_compute_capability, alt_gpu_compute_capability):
    """A compute capability not used in any cubin or fatbin test binary"""
    cc_majors = {6, 7, 8, 9, 10, 12}
    cc_majors.remove(gpu_compute_capability[0])
    cc_majors.remove(alt_gpu_compute_capability[0])
    return (cc_majors.pop(), 0)


@pytest.fixture(scope="session")
def gpu_arch_flag(gpu_compute_capability):
    """nvJitLink arch flag to link for the current GPU"""
    major, minor = gpu_compute_capability
    return f"-arch=sm_{major}{minor}"


@pytest.fixture(scope="session")
def alt_gpu_arch_flag(alt_gpu_compute_capability):
    """nvJitLink arch flag to link for a different kind of GPU to the current
    one"""
    major, minor = alt_gpu_compute_capability
    return f"-arch=sm_{major}{minor}"


@pytest.fixture(scope="session")
def absent_gpu_arch_flag(absent_gpu_compute_capability):
    """nvJitLink arch flag to link for an architecture not in any cubin or
    fatbin"""
    major, minor = absent_gpu_compute_capability
    return f"-arch=sm_{major}{minor}"


def read_test_file(filename):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(test_dir, filename)
    with open(path, "rb") as f:
        return filename, f.read()


@pytest.fixture(scope="session")
def device_functions_cusource():
    return read_test_file("test_device_functions.cu")


@pytest.fixture(scope="session")
def device_functions_cubin():
    return read_test_file("test_device_functions.cubin")


@pytest.fixture(scope="session")
def device_functions_fatbin():
    return read_test_file("test_device_functions.fatbin")


@pytest.fixture(scope="session")
def device_functions_ltoir():
    return read_test_file("test_device_functions.ltoir")


@pytest.fixture(scope="session")
def device_functions_ltoir_object():
    return read_test_file("test_device_functions.ltoir.o")


@pytest.fixture(scope="session")
def device_functions_object():
    return read_test_file("test_device_functions.o")


@pytest.fixture(scope="session")
def device_functions_archive():
    return read_test_file("test_device_functions.a")


@pytest.fixture(scope="session")
def device_functions_ptx():
    return read_test_file("test_device_functions.ptx")


@pytest.fixture(scope="session")
def undefined_extern_cubin():
    return read_test_file("undefined_extern.cubin")


@pytest.fixture(scope="session")
def linkable_code_archive(device_functions_archive):
    name, data = device_functions_archive
    return Archive(data, name=name)


@pytest.fixture(scope="session")
def linkable_code_cubin(device_functions_cubin):
    name, data = device_functions_cubin
    return Cubin(data, name=name)


@pytest.fixture(scope="session")
def linkable_code_cusource(device_functions_cusource):
    name, data = device_functions_cusource
    return CUSource(data, name=name)


@pytest.fixture(scope="session")
def linkable_code_fatbin(device_functions_fatbin):
    name, data = device_functions_fatbin
    return Fatbin(data, name=name)


@pytest.fixture(scope="session")
def linkable_code_object(device_functions_object):
    name, data = device_functions_object
    return Object(data, name=name)


@pytest.fixture(scope="session")
def linkable_code_ptx(device_functions_ptx):
    name, data = device_functions_ptx
    return PTXSource(data, name=name)


@pytest.fixture(scope="session")
def linkable_code_ltoir(device_functions_ltoir):
    name, data = device_functions_ltoir
    return LTOIR(data, name=name)
