# Copyright (c) 2024, NVIDIA CORPORATION.

import os
import pytest

from numba import cuda
from pynvjitlink.patch import Archive, Cubin, CUSource, Fatbin, Object, PTXSource


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
def device_functions_archive():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(test_dir, "test_device_functions.a")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def device_functions_cubin():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(test_dir, "test_device_functions.cubin")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def device_functions_cusource():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(test_dir, "test_device_functions.cu")
    with open(path, "r") as f:
        return f.read()


@pytest.fixture(scope="session")
def device_functions_fatbin():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(test_dir, "test_device_functions.fatbin")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def device_functions_object():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(test_dir, "test_device_functions.o")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def device_functions_ptx():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(test_dir, "test_device_functions.ptx")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def undefined_extern_cubin():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fatbin_path = os.path.join(test_dir, "undefined_extern.cubin")
    with open(fatbin_path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def linkable_code_archive(device_functions_archive):
    return Archive(device_functions_archive)


@pytest.fixture(scope="session")
def linkable_code_cubin(device_functions_cubin):
    return Cubin(device_functions_cubin)


@pytest.fixture(scope="session")
def linkable_code_cusource(device_functions_cusource):
    return CUSource(device_functions_cusource)


@pytest.fixture(scope="session")
def linkable_code_fatbin(device_functions_fatbin):
    return Fatbin(device_functions_fatbin)


@pytest.fixture(scope="session")
def linkable_code_object(device_functions_object):
    return Object(device_functions_object)


@pytest.fixture(scope="session")
def linkable_code_ptx(device_functions_ptx):
    return PTXSource(device_functions_ptx)
