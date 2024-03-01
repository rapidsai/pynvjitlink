# Copyright (c) 2024, NVIDIA CORPORATION.

import os
import pytest

from pynvjitlink.patch import Archive, Cubin, Fatbin, Object, PTXSource


@pytest.fixture(scope="session")
def device_functions_archive():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    cubin_path = os.path.join(test_dir, "test_device_functions.a")
    with open(cubin_path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def device_functions_cubin():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    cubin_path = os.path.join(test_dir, "test_device_functions.cubin")
    with open(cubin_path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def device_functions_fatbin():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    fatbin_path = os.path.join(test_dir, "test_device_functions.fatbin")
    with open(fatbin_path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def device_functions_object():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    cubin_path = os.path.join(test_dir, "test_device_functions.o")
    with open(cubin_path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def device_functions_ptx():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    cubin_path = os.path.join(test_dir, "test_device_functions.ptx")
    with open(cubin_path, "rb") as f:
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
def linkable_code_fatbin(device_functions_fatbin):
    return Fatbin(device_functions_fatbin)


@pytest.fixture(scope="session")
def linkable_code_object(device_functions_object):
    return Object(device_functions_object)


@pytest.fixture(scope="session")
def linkable_code_ptx(device_functions_ptx):
    return PTXSource(device_functions_ptx)
