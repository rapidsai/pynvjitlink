# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import sys
from unittest.mock import patch as mock_patch

import pytest
from numba import cuda
from pynvjitlink import NvJitLinkError, patch
from pynvjitlink.patch import (
    PatchedLinker,
    _numba_version_ok,
    new_patched_linker,
    patch_numba_linker,
    required_numba_ver,
)


def test_numba_patching_numba_not_ok():
    with mock_patch.multiple(patch, _numba_version_ok=False, _numba_error="<error>"):
        with pytest.raises(RuntimeError, match="Cannot patch Numba: <error>"):
            patch_numba_linker()


@pytest.mark.skipif(
    not _numba_version_ok,
    reason=f"Requires Numba == {required_numba_ver[0]}.{required_numba_ver[1]}",
)
def test_numba_patching():
    # We import the linker here rather than at the top level because the import
    # may fail if if Numba is not present or an unsupported version.
    from numba.cuda.cudadrv.driver import Linker

    patch_numba_linker()
    assert Linker.new.func is new_patched_linker


def test_create():
    patched_linker = PatchedLinker(cc=(7, 5))
    assert "-arch=sm_75" in patched_linker.options


def test_create_no_cc_error():
    # nvJitLink expects at least the architecture to be specified.
    with pytest.raises(RuntimeError, match="PatchedLinker requires CC to be specified"):
        PatchedLinker()


def test_invalid_arch_error():
    # CC 0.0 is not a valid compute capability
    with pytest.raises(
        NvJitLinkError, match="NVJITLINK_ERROR_UNRECOGNIZED_OPTION error"
    ):
        PatchedLinker(cc=(0, 0))


def test_invalid_cc_type_error():
    with pytest.raises(TypeError, match="`cc` must be a list or tuple of length 2"):
        PatchedLinker(cc=0)


@pytest.mark.parametrize("max_registers", (None, 32))
@pytest.mark.parametrize("lineinfo", (False, True))
@pytest.mark.parametrize("lto", (False, True))
@pytest.mark.parametrize("additional_flags", (None, ("-g",), ("-g", "-time")))
def test_ptx_compile_options(max_registers, lineinfo, lto, additional_flags):
    patched_linker = PatchedLinker(
        cc=(7, 5),
        max_registers=max_registers,
        lineinfo=lineinfo,
        lto=lto,
        additional_flags=additional_flags,
    )

    assert "-arch=sm_75" in patched_linker.options

    if max_registers:
        assert f"-maxrregcount={max_registers}" in patched_linker.options
    else:
        assert "-maxrregcount" not in patched_linker.options

    if lineinfo:
        assert "-lineinfo" in patched_linker.options
    else:
        assert "-lineinfo" not in patched_linker.options

    if lto:
        assert "-lto" in patched_linker.options
    else:
        assert "-lto" not in patched_linker.options

    if additional_flags:
        for flag in additional_flags:
            assert flag in patched_linker.options


@pytest.mark.parametrize(
    "file",
    (
        "linkable_code_archive",
        "linkable_code_cubin",
        "linkable_code_cusource",
        "linkable_code_fatbin",
        "linkable_code_object",
        "linkable_code_ptx",
    ),
)
def test_add_file_guess_ext_linkable_code(file, gpu_compute_capability, request):
    file = request.getfixturevalue(file)
    patched_linker = PatchedLinker(cc=gpu_compute_capability)
    patched_linker.add_file_guess_ext(file)


def test_add_file_guess_ext_invalid_input(
    device_functions_cubin, gpu_compute_capability
):
    # Feeding raw data as bytes to add_file_guess_ext should raise, because
    # there's no way to know what kind of file to treat it as
    patched_linker = PatchedLinker(cc=gpu_compute_capability)
    with pytest.raises(TypeError, match="Expected path to file or a LinkableCode"):
        patched_linker.add_file_guess_ext(device_functions_cubin)


@pytest.mark.skipif(
    not _numba_version_ok,
    reason=f"Requires Numba == {required_numba_ver[0]}.{required_numba_ver[1]}",
)
@pytest.mark.parametrize(
    "file",
    (
        "linkable_code_archive",
        "linkable_code_cubin",
        "linkable_code_cusource",
        "linkable_code_fatbin",
        "linkable_code_object",
        "linkable_code_ptx",
    ),
)
def test_jit_with_linkable_code(file, request):
    file = request.getfixturevalue(file)
    patch_numba_linker()

    sig = "uint32(uint32, uint32)"
    add_from_numba = cuda.declare_device("add_from_numba", sig)

    @cuda.jit(link=[file])
    def kernel(result):
        result[0] = add_from_numba(1, 2)

    result = cuda.device_array(1)
    kernel[1, 1](result)
    assert result[0] == 3


@pytest.fixture
def numba_linking_with_lto():
    """
    Patch the linker for LTO for the duration of the test.
    Afterwards, restore the linker to whatever it was before.
    """
    from numba.cuda.cudadrv.driver import Linker

    old_new = Linker.new
    patch_numba_linker(lto=True)
    yield
    Linker.new = old_new


def test_jit_with_linkable_code_lto(linkable_code_ltoir, numba_linking_with_lto):
    sig = "uint32(uint32, uint32)"
    add_from_numba = cuda.declare_device("add_from_numba", sig)

    @cuda.jit(link=[linkable_code_ltoir])
    def kernel(result):
        result[0] = add_from_numba(1, 2)

    result = cuda.device_array(1)
    kernel[1, 1](result)
    assert result[0] == 3


@pytest.mark.skipif(
    not _numba_version_ok,
    reason=f"Requires Numba == {required_numba_ver[0]}.{required_numba_ver[1]}",
)
def test_jit_with_invalid_linkable_code(device_functions_cubin):
    # Attempting to pass raw bytes to the `link` kwarg should fail as in
    # test_add_file_guess_ext_invalid_input - this is testing the same error
    # checking triggered through the "public" API of the patched behaviour
    with pytest.raises(TypeError, match="Expected path to file or a LinkableCode"):

        @cuda.jit("void()", link=[device_functions_cubin])
        def kernel():
            pass


if __name__ == "__main__":
    sys.exit(pytest.main())
