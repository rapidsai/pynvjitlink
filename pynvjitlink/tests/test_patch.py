# Copyright (c) 2023, NVIDIA CORPORATION.

import pytest
import sys

from pynvjitlink import patch, NvJitLinkError
from pynvjitlink.patch import (PatchedLinker, patch_numba_linker,
                               new_patched_linker, required_numba_ver,
                               _numba_version_ok)
from unittest.mock import patch as mock_patch


def test_numba_patching_numba_not_ok():
    with mock_patch.multiple(
            patch,
            _numba_version_ok=False,
            _numba_error='<error>'):
        with pytest.raises(RuntimeError, match='Cannot patch Numba: <error>'):
            patch_numba_linker()


@pytest.mark.skipif(
    not _numba_version_ok,
    reason=f"Requires Numba == {required_numba_ver[0]}.{required_numba_ver[1]}"
)
def test_numba_patching():
    # We import the linker here rather than at the top level because the import
    # may fail if if Numba is not present or an unsupported version.
    from numba.cuda.cudadrv.driver import Linker
    patch_numba_linker()
    assert Linker.new is new_patched_linker


def test_create():
    patched_linker = PatchedLinker(cc=(7, 5))
    assert "-arch=sm_75" in patched_linker.options


def test_create_no_cc_error():
    # nvJitLink expects at least the architecture to be specified.
    with pytest.raises(RuntimeError,
                       match='PatchedLinker requires CC to be specified'):
        PatchedLinker()


def test_invalid_arch_error():
    # CC 0.0 is not a valid compute capability
    with pytest.raises(NvJitLinkError,
                       match='NVJITLINK_ERROR_UNRECOGNIZED_OPTION error'):
        PatchedLinker(cc=(0, 0))


def test_invalid_cc_type_error():
    with pytest.raises(TypeError,
                       match='`cc` must be a list or tuple of length 2'):
        PatchedLinker(cc=0)


@pytest.mark.parametrize('max_registers', (None, 32))
@pytest.mark.parametrize('lineinfo', (False, True))
@pytest.mark.parametrize('lto', (False, True))
@pytest.mark.parametrize('additional_flags', (None, ('-g',), ('-g', '-time')))
def test_ptx_compile_options(max_registers, lineinfo, lto, additional_flags):
    patched_linker = PatchedLinker(
        cc=(7, 5),
        max_registers=max_registers,
        lineinfo=lineinfo,
        lto=lto,
        additional_flags=additional_flags
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


if __name__ == '__main__':
    sys.exit(pytest.main())
