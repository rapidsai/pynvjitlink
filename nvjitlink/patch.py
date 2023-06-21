# Copyright (c) 2023, NVIDIA CORPORATION.

from ptxcompiler.api import compile_ptx
from ptxcompiler.patch import get_logger, patch_needed
from nvjitlink.api import NvJitLinker, NvJitLinkError

import os
import pathlib

_numba_version_ok = False
_numba_error = None

required_numba_ver = (0, 56)

mvc_docs_url = ("https://numba.readthedocs.io/en/stable/cuda/"
                "minor_version_compatibility.html")

try:
    import numba
    ver = numba.version_info.short
    if ver < required_numba_ver:
        _numba_error = (f"version {numba.__version__} is insufficient for "
                        "patching - %s.%s is needed." % required_numba_ver)
    elif ver > required_numba_ver:
        _numba_error = (
            f"version {numba.__version__} should not be patched. "
            "Set the environment variable "
            "NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1 instead. "
            f"See {mvc_docs_url} for more details.")
    else:
        _numba_version_ok = True
except ImportError as ie:
    _numba_error = f"failed to import Numba: {ie}."

if _numba_version_ok:
    from numba.core import config
    from numba.cuda.cudadrv.driver import (FILE_EXTENSION_MAP, NvrtcProgram,
                                           Linker, LinkerError)
else:
    # Prevent the definition of PatchedLinker failing if we have no Numba
    # Linker - it won't be used anyway.
    Linker = object


class PatchedLinker(Linker):
    def __init__(self, max_registers=None, lineinfo=False, cc=None):
        if cc is None:
            raise RuntimeError("PatchedLinker requires CC to be specified")
        if not any(isinstance(cc, t) for t in [list, tuple]):
            raise TypeError("`cc` must be a list or tuple of length 2")

        sm_ver = f"{cc[0] * 10 + cc[1]}"
        ptx_compile_opts = ['--gpu-name', f'sm_{sm_ver}', '-c']
        if max_registers:
            arg = f"--maxrregcount={max_registers}"
            ptx_compile_opts.append(arg)
        if lineinfo:
            ptx_compile_opts.append('--generate-line-info')
        self.ptx_compile_options = tuple(ptx_compile_opts)

        arch = f"-arch=sm_{sm_ver}"
        self._linker = NvJitLinker(arch)

    @property
    def info_log(self):
        return self._linker.info_log

    @property
    def error_log(self):
        return self._linker.error_log

    def add_ptx(self, ptx, name='<cudapy-ptx>'):
        self._linker.add_ptx(ptx, name)

    def add_file(self, path, kind):
        try:
            with open(path, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            raise LinkerError(f'{path} not found')

        name = pathlib.Path(path).name
        if kind == FILE_EXTENSION_MAP['cubin']:
            fn = self._linker.add_cubin
        elif kind == FILE_EXTENSION_MAP['fatbin']:
            fn = self._linker.add_fatbin
        elif kind == FILE_EXTENSION_MAP['a']:
            raise LinkerError("Don't know how to link archives")
        elif kind == FILE_EXTENSION_MAP['ptx']:
            return self.add_ptx(data, name)
        else:
            raise LinkerError(f"Don't know how to link {kind}")

        try:
            fn(data, name)
        except NvJitLinkError as e:
            raise LinkerError from e

    def add_cu(self, cu, name):
        program = NvrtcProgram(cu, name)

        if config.DUMP_ASSEMBLY:
            print(("ASSEMBLY %s" % name).center(80, '-'))
            print(program.ptx.decode())
            print('=' * 80)

        # Link the program's PTX using the normal linker mechanism
        ptx_name = os.path.splitext(name)[0] + ".ptx"
        self.add_ptx(program.ptx.rstrip(b'\x00'), ptx_name)

    def complete(self):
        try:
            cubin = self._linker.get_linked_cubin()
            self._linker._complete = True
            return cubin
        except NvJitLinkError as e:
            raise LinkerError from e


def new_patched_linker(max_registers=0, lineinfo=False, cc=None):
    logger = get_logger()
    logger.debug("Creating new PatchedLinker")
    return PatchedLinker(max_registers, lineinfo, cc)


def patch_numba_linker_if_needed():
    if not _numba_version_ok:
        msg = f"Cannot patch Numba: {_numba_error}"
        raise RuntimeError(msg)

    logger = get_logger()

    if patch_needed():
        logger.debug("Patching Numba to use NvJitLinker")
        Linker.new = new_patched_linker
    else:
        logger.debug("Not patching Numba to use NvJitLinker")
