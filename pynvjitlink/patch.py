# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import os
import pathlib
from functools import partial

from pynvjitlink.api import NvJitLinker, NvJitLinkError

_numba_version_ok = False
_numba_error = None

required_numba_ver = (0, 58)

mvc_docs_url = (
    "https://numba.readthedocs.io/en/stable/cuda/" "minor_version_compatibility.html"
)

try:
    import numba

    ver = numba.version_info.short
    if ver < required_numba_ver:
        _numba_error = (
            f"version {numba.__version__} is insufficient for "
            "patching - %s.%s is needed." % required_numba_ver
        )
    else:
        _numba_version_ok = True
except ImportError as ie:
    _numba_error = f"failed to import Numba: {ie}."

if _numba_version_ok:
    from numba import cuda
    from numba.core import config
    from numba.cuda.cudadrv import nvrtc
    from numba.cuda.cudadrv.driver import (
        FILE_EXTENSION_MAP,
        Linker,
        LinkerError,
        driver,
    )
else:
    # Prevent the definition of PatchedLinker failing if we have no Numba
    # Linker - it won't be used anyway.
    Linker = object


class LinkableCode:
    """An object that can be passed in the `link` list argument to `@cuda.jit`
    kernels to supply code to be linked from memory."""

    def __init__(self, data, name=None):
        self.data = data
        self._name = name

    @property
    def name(self):
        return self._name or self.default_name


class PTXSource(LinkableCode):
    """PTX Source code in memory"""

    kind = FILE_EXTENSION_MAP["ptx"]
    default_name = "<unnamed-ptx>"


class CUSource(LinkableCode):
    """CUDA C/C++ Source code in memory"""

    kind = "cu"
    default_name = "<unnamed-cu>"


class Fatbin(LinkableCode):
    """A fatbin ELF in memory"""

    kind = FILE_EXTENSION_MAP["fatbin"]
    default_name = "<unnamed-fatbin>"


class Cubin(LinkableCode):
    """A cubin ELF in memory"""

    kind = FILE_EXTENSION_MAP["cubin"]
    default_name = "<unnamed-cubin>"


class Archive(LinkableCode):
    """An archive of objects in memory"""

    kind = FILE_EXTENSION_MAP["a"]
    default_name = "<unnamed-archive>"


class Object(LinkableCode):
    """An object file in memory"""

    kind = FILE_EXTENSION_MAP["o"]
    default_name = "<unnamed-object>"


class LTOIR(LinkableCode):
    """An LTOIR file in memory"""

    kind = "ltoir"
    default_name = "<unnamed-ltoir>"


class PatchedLinker(Linker):
    def __init__(
        self,
        max_registers=None,
        lineinfo=False,
        cc=None,
        lto=False,
        additional_flags=None,
    ):
        if cc is None:
            raise RuntimeError("PatchedLinker requires CC to be specified")
        if not any(isinstance(cc, t) for t in [list, tuple]):
            raise TypeError("`cc` must be a list or tuple of length 2")

        sm_ver = f"{cc[0] * 10 + cc[1]}"
        arch = f"-arch=sm_{sm_ver}"
        options = [arch]
        if max_registers:
            options.append(f"-maxrregcount={max_registers}")
        if lineinfo:
            options.append("-lineinfo")
        if lto:
            options.append("-lto")
        if additional_flags is not None:
            options.extend(additional_flags)

        self._linker = NvJitLinker(*options)
        self.lto = lto
        self.options = options

    @property
    def info_log(self):
        return self._linker.info_log

    @property
    def error_log(self):
        return self._linker.error_log

    def add_ptx(self, ptx, name="<cudapy-ptx>"):
        self._linker.add_ptx(ptx, name)

    def add_fatbin(self, fatbin, name="<external-fatbin>"):
        self._linker.add_fatbin(fatbin, name)

    def add_ltoir(self, ltoir, name="<external-ltoir>"):
        self._linker.add_ltoir(ltoir, name)

    def add_object(self, obj, name="<external-object>"):
        self._linker.add_object(obj, name)

    def add_file_guess_ext(self, path_or_code):
        # Numba's add_file_guess_ext expects to always be passed a path to a
        # file that it will load from the filesystem to link. We augment it
        # here with the ability to provide a file from memory.

        # To maintain compatibility with the original interface, all strings
        # are treated as paths in the filesystem.
        if isinstance(path_or_code, str):
            # Upstream numba does not yet recognize LTOIR, so handle that
            # separately here.
            extension = pathlib.Path(path_or_code).suffix
            if extension == ".ltoir":
                self.add_file(path_or_code, "ltoir")
            else:
                # Use Numba's logic for non-LTOIR
                super().add_file_guess_ext(path_or_code)

            return

        # Otherwise, we should have been given a LinkableCode object
        if not isinstance(path_or_code, LinkableCode):
            raise TypeError("Expected path to file or a LinkableCode object")

        if path_or_code.kind == "cu":
            self.add_cu(path_or_code.data, path_or_code.name)
        else:
            self.add_data(path_or_code.data, path_or_code.kind, path_or_code.name)

    def add_file(self, path, kind):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            raise LinkerError(f"{path} not found")

        name = pathlib.Path(path).name
        self.add_data(data, kind, name)

    def add_data(self, data, kind, name):
        if kind == FILE_EXTENSION_MAP["cubin"]:
            fn = self._linker.add_cubin
        elif kind == FILE_EXTENSION_MAP["fatbin"]:
            fn = self._linker.add_fatbin
        elif kind == FILE_EXTENSION_MAP["a"]:
            fn = self._linker.add_library
        elif kind == FILE_EXTENSION_MAP["ptx"]:
            return self.add_ptx(data, name)
        elif kind == FILE_EXTENSION_MAP["o"]:
            fn = self._linker.add_object
        elif kind == "ltoir":
            fn = self._linker.add_ltoir
        else:
            raise LinkerError(f"Don't know how to link {kind}")

        try:
            fn(data, name)
        except NvJitLinkError as e:
            raise LinkerError from e

    def add_cu(self, cu, name):
        with driver.get_active_context() as ac:
            dev = driver.get_device(ac.devnum)
            cc = dev.compute_capability

        ptx, log = nvrtc.compile(cu, name, cc)

        if config.DUMP_ASSEMBLY:
            print((f"ASSEMBLY {name}").center(80, "-"))
            print(ptx)
            print("=" * 80)

        # Link the program's PTX using the normal linker mechanism
        ptx_name = os.path.splitext(name)[0] + ".ptx"
        self.add_ptx(ptx.encode(), ptx_name)

    def complete(self):
        try:
            cubin = self._linker.get_linked_cubin()
            self._linker._complete = True
            return cubin
        except NvJitLinkError as e:
            raise LinkerError from e


def new_patched_linker(
    max_registers=0, lineinfo=False, cc=None, lto=False, additional_flags=None
):
    return PatchedLinker(
        max_registers=max_registers,
        lineinfo=lineinfo,
        cc=cc,
        lto=lto,
        additional_flags=additional_flags,
    )


def patch_numba_linker(*, lto=False):
    if not _numba_version_ok:
        msg = f"Cannot patch Numba: {_numba_error}"
        raise RuntimeError(msg)

    # Replace the built-in linker that uses the Driver API with our linker that
    # uses nvJitLink
    Linker.new = partial(new_patched_linker, lto=lto)

    # Add linkable code objects to Numba's top-level API
    cuda.Archive = Archive
    cuda.CUSource = CUSource
    cuda.Cubin = Cubin
    cuda.Fatbin = Fatbin
    cuda.Object = Object
    cuda.PTXSource = PTXSource
    cuda.LTOIR = LTOIR
