# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import weakref
from enum import Enum

from pynvjitlink import _nvjitlinklib


class InputType(Enum):
    NONE = 0
    CUBIN = 1
    PTX = 2
    LTOIR = 3
    FATBIN = 4
    OBJECT = 5
    LIBRARY = 6


def nvjitlink_version():
    return _nvjitlinklib.nvjitlink_version()


class NvJitLinkError(RuntimeError):
    pass


class NvJitLinker:
    def __init__(self, *options):
        try:
            self.handle = _nvjitlinklib.create(*options)
        except RuntimeError as e:
            raise NvJitLinkError(f"{e}")

        weakref.finalize(self, _nvjitlinklib.destroy, self.handle)

        self._info_log = None
        self._error_log = None
        self._complete = False

    @property
    def info_log(self):
        return self._info_log

    @property
    def error_log(self):
        return self._error_log

    def add_data(self, input_type, data, name):
        if self._complete:
            raise NvJitLinkError("Cannot add data to already-completeted link")

        try:
            _nvjitlinklib.add_data(self.handle, input_type.value, data, name)
        except RuntimeError as e:
            self._info_log = _nvjitlinklib.get_info_log(self.handle)
            self._error_log = _nvjitlinklib.get_error_log(self.handle)
            raise NvJitLinkError(f"{e}\n{self.error_log}")

    def add_cubin(self, cubin, name=None):
        name = name or "unnamed-cubin"
        self.add_data(InputType.CUBIN, cubin, name)

    def add_ptx(self, ptx, name=None):
        name = name or "unnamed-ptx"
        self.add_data(InputType.PTX, ptx, name)

    def add_ltoir(self, ltoir, name=None):
        name = name or "unnamed-ltoir"
        self.add_data(InputType.LTOIR, ltoir, name)

    def add_object(self, object_, name=None):
        name = name or "unnamed-object"
        self.add_data(InputType.OBJECT, object_, name)

    def add_fatbin(self, fatbin, name=None):
        name = name or "unnamed-fatbin"
        self.add_data(InputType.FATBIN, fatbin, name)

    def add_library(self, library, name=None):
        self.add_data(InputType.LIBRARY, library, name)

    def get_linked_cubin(self):
        try:
            _nvjitlinklib.complete(self.handle)
            self._complete = True
            return _nvjitlinklib.get_linked_cubin(self.handle)
        except RuntimeError as e:
            self._error_log = _nvjitlinklib.get_error_log(self.handle)
            raise NvJitLinkError(f"{e}\n{self.error_log}")
        finally:
            self._info_log = _nvjitlinklib.get_info_log(self.handle)

    def get_linked_ptx(self):
        try:
            _nvjitlinklib.complete(self.handle)
            self._complete = True
            return _nvjitlinklib.get_linked_ptx(self.handle)
        except RuntimeError as e:
            self._error_log = _nvjitlinklib.get_error_log(self.handle)
            raise NvJitLinkError(f"{e}\n{self.error_log}")
        finally:
            self._info_log = _nvjitlinklib.get_info_log(self.handle)
