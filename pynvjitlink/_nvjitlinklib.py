# Copyright (c) 2024, NVIDIA CORPORATION.

# A shim layer to use new bindings from CUDA Python
from cuda.bindings import nvjitlink
from cuda.bindings.nvjitlink import nvJitLinkError


def nvjitlink_version():
    return nvjitlink.version()


def create(*options):
    return nvjitlink.create(len(options), options)


def destroy(handle):
    return nvjitlink.destroy(handle)


def add_data(handle, input_type, data, filename):
    nvjitlink.add_data(handle, input_type, data, len(data), filename)


def add_file(*args, **kwargs):
    raise NotImplementedError("seems unused by pynvjitlink")


def complete(handle):
    nvjitlink.complete(handle)


def get_error_log(handle):
    log_size = nvjitlink.get_error_log_size(handle)
    log_size += 1
    print(f"{log_size=}")
    log = bytearray(log_size)
    nvjitlink.get_error_log(handle, log)
    return log.decode()


def get_info_log(handle):
    log_size = nvjitlink.get_info_log_size(handle)
    log = bytearray(log_size)
    nvjitlink.get_info_log(handle, log)
    return log.decode()


def get_linked_ptx(handle):
    ptx_size = nvjitlink.get_linked_ptx_size(handle)
    ptx = bytearray(ptx_size)
    nvjitlink.get_linked_ptx(handle, ptx)
    return ptx.decode()


def get_linked_cubin(handle):
    cubin_size = nvjitlink.get_linked_cubin_size(handle)
    cubin = bytearray(cubin_size)
    nvjitlink.get_linked_cubin(handle, cubin)
    return cubin
