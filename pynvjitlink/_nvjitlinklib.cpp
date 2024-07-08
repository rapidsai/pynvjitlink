/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define PY_SSIZE_T_CLEAN
#include "nvJitLink.h"
#include <Python.h>
#include <new>

static const char *nvJitLinkGetErrorEnum(nvJitLinkResult error) {
  switch (error) {
  case NVJITLINK_SUCCESS:
    return "NVJITLINK_SUCCESS";

  case NVJITLINK_ERROR_UNRECOGNIZED_OPTION:
    return "NVJITLINK_ERROR_UNRECOGNIZED_OPTION";

  case NVJITLINK_ERROR_MISSING_ARCH:
    return "NVJITLINK_ERROR_MISSING_ARCH";

  case NVJITLINK_ERROR_INVALID_INPUT:
    return "NVJITLINK_ERROR_INVALID_INPUT";

  case NVJITLINK_ERROR_PTX_COMPILE:
    return "NVJITLINK_ERROR_PTX_COMPILE";

  case NVJITLINK_ERROR_NVVM_COMPILE:
    return "NVJITLINK_ERROR_NVVM_COMPILE";

  case NVJITLINK_ERROR_INTERNAL:
    return "NVJITLINK_ERROR_INTERNAL";

  default:
    return "<unknown>";
  }
}

static void set_exception(PyObject *exception_type, const char *message_format,
                          nvJitLinkResult error) {
  char exception_message[256];
  sprintf(exception_message, message_format, nvJitLinkGetErrorEnum(error));

  PyErr_SetString(exception_type, exception_message);
}

static PyObject *nvjitlink_version(PyObject *self, PyObject *Py_UNUSED(args)) {
  unsigned int major;
  unsigned int minor;

  nvJitLinkResult res = nvJitLinkVersion(&major, &minor);

  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError, "%s error when calling nvJitLinkVersion",
                  res);
    return nullptr;
  }

  PyObject *py_version = PyTuple_New(2);
  PyObject *py_major = PyLong_FromUnsignedLong(major);
  PyObject *py_minor = PyLong_FromUnsignedLong(minor);
  if (!py_version || !py_major || !py_minor) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create version tuple");
    if (py_major) {
      Py_DecRef(py_major);
    }
    if (py_minor) {
      Py_DecRef(py_minor);
    }
    if (py_version) {
      Py_DecRef(py_version);
    }
    return nullptr;
  }

  PyTuple_SetItem(py_version, 0, py_major);
  PyTuple_SetItem(py_version, 1, py_minor);
  return py_version;
}

static PyObject *create(PyObject *self, PyObject *args) {
  PyObject *ret = nullptr;
  const char **jitlink_options;
  nvJitLinkHandle *jitlink;

  Py_ssize_t n_args = PyTuple_Size(args);

  try {
    jitlink_options = new const char *[n_args];
  } catch (const std::bad_alloc &) {
    PyErr_NoMemory();
    return nullptr;
  }

  for (Py_ssize_t i = 0; i < n_args; ++i) {
    PyObject *py_option = PyTuple_GetItem(args, i);
    if (!PyUnicode_Check(py_option)) {
      PyErr_SetString(PyExc_TypeError,
                      "Expecting only strings for jitlink args");
      delete[] jitlink_options;
      return nullptr;
    }

    jitlink_options[i] = PyUnicode_AsUTF8AndSize(py_option, nullptr);
  }

  try {
    jitlink = new nvJitLinkHandle;
  } catch (const std::bad_alloc &) {
    PyErr_NoMemory();
    delete[] jitlink_options;
    return nullptr;
  }

  nvJitLinkResult res = nvJitLinkCreate(jitlink, n_args, jitlink_options);
  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError, "%s error when calling nvJitLinkCreate",
                  res);
    goto error;
  }

  if ((ret = PyLong_FromUnsignedLongLong((unsigned long long)jitlink)) ==
      nullptr) {
    // Attempt to destroy the linker - since we're already in an error
    // condition, there's no point in checking the return code and taking any
    // further action based on it though.
    nvJitLinkDestroy(jitlink);
    goto error;
  }

  delete[] jitlink_options;
  return ret;

error:
  delete jitlink;
  delete[] jitlink_options;
  return nullptr;
}

static PyObject *destroy(PyObject *self, PyObject *args) {
  nvJitLinkHandle *jitlink;
  if (!PyArg_ParseTuple(args, "K", &jitlink))
    return nullptr;

  nvJitLinkResult res = nvJitLinkDestroy(jitlink);

  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError, "%s error when calling nvJitLinkDestroy",
                  res);
    return nullptr;
  }

  delete jitlink;

  Py_RETURN_NONE;
}

static PyObject *add_data(PyObject *self, PyObject *args) {
  nvJitLinkHandle *jitlink;
  nvJitLinkInputType input_type;
  Py_buffer buf;
  const char *name;

  if (!PyArg_ParseTuple(args, "Kiy*s", &jitlink, &input_type, &buf, &name)) {
    return nullptr;
  }

  const void *data = buf.buf;
  size_t size = buf.len;
  nvJitLinkResult res =
      nvJitLinkAddData(*jitlink, input_type, data, size, name);

  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError, "%s error when calling nvJitLinkAddData",
                  res);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyObject *add_file(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}
static PyObject *complete(PyObject *self, PyObject *args) {
  nvJitLinkHandle *jitlink;
  if (!PyArg_ParseTuple(args, "K", &jitlink))
    return nullptr;

  nvJitLinkResult res = nvJitLinkComplete(*jitlink);

  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError, "%s error when calling nvJitLinkComplete",
                  res);
    return nullptr;
  }

  Py_RETURN_NONE;

  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}
static PyObject *get_error_log(PyObject *self, PyObject *args) {
  nvJitLinkHandle *jitlink;
  if (!PyArg_ParseTuple(args, "K", &jitlink))
    return nullptr;

  size_t error_log_size;
  nvJitLinkResult res = nvJitLinkGetErrorLogSize(*jitlink, &error_log_size);
  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError,
                  "%s error when calling nvJitLinkGetErrorLogSize", res);
    return nullptr;
  }

  // The size returned doesn't include a trailing null byte
  char *error_log = new char[error_log_size + 1];
  res = nvJitLinkGetErrorLog(*jitlink, error_log);
  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError,
                  "%s error when calling nvJitLinkGetErrorLog", res);
    return nullptr;
  }

  PyObject *py_log = PyUnicode_FromStringAndSize(error_log, error_log_size);
  // Once we've copied the log to a Python object we can delete it - we don't
  // need to check whether creation of the Unicode object succeeded, because we
  // delete the log either way.
  delete[] error_log;

  return py_log;
}

static PyObject *get_info_log(PyObject *self, PyObject *args) {
  nvJitLinkHandle *jitlink;
  if (!PyArg_ParseTuple(args, "K", &jitlink))
    return nullptr;

  size_t info_log_size;
  nvJitLinkResult res = nvJitLinkGetInfoLogSize(*jitlink, &info_log_size);
  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError,
                  "%s error when calling nvJitLinkGetInfoLogSize", res);
    return nullptr;
  }

  // The size returned doesn't include a trailing null byte
  char *info_log = new char[info_log_size + 1];
  res = nvJitLinkGetInfoLog(*jitlink, info_log);
  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError,
                  "%s error when calling nvJitLinkGetInfoLog", res);
    return nullptr;
  }

  PyObject *py_log = PyUnicode_FromStringAndSize(info_log, info_log_size);
  // Once we've copied the log to a Python object we can delete it - we don't
  // need to check whether creation of the Unicode object succeeded, because we
  // delete the log either way.
  delete[] info_log;

  return py_log;
}

static PyObject *get_linked_ptx(PyObject *self, PyObject *args) {
  nvJitLinkHandle *jitlink;
  if (!PyArg_ParseTuple(args, "K", &jitlink))
    return nullptr;

  size_t linked_ptx_size;
  nvJitLinkResult res = nvJitLinkGetLinkedPtxSize(*jitlink, &linked_ptx_size);
  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError,
                  "%s error when calling nvJitLinkGetLinkedPtxSize", res);
    return nullptr;
  }

  char *linked_ptx = new char[linked_ptx_size];
  res = nvJitLinkGetLinkedPtx(*jitlink, linked_ptx);
  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError,
                  "%s error when calling nvJitLinkGetLinkedPtx", res);
  }

  PyObject *py_ptx = PyBytes_FromStringAndSize(linked_ptx, linked_ptx_size);
  // Once we've copied the compiled program to a Python object we can delete it
  // - we don't need to check whether creation of the Unicode object succeeded,
  // because we delete the compiled program either way.
  delete[] linked_ptx;

  return py_ptx;
}
static PyObject *get_linked_cubin(PyObject *self, PyObject *args) {
  nvJitLinkHandle *jitlink;
  if (!PyArg_ParseTuple(args, "K", &jitlink))
    return nullptr;

  size_t linked_cubin_size;
  nvJitLinkResult res =
      nvJitLinkGetLinkedCubinSize(*jitlink, &linked_cubin_size);
  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError,
                  "%s error when calling nvJitLinkGetLinkedCubinSize", res);
    return nullptr;
  }

  char *linked_cubin = new char[linked_cubin_size];
  res = nvJitLinkGetLinkedCubin(*jitlink, linked_cubin);
  if (res != NVJITLINK_SUCCESS) {
    set_exception(PyExc_RuntimeError,
                  "%s error when calling nvJitLinkGetLinkedCubin", res);
  }

  PyObject *py_cubin =
      PyBytes_FromStringAndSize(linked_cubin, linked_cubin_size);
  // Once we've copied the compiled program to a Python object we can delete it
  // - we don't need to check whether creation of the Unicode object succeeded,
  // because we delete the compiled program either way.
  delete[] linked_cubin;

  return py_cubin;
}

static PyMethodDef ext_methods[] = {
    {"nvjitlink_version", (PyCFunction)nvjitlink_version, METH_NOARGS,
     "Returns the nvJitLink version"},
    {"create", (PyCFunction)create, METH_VARARGS,
     "Returns a handle to a new nvJitLink object"},
    {"destroy", (PyCFunction)destroy, METH_VARARGS,
     "Given a handle, destroy an nvJitLink object"},
    {"add_data", (PyCFunction)add_data, METH_VARARGS,
     "Add data to the link for the given handle"},
    {"add_file", (PyCFunction)add_file, METH_VARARGS,
     "Add a file to the link for the given handle"},
    {"complete", (PyCFunction)complete, METH_VARARGS,
     "Given a handle, complete the link"},
    {"get_error_log", (PyCFunction)get_error_log, METH_VARARGS,
     "Given a handle, return the error log"},
    {"get_info_log", (PyCFunction)get_info_log, METH_VARARGS,
     "Given a handle, return the info log"},
    {"get_linked_ptx", (PyCFunction)get_linked_ptx, METH_VARARGS,
     "Given a handle, provide the linked PTX"},
    {"get_linked_cubin", (PyCFunction)get_linked_cubin, METH_VARARGS,
     "Given a handle, provide the linked cubin"},
    {nullptr}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "pynvjitlink",
    "Provides access to nvJitLink API methods", -1, ext_methods};

PyMODINIT_FUNC PyInit__nvjitlinklib(void) {
  PyObject *m = PyModule_Create(&moduledef);
  return m;
}
