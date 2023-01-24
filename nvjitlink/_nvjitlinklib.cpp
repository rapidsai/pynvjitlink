/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

static PyObject *create(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}

static PyObject *destroy(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}

static PyObject *add_data(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}
static PyObject *add_file(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}
static PyObject *complete(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}
static PyObject *get_error_log(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}
static PyObject *get_info_log(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}
static PyObject *get_linked_ptx(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}
static PyObject *get_linked_cubin(PyObject *self, PyObject *args) {
  set_exception(PyExc_NotImplementedError, "Unimplemented", NVJITLINK_SUCCESS);

  return nullptr;
}

static PyMethodDef ext_methods[] = {
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
    PyModuleDef_HEAD_INIT, "nvjitlink",
    "Provides access to nvJitLink API methods", -1, ext_methods};

PyMODINIT_FUNC PyInit__nvjitlinklib(void) {
  PyObject *m = PyModule_Create(&moduledef);
  return m;
}
