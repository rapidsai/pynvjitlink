# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path
import shutil
import sys
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_config_var, get_python_inc
from setuptools import setup, Extension

include_dirs = [os.path.dirname(get_python_inc())]
library_dirs = [get_config_var("LIBDIR")]

# Find and add CUDA include paths
CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))
if not os.path.isdir(CUDA_HOME):
    raise OSError(f"Invalid CUDA_HOME: directory does not exist: {CUDA_HOME}")
include_dirs.append(os.path.join(CUDA_HOME, "include"))
library_dirs.append(os.path.join(CUDA_HOME, "lib64"))

default_werror_wall = sys.platform.startswith('linux')

nvjitlink_be_user_options = [
    ('werror', None, 'Build extensions with -Werror'),
    ('wall', None, 'Build extensions with -Wall'),
    ('noopt', None, 'Build extensions without optimization'),
]


class NvJitLinkBuildExt(build_ext):

    user_options = build_ext.user_options + nvjitlink_be_user_options
    boolean_options = build_ext.boolean_options + ['werror', 'wall', 'noopt']

    def initialize_options(self):
        super().initialize_options()
        self.werror = default_werror_wall
        self.wall = default_werror_wall
        self.noopt = 0

    def run(self):
        extra_compile_args = []
        if self.noopt:
            if sys.platform == 'win32':
                extra_compile_args.append('/Od')
            else:
                extra_compile_args.append('-O0')
        if self.werror:
            extra_compile_args.append('-Werror')
        if self.wall:
            extra_compile_args.append('-Wall')
        for ext in self.extensions:
            ext.extra_compile_args.extend(extra_compile_args)

        super().run()


cmdclass = {}
cmdclass['build_ext'] = NvJitLinkBuildExt

module = Extension(
    'nvjitlink._nvjitlinklib',
    sources=['nvjitlink/_nvjitlinklib.cpp'],
    include_dirs=include_dirs,
    libraries=['nvJitLink_static', 'nvptxcompiler_static'],
    library_dirs=library_dirs,
    extra_compile_args=['-Wall', '-Werror'],
)

setup(
    name='nvjitlink',
    description='nvJitLink Python binding',
    ext_modules=[module],
    packages=['nvjitlink', 'nvjitlink.tests'],
    cmdclass=cmdclass,
    package_data={'nvjitlink.tests': ['test_device_functions.*', 'undefined_extern.cubin']},
)
