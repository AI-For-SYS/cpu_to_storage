"""
Setup script for building the iouring_ext C++ extension.

This script compiles the io_uring-based I/O utilities into a Python extension module.
Requires liburing to be installed (system-wide or at $HOME/.local).

Usage:
    python setup_iouring.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
iouring_utils_dir = os.path.join(current_dir, 'backends', 'benchmark_iouring_utils')

# liburing: check $HOME/.local first (built from source), then system paths
home_local = os.path.join(os.path.expanduser('~'), '.local')
include_dirs = [iouring_utils_dir]
library_dirs = []

if os.path.isfile(os.path.join(home_local, 'include', 'liburing.h')):
    include_dirs.append(os.path.join(home_local, 'include'))
    library_dirs.append(os.path.join(home_local, 'lib'))

sources = [
    os.path.join(iouring_utils_dir, 'iouring_utils.cpp'),
]

extra_compile_args = [
    '-std=c++17',
    '-O3',
    '-march=native',
    '-fPIC',
]

# Linker flags: link liburing + set rpath for torch and liburing
import torch
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
extra_link_args = [f'-Wl,-rpath,{torch_lib_dir}']
if library_dirs:
    extra_link_args.append(f'-Wl,-rpath,{library_dirs[0]}')

ext_modules = [
    CppExtension(
        name='iouring_ext',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['uring'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name='iouring_ext',
    version='1.0.0',
    description='io_uring-based parallel file I/O utilities for PyTorch',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
    ],
)
