"""
Setup script for building the threaded_tunable_ext C++ extension.

Usage:
    python setup_threaded_tunable.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
cpp_utils_dir = os.path.join(current_dir, 'backends', 'benchmark_cpp_utils')

sources = [
    os.path.join(cpp_utils_dir, 'threaded_tunable_utils.cpp'),
    os.path.join(cpp_utils_dir, 'simple_thread_pool.cpp'),
]

include_dirs = [cpp_utils_dir]

extra_compile_args = [
    '-std=c++17',
    '-O3',
    '-march=native',
    '-fPIC',
]

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
extra_link_args = [f'-Wl,-rpath,{torch_lib_dir}']

ext_modules = [
    CppExtension(
        name='threaded_tunable_ext',
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name='threaded_tunable_ext',
    version='1.0.0',
    description='Tunable thread-based parallel file I/O for Optuna auto-tuning',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.7',
    install_requires=['torch>=1.8.0'],
)
