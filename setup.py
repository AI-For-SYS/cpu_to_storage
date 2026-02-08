"""
Setup script for building the benchmark_utils C++ extension.

This script compiles the C++ file I/O utilities into a Python extension module
that can be imported and used for high-performance parallel file operations.

Usage:
    python setup.py build_ext --inplace    # Build in current directory
    python setup.py install                 # Install to site-packages
    python setup.py clean                   # Clean build artifacts
"""

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

# Get the directory containing this setup.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
cpp_utils_dir = os.path.join(current_dir, 'benchmark_cpp_utils')

# Source files for the extension
sources = [
    os.path.join(cpp_utils_dir, 'benchmark_utils.cpp'),
    os.path.join(cpp_utils_dir, 'file_io.cpp'),
    os.path.join(cpp_utils_dir, 'simple_thread_pool.cpp'),
]

# Include directories
include_dirs = [cpp_utils_dir]

# Compiler flags for optimization and C++17 support
extra_compile_args = {
    'cxx': [
        '-std=c++17',      # C++17 standard (required for std::filesystem)
        '-O3',             # Maximum optimization
        '-march=native',   # Optimize for current CPU architecture
        '-fPIC',           # Position independent code
    ]
}

# Linker flags
extra_link_args = []

# Define the extension module
ext_modules = [
    CppExtension(
        name='benchmark_utils',
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args['cxx'],
        extra_link_args=extra_link_args,
    )
]

setup(
    name='benchmark_utils',
    version='1.0.0',
    author='Your Name',
    description='High-performance parallel file I/O utilities for PyTorch',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
    ],
)
