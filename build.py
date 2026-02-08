#!/usr/bin/env python3
"""
Convenience script for building the benchmark_utils C++ extension.

This script provides a simple interface for building, cleaning, and testing
the C++ extension module.

Usage:
    python build.py              # Build the extension
    python build.py clean        # Clean build artifacts
    python build.py rebuild      # Clean and rebuild
    python build.py test         # Build and test import
"""

import sys
import os
import shutil
import subprocess
import argparse


def clean_build():
    """Remove build artifacts and compiled extensions."""
    print("Cleaning build artifacts...")
    
    # Directories to remove
    dirs_to_remove = ['build', 'dist', '*.egg-info']
    
    for pattern in dirs_to_remove:
        if '*' in pattern:
            # Handle glob patterns
            import glob
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    print(f"  Removing directory: {path}")
                    shutil.rmtree(path, ignore_errors=True)
        else:
            if os.path.exists(pattern):
                print(f"  Removing directory: {pattern}")
                shutil.rmtree(pattern, ignore_errors=True)
    
    # Remove compiled extension files
    extensions = [
        'benchmark_utils.so',           # Linux
        'benchmark_utils.dylib',        # macOS
        'benchmark_utils.pyd',          # Windows
        'benchmark_utils*.so',          # Linux with version suffix
    ]
    
    import glob
    for pattern in extensions:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                print(f"  Removing file: {file}")
                os.remove(file)
    
    print("Clean complete!\n")


def build_extension():
    """Build the C++ extension."""
    print("Building C++ extension...")
    print("-" * 60)
    
    # Run setup.py build_ext --inplace
    cmd = [sys.executable, 'setup.py', 'build_ext', '--inplace']
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("-" * 60)
        print("Build successful!\n")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"Build failed with error code {e.returncode}\n")
        return False


def test_import():
    """Test if the extension can be imported."""
    print("Testing extension import...")
    
    try:
        import benchmark_utils
        print("✓ Successfully imported benchmark_utils")
        
        # Check available functions
        functions = [
            'cpp_write_blocks',
            'cpp_read_blocks',
            'set_io_thread_count',
            'get_io_thread_count',
        ]
        
        print("\nAvailable functions:")
        for func_name in functions:
            if hasattr(benchmark_utils, func_name):
                print(f"  ✓ {func_name}")
            else:
                print(f"  ✗ {func_name} (missing!)")
        
        # Get thread count
        thread_count = benchmark_utils.get_io_thread_count()
        print(f"\nDefault thread count: {thread_count}")
        
        print("\nExtension is ready to use!\n")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import benchmark_utils: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build script for benchmark_utils C++ extension',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py              Build the extension
  python build.py clean        Clean build artifacts
  python build.py rebuild      Clean and rebuild
  python build.py test         Build and test import
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='build',
        choices=['build', 'clean', 'rebuild', 'test'],
        help='Command to execute (default: build)'
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 60)
    print("Benchmark Utils C++ Extension Builder")
    print("=" * 60)
    print()
    
    if args.command == 'clean':
        clean_build()
        
    elif args.command == 'build':
        success = build_extension()
        if not success:
            sys.exit(1)
            
    elif args.command == 'rebuild':
        clean_build()
        success = build_extension()
        if not success:
            sys.exit(1)
            
    elif args.command == 'test':
        success = build_extension()
        if success:
            test_import()
        else:
            sys.exit(1)
    
    print("=" * 60)


if __name__ == '__main__':
    main()
