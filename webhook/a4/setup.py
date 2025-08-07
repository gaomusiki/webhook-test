# setup.py, compile the reference package by:
# python setup.py build_ext --inplace

import os
import glob
import shutil

from setuptools import setup
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext

def get_export_symbols_fixed(self, ext):
    pass  # return [] also does the job!

# replace wrong version with the fixed:
build_ext.get_export_symbols = get_export_symbols_fixed

dir_name = "ref"
package_name = "src_ref"

# backup the reference package
if os.path.exists(package_name):
    shutil.rmtree(package_name)
shutil.copytree(dir_name, package_name, ignore=shutil.ignore_patterns("*.c", "*.so", "__pycache__"))

# match all .py files in ref/
source_files = glob.glob(f"{package_name}/**/*.py", recursive=True)

# compile
setup(
    name=package_name,
    ext_modules=cythonize(
        f"{package_name}/**/*.py",  # compile all .py files in ref/ to .so files
        compiler_directives={'language_level': "3", 'infer_types': True, 'boundscheck': False}
    ),
    zip_safe=False,
)

# clean up the .c files created temporarily
for c_file in glob.glob(f"{package_name}/**/*.c", recursive=True):
    os.remove(c_file)
    
# clean up all the .py files to hide the reference codes
for py_file in source_files:
    os.remove(py_file)
    
# clean up the __pycache__ dir in ref/
for cache_dir in glob.glob(f"{package_name}/**/__pycache__", recursive=True):
    shutil.rmtree(cache_dir)
