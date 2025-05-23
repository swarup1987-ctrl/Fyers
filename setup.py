from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="cycore",
    ext_modules=cythonize("cycore.pyx"),
    include_dirs=[numpy.get_include()],
)
