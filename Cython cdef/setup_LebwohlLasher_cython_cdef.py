from distutils.core import setup
from Cython.Build import cythonize

setup(name="LebwohlLasher_cython_cdef",
      ext_modules=cythonize("LebwohlLasher_cython_cdef.pyx"))

