from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
#Changed the name of the file
ext_modules = [
    Extension(
        "LebwohlLasher_cython_cdef_omp",
        ["LebwohlLasher_cython_cdef_omp.pyx"],
        extra_compile_args=['-fopenmp','-v'],
#        extra_link_args=['-fopenmp'],
        extra_link_args=['-lgomp', '-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/13/'],
    )
]

setup(name="LebwohlLasher_cython_cdef_omp",
      ext_modules=cythonize(ext_modules))
