#!/usr/bin/env python
#----------------------------------------------------------------------
# Setup script for the tables package

import sys, os

from distutils.core import Extension
from distutils.core import setup
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs


# Version
VERSION = open('VERSION').read().strip()

# Global variables
CFLAGS = ""
LFLAGS = ""
lib_dirs = []
libs = []
inc_dirs = ['carray', 'blosc']
optional_libs = []   # for linking with zlib or LZO (if I ever implemented that!)

# Include NumPy header dirs 
inc_dirs.extend(get_numpy_include_dirs())

def_macros = [('NDEBUG', 1)]
# Define macros for Windows platform
if os.name == 'nt':
    def_macros.append(('WIN32', 1))


classifiers = """\
Development Status :: 1 - Alpha
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
"""
setup(name = "carray",
      version = VERSION,
      description = 'Compressed in-memory array',
      long_description = """\

carray is a container for data that can be compressed.  Nowadays
memory access is the most common bottleneck in many computational
scenarios, and CPUs spend most of its time waiting for data.

Having data compressed in-memory can reduce the stress of the memory
subsystem.  The net result is that carray operations can be faster
than using a traditional ndarray object from NumPy.

""",
      classifiers = filter(None, classifiers.split("\n")),
      author = 'Francesc Alted',
      author_email = 'faltet@pytables.org',
      maintainer = 'Francesc Alted',
      maintainer_email = 'faltet@pytables.org',
      url = 'http://carray.pytables.org/',
      license = 'http://www.opensource.org/licenses/bsd-license.php',
      download_url = "http://carray.pytables.org/download/stable/carray-%s.tar.gz" % VERSION,
      platforms = ['any'],
      ext_modules = [
        Extension( "carray.carrayExtension",
                   include_dirs=inc_dirs,
                   define_macros=def_macros,
                   sources = [ "carray/carrayExtension.pyx", "carray/utils.c",
                               "blosc/blosc.c", "blosc/blosclz.c",
                               "blosc/shuffle.c" ],
                   depends = [ "carray/definitions.pxd", "carray/utils.h",
                               "blosc/blosc.h", "blosc/blosclz.h",
                               "blosc/shuffle.h" ],
                   library_dirs=lib_dirs,
                   libraries=libs,
                   extra_link_args=LFLAGS,
                   extra_compile_args=CFLAGS ),
        ],
      cmdclass = {'build_ext': build_ext},
      packages = ['carray', 'carray.tests'],

)
