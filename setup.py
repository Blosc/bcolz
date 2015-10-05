########################################################################
#
# License: BSD
#       Created: August 16, 2012
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

from __future__ import absolute_import

from sys import version_info as v

# Check this Python version is supported
if any([v < (2, 6), (3,) < v < (3, 3)]):
    raise Exception("Unsupported Python version %d.%d. Requires Python >= 2.6 "
                    "or >= 3.3." % v[:2])

import platform
import os
from os.path import join, abspath
from glob import glob
import sys
import re

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


# Prevent numpy from thinking it is still in its setup process:
__builtins__.__NUMPY_SETUP__ = False


class BuildExtNumpyInc(build_ext):
    def build_extensions(self):
        from numpy.distutils.misc_util import get_numpy_include_dirs
        for e in self.extensions:
            e.include_dirs.extend(get_numpy_include_dirs())

        build_ext.build_extensions(self)


# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()
# Allow setting the Blosc dir if installed in the system
BLOSC_DIR = os.environ.get('BLOSC_DIR', '')

# Sources & libraries
inc_dirs = [abspath('bcolz')]
lib_dirs = []
libs = []
def_macros = []
sources = [abspath('bcolz/carray_ext.pyx')]

optional_libs = []

# Handle --blosc=[PATH] --lflags=[FLAGS] --cflags=[FLAGS]
args = sys.argv[:]
for arg in args:
    if arg.find('--blosc=') == 0:
        BLOSC_DIR = os.path.expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    if arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    if arg.find('--cflags=') == 0:
        CFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)

if not BLOSC_DIR:
    # Compiling everything from sources
    # Blosc + BloscLZ sources
    sources += glob('c-blosc/blosc/*.c')
    # LZ4 sources
    sources += glob('c-blosc/internal-complibs/lz4*/*.c')
    # Snappy sources
    sources += glob('c-blosc/internal-complibs/snappy*/*.cc')
    # Zlib sources
    sources += glob('c-blosc/internal-complibs/zlib*/*.c')
    # Finally, add all the include dirs...
    inc_dirs += [join('c-blosc', 'blosc')]
    inc_dirs += glob('c-blosc/internal-complibs/*')
    # ...and the macros for all the compressors supported
    def_macros += [('HAVE_LZ4', 1), ('HAVE_SNAPPY', 1), ('HAVE_ZLIB', 1)]
else:
    inc_dirs.append(join(BLOSC_DIR, 'include'))
    lib_dirs.append(join(BLOSC_DIR, 'lib'))
    libs.append('blosc')

# Add -msse2 flag for optimizing shuffle in included c-blosc
# (only necessary for 32-bit Intel architectures)
if os.name == 'posix' and re.match("i.86", platform.machine()):
    CFLAGS.append("-msse2")

tests_require = []

if v < (3,):
    tests_require.extend(['unittest2', 'mock'])

# compile and link code instrumented for coverage analysis
if os.getenv('TRAVIS') and os.getenv('CI') and v[0:2] == (2, 7):
    CFLAGS.extend(["-fprofile-arcs", "-ftest-coverage"])
    LFLAGS.append("-lgcov")

setup(
    name="bcolz",
    use_scm_version={
        'version_scheme': 'guess-next-dev',
        'local_scheme': 'dirty-tag',
        'write_to': 'bcolz/version.py'
    },
    description='columnar and compressed data containers.',
    long_description="""\

bcolz provides columnar and compressed data containers.  Column
storage allows for efficiently querying tables with a large number of
columns.  It also allows for cheap addition and removal of column.  In
addition, bcolz objects are compressed by default for reducing
memory/disk I/O needs.  The compression process is carried out
internally by Blosc, a high-performance compressor that is optimized
for binary data.

""",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    author='Francesc Alted',
    author_email='francesc@blosc.org',
    maintainer='Francesc Alted',
    maintainer_email='francesc@blosc.org',
    url='https://github.com/Blosc/bcolz',
    license='BSD',
    platforms=['any'],
    ext_modules=[
        Extension(
            'bcolz.carray_ext',
            include_dirs=inc_dirs,
            define_macros=def_macros,
            sources=sources,
            library_dirs=lib_dirs,
            libraries=libs,
            extra_link_args=LFLAGS,
            extra_compile_args=CFLAGS
        )
    ],
    cmdclass={'build_ext': BuildExtNumpyInc},
    install_requires=['numpy>=1.7'],
    setup_requires=[
        'cython>=0.22',
        'numpy>=1.7',
        'setuptools>18.3',
        'setuptools-scm>1.5.4'
    ],
    tests_require=tests_require,
    extras_require=dict(
        optional=[
            'numexpr>=1.4.1',
            'pandas',
            'tables'
        ],
        test=tests_require
    ),
    packages=find_packages(),
    package_data={'bcolz': ['carray_ext.pxd']}
)
