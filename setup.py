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
from glob import glob
import sys
import re

from setuptools import setup, Extension, find_packages
from pkg_resources import resource_filename

# For guessing the capabilities of the CPU for C-Blosc
import cpuinfo


class LazyCommandClass(dict):
    """
    Lazy command class that defers operations requiring Cython and numpy until
    they've actually been downloaded and installed by setup_requires.
    """
    def __contains__(self, key):
        return (
            key == 'build_ext'
            or super(LazyCommandClass, self).__contains__(key)
        )

    def __setitem__(self, key, value):
        if key == 'build_ext':
            raise AssertionError("build_ext overridden!")
        super(LazyCommandClass, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key != 'build_ext':
            return super(LazyCommandClass, self).__getitem__(key)

        from Cython.Distutils import build_ext as cython_build_ext

        class build_ext(cython_build_ext):
            """
            Custom build_ext command that lazily adds numpy's include_dir to
            extensions.
            """
            def build_extensions(self):
                """
                Lazily append numpy's include directory to Extension includes.

                This is done here rather than at module scope because setup.py
                may be run before numpy has been installed, in which case
                importing numpy and calling `numpy.get_include()` will fail.
                """
                numpy_incl = resource_filename('numpy', 'core/include')
                for ext in self.extensions:
                    ext.include_dirs.append(numpy_incl)

                # This explicitly calls the superclass method rather than the
                # usual super() invocation because distutils' build_class, of
                # which Cython's build_ext is a subclass, is an old-style class
                # in Python 2, which doesn't support `super`.
                cython_build_ext.build_extensions(self)
        return build_ext


# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()
# Allow setting the Blosc dir if installed in the system
BLOSC_DIR = os.environ.get('BLOSC_DIR', '')

# Sources & libraries
inc_dirs = ['bcolz']
lib_dirs = []
libs = []
def_macros = []
sources = ['bcolz/carray_ext.pyx']

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

if BLOSC_DIR != '':
    # Using the Blosc library
    lib_dirs += [os.path.join(BLOSC_DIR, 'lib')]
    inc_dirs += [os.path.join(BLOSC_DIR, 'include')]
    libs += ['blosc']
else:
    # Compiling everything from sources
    sources += [f for f in glob('c-blosc/blosc/*.c')
                if 'avx2' not in f and 'sse2' not in f]
    sources += glob('c-blosc/internal-complibs/lz4*/*.c')
    sources += glob('c-blosc/internal-complibs/snappy*/*.cc')
    sources += glob('c-blosc/internal-complibs/zlib*/*.c')
    sources += glob('c-blosc/internal-complibs/zstd*/*/*.c')
    inc_dirs += [os.path.join('c-blosc', 'blosc')]
    inc_dirs += [d for d in glob('c-blosc/internal-complibs/*')
                 if os.path.isdir(d)]
    inc_dirs += [d for d in glob('c-blosc/internal-complibs/zstd*/*')
                 if os.path.isdir(d)]
    def_macros += [('HAVE_LZ4', 1), ('HAVE_SNAPPY', 1), ('HAVE_ZLIB', 1),
                   ('HAVE_ZSTD', 1)]

    # Guess SSE2 or AVX2 capabilities
    cpu_info = cpuinfo.get_cpu_info()
    # SSE2
    if 'DISABLE_BCOLZ_SSE2' not in os.environ and 'sse2' in cpu_info['flags']:
        print('SSE2 detected')
        CFLAGS.append('-DSHUFFLE_SSE2_ENABLED')
        sources += [f for f in glob('c-blosc/blosc/*.c') if 'sse2' in f]
        if os.name == 'posix':
            CFLAGS.append('-msse2')
        elif os.name == 'nt':
            def_macros += [('__SSE2__', 1)]

    # AVX2
    if 'DISABLE_BCOLZ_AVX2' not in os.environ and 'avx2' in cpu_info['flags']:
        print('AVX2 detected')
        CFLAGS.append('-DSHUFFLE_AVX2_ENABLED')
        sources += [f for f in glob('c-blosc/blosc/*.c') if 'avx2' in f]
        if os.name == 'posix':
            CFLAGS.append('-mavx2')
        elif os.name == 'nt':
            def_macros += [('__AVX2__', 1)]


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
        'Development Status :: 5 - Production/Stable',
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
        'Programming Language :: Python :: 3.5',
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
    install_requires=['numpy>=1.7'],
    setup_requires=[
        'cython>=0.22',
        'numpy>=1.7',
        'setuptools>18.0',
        'setuptools-scm>1.5.4'
    ],
    tests_require=tests_require,
    extras_require=dict(
        optional=[
            'numexpr>=2.5.2',
            'dask>=0.9.0',
            'pandas',
            'tables'
        ],
        test=tests_require
    ),
    packages=find_packages(),
    package_data={'bcolz': ['carray_ext.pxd']},
    cmdclass=LazyCommandClass(),
)
