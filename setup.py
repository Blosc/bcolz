#!/usr/bin/env python
########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: setup.py  $
#
########################################################################

import sys, os

from distutils.core import Extension
from distutils.core import setup
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs
import textwrap


# Some functions for showing errors and warnings.
def _print_admonition(kind, head, body):
    tw = textwrap.TextWrapper(
        initial_indent='   ', subsequent_indent='   ')

    print ".. %s:: %s" % (kind.upper(), head)
    for line in tw.wrap(body):
        print line

def exit_with_error(head, body=''):
    _print_admonition('error', head, body)
    sys.exit(1)

def print_warning(head, body=''):
    _print_admonition('warning', head, body)

def check_import(pkgname, pkgver):
    try:
        mod = __import__(pkgname)
    except ImportError:
            exit_with_error(
                "You need %(pkgname)s %(pkgver)s or greater to run carray!"
                % {'pkgname': pkgname, 'pkgver': pkgver} )
    else:
        if mod.__version__ < pkgver:
            exit_with_error(
                "You need %(pkgname)s %(pkgver)s or greater to run carray!"
                % {'pkgname': pkgname, 'pkgver': pkgver} )

    print ( "* Found %(pkgname)s %(pkgver)s package installed."
            % {'pkgname': pkgname, 'pkgver': mod.__version__} )
    globals()[pkgname] = mod


# Check for Python
if not (sys.version_info[0] >= 2 and sys.version_info[1] >= 6):
    exit_with_error("You need Python 2.6 or greater to install carray!")

# The minimum version of Cython required for generating extensions
min_cython_version = '0.12.1'
# The minimum version of NumPy required
min_numpy_version = '1.4'

# Check for Cython
cython = False
try:
    from Cython.Compiler.Main import Version
    cython = True
except:
    exit_with_error(
        "You need %(pkgname)s %(pkgver)s or greater to run carray!"
        % {'pkgname': 'Cython', 'pkgver': min_cython_version} )

if cython:
    if Version.version < min_cython_version:
        exit_with_error(
            "At least Cython %s is needed so as to generate extensions!"
            % (min_cython_version) )
    else:
        print ( "* Found %(pkgname)s %(pkgver)s package installed."
                % {'pkgname': 'Cython', 'pkgver': Version.version} )


# Check for NumPy
check_import('numpy', min_numpy_version)



# carray version
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('carray/version.py', 'w').write('__version__ = "%s"\n' % VERSION)


# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()
lib_dirs = []
libs = []
inc_dirs = ['carray', 'blosc']
optional_libs = []   # for linking with zlib or LZO (if I ever implemented that!)

# Handle --lflags=[FLAGS] --cflags=[FLAGS]
args = sys.argv[:]
for arg in args:
    if arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    elif arg.find('--cflags=') == 0:
        CFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)

# Add -msse2 flag for optimizing shuffle in Blosc
CFLAGS.append("-msse2")

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

carray is a container for numerical data that can be compressed
in-memory.  The compresion process is carried out internally by Blosc,
a high-performance compressor that is optimized for binary data.

Having data compressed in-memory can reduce the stress of the memory
subsystem.  The net result is that carray operations can be faster
than using a traditional ndarray object from NumPy.

""",
      classifiers = filter(None, classifiers.split("\n")),
      author = 'Francesc Alted',
      author_email = 'faltet@pytables.org',
      maintainer = 'Francesc Alted',
      maintainer_email = 'faltet@pytables.org',
      url = 'http://github.com/FrancescAlted/carray',
      license = 'http://www.opensource.org/licenses/bsd-license.php',
      download_url = 'http://github.com/downloads/FrancescAlted/carray/carray-%s.tar.gz' % VERSION,
      platforms = ['any'],
      ext_modules = [
        Extension( "carray.carrayExtension",
                   include_dirs=inc_dirs,
                   define_macros=def_macros,
                   sources = [ "carray/carrayExtension.pyx",
                               "blosc/blosc.c", "blosc/blosclz.c",
                               "blosc/shuffle.c" ],
                   depends = [ "carray/definitions.pxd",
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
