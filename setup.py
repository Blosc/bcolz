########################################################################
#
#       License: BSD
#       Created: August 16, 2012
#       Author:  Francesc Alted - francesc@blosc.io
#
########################################################################

import sys, os

from distutils.core import Extension
from distutils.core import setup
import textwrap


########### Some utils for version checking ################

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
                "You need %(pkgname)s %(pkgver)s or greater to run bcolz!"
                % {'pkgname': pkgname, 'pkgver': pkgver} )
    else:
        if mod.__version__ < pkgver:
            exit_with_error(
                "You need %(pkgname)s %(pkgver)s or greater to run bcolz!"
                % {'pkgname': pkgname, 'pkgver': pkgver} )

    print ( "* Found %(pkgname)s %(pkgver)s package installed."
            % {'pkgname': pkgname, 'pkgver': mod.__version__} )
    globals()[pkgname] = mod


########### Check versions ##########

# The minimum version of Cython required for generating extensions
min_cython_version = '0.20'
# The minimum version of NumPy required
min_numpy_version = '1.7'
# The minimum version of Numexpr (optional)
min_numexpr_version = '1.4.1'

# Check for Python
if sys.version_info[0] == 2:
    if sys.version_info[1] < 6:
        exit_with_error("You need Python 2.6 or greater to run bcolz!")
elif sys.version_info[0] == 3:
    if sys.version_info[1] < 1:
        exit_with_error("You need Python 3.1 or greater to run bcolz!")
else:
    exit_with_error("You need Python 2.6/3.1 or greater to run bcolz!")

# Check if Cython is installed or not (requisite)
try:
    from Cython.Distutils import build_ext
    from Cython.Compiler.Main import Version
except:
    exit_with_error(
        "You need %(pkgname)s %(pkgver)s or greater to compile bcolz!"
        % {'pkgname': 'Cython', 'pkgver': min_cython_version} )

if Version.version < min_cython_version:
    exit_with_error(
        "At least Cython %s is needed so as to generate extensions!"
        % (min_cython_version) )
else:
    print ( "* Found %(pkgname)s %(pkgver)s package installed."
            % {'pkgname': 'Cython', 'pkgver': Version.version} )

# Check for NumPy
check_import('numpy', min_numpy_version)

# Check for Numexpr
numexpr_here = False
try:
    import numexpr
except ImportError:
    print_warning(
        "Numexpr is not installed.  For faster bcolz operation, "
        "please consider installing it.")
else:
    if numexpr.__version__ >= min_numexpr_version:
        numexpr_here = True
        print ( "* Found %(pkgname)s %(pkgver)s package installed."
                % {'pkgname': 'numexpr', 'pkgver': numexpr.__version__} )
    else:
        print_warning(
            "Numexpr %s installed, but version is not >= %s.  "
            "Disabling support for it." % (
            numexpr.__version__, min_numexpr_version))

########### End of checks ##########


# bcolz version
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('bcolz/version.py', 'w').write('__version__ = "%s"\n' % VERSION)


# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()
lib_dirs = []
libs = []
inc_dirs = ['blosc']
# Include NumPy header dirs
from numpy.distutils.misc_util import get_numpy_include_dirs
inc_dirs.extend(get_numpy_include_dirs())
optional_libs = []

# Handle --lflags=[FLAGS] --cflags=[FLAGS]
args = sys.argv[:]
for arg in args:
    if arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    elif arg.find('--cflags=') == 0:
        CFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)

# Add -msse2 flag for optimizing shuffle in include Blosc
if os.name == 'posix':
    CFLAGS.append("-msse2")

# Add some macros here for debugging purposes, if needed
def_macros = []


classifiers = """\
Development Status :: 4 - Beta
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
"""
setup(name = "bcolz",
      version = VERSION,
      description = 'columnar and compressed data containers.',
      long_description = """\

bcolz provides columnar and compressed data containers.  Column storage
allows for efficiently querying tables with a large number of columns.  It
also allows for cheap addition and removal of column.  In addition,
bcolz objects are compressed by default for reducing memory/disk I/O needs.
The compression process is carried out internally by Blosc,
a high-performance compressor that is optimized for binary data.

""",
      classifiers = filter(None, classifiers.split("\n")),
      author = 'Francesc Alted',
      author_email = 'francesc@blosc.io',
      maintainer = 'Francesc Alted',
      maintainer_email = 'francesc@blosc.io',
      url = 'https://github.com/Blosc/bcolz',
      license = 'http://www.opensource.org/licenses/bsd-license.php',
      # It is better to upload manually to PyPI
      #download_url = 'http://github.com/downloads/Blosc/bcolz/python-bcolz-%s.tar.gz' % (VERSION,),
      platforms = ['any'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [
        Extension( "bcolz.bcolz_ext",
                   include_dirs=inc_dirs,
                   define_macros=def_macros,
                   sources = [ "bcolz/bcolz_ext.pyx",
                               "blosc/blosc.c", "blosc/blosclz.c",
                               "blosc/shuffle.c" ],
                   depends = [ "blosc/blosc.h", "blosc/blosclz.h",
                               "blosc/shuffle.h" ],
                   library_dirs=lib_dirs,
                   libraries=libs,
                   extra_link_args=LFLAGS,
                   extra_compile_args=CFLAGS ),
        ],
      packages = ['bcolz', 'bcolz.tests'],

)

