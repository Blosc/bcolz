########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - francesc@blosc.io
#
########################################################################

"""
bcolz: columnar and compressed data containers
==============================================

bcolz provides columnar and compressed data containers.  Column storage
allows for efficiently querying tables with a large number of columns.  It
also allows for cheap addition and removal of column.  In addition,
bcolz objects are compressed by default for reducing memory/disk I/O needs.
The compression process is carried out internally by Blosc,
a high-performance compressor that is optimized for binary data.

"""

min_numexpr_version = '1.4.1'  # the minimum version of Numexpr needed
numexpr_here = False
try:
    import numexpr
except ImportError:
    pass
else:
    if numexpr.__version__ >= min_numexpr_version:
        numexpr_here = True

# Print array functions (imported from NumPy)
from bcolz.arrayprint import (
    array2string, set_printoptions, get_printoptions )

from bcolz.carray_ext import (
    carray, blosc_version, blosc_compressor_list,
    _blosc_set_nthreads as blosc_set_nthreads,
    _blosc_init, _blosc_destroy)
from bcolz.ctable import ctable
from bcolz.toplevel import (
    detect_number_of_cores, set_nthreads,
    open, fromiter, arange, zeros, ones, fill,
    iterblocks, cparams, eval, walk)
from bcolz import defaults
from bcolz.version import __version__
from bcolz.tests import test

# Initialization code for the Blosc library
_blosc_init()
ncores = detect_number_of_cores()
blosc_set_nthreads(ncores)
import atexit
atexit.register(_blosc_destroy)
