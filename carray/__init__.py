########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################

"""
carray: a compressed and enlargeable in-memory data container
=============================================================

carray is a container for numerical data that can be compressed
in-memory.  The compresion process is carried out internally by Blosc,
a high-performance compressor that is optimized for binary data.

Public variables
----------------

* __version__ : the version of carray package
* ncores : the number of detected cores

Public functions
----------------

* blosc_set_nthreads
* blosc_version
* detect_number_of_cores
* fromiter
* set_nthreads

Public classes
--------------

* carray
* cparams
* ctable

"""

# The minimum version of Numexpr required
min_numexpr_version = '1.4'
numexpr_here = False
try:
    import numexpr
except ImportError:
    pass
else:
    if numexpr.__version__ >= min_numexpr_version:
        numexpr_here = True

from carray.carrayExtension import (
    carray, blosc_version, _blosc_set_nthreads as blosc_set_nthreads)
from carray.ctable import ctable
from carray.toplevel import (
    detect_number_of_cores, set_nthreads, fromiter,
    arange, zeros, fill,
    cparams, eval)
from carray.version import __version__
from carray.tests import test


# Initialize Blosc
ncores = detect_number_of_cores()
blosc_set_nthreads(ncores)
