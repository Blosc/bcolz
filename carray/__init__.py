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
* default_vm : the virtual machine to be used in computations
* min_numexpr_version : the minimum version of numexpr needed
* ncores : the number of detected cores
* numexpr_here : whether minimum version of numexpr has been detected

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

min_numexpr_version = '1.4.1'  # the minimum version of Numexpr needed
numexpr_here = False
default_vm = "python"
try:
    import numexpr
except ImportError:
    pass
else:
    if numexpr.__version__ >= min_numexpr_version:
        numexpr_here = True
        default_vm = "numexpr"

from carray.carrayExtension import (
    carray, blosc_version, _blosc_set_nthreads as blosc_set_nthreads)
from carray.ctable import ctable
from carray.toplevel import (
    detect_number_of_cores, set_nthreads,
    fromiter, arange, zeros, ones, fill,
    cparams, eval, set_vm)
from carray.version import __version__
from carray.tests import test


# Initialize Blosc
ncores = detect_number_of_cores()
blosc_set_nthreads(ncores)
