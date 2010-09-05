########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: __init__.py  $
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

* __version__
* ncores

Public functions
----------------

* detect_number_of_cores
* blosc_set_num_threads
* blosc_version

Public classes
--------------

* carray
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
    carray, blosc_version, blosc_set_num_threads)
from carray.ctable import ctable
from carray.utils import detect_number_of_cores
from carray.version import __version__



# Initialize Blosc
ncores = detect_number_of_cores()
blosc_set_num_threads(ncores)
