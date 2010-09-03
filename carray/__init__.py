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

* detectNumberOfCores
* setBloscMaxThreads
* whichLibVersion

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
    carray, whichLibVersion, setBloscMaxThreads)
from carray.ctable import ctable
from carray.utils import detectNumberOfCores
from carray.version import __version__



# Initialize Blosc
ncores = detectNumberOfCores()
setBloscMaxThreads(ncores)
