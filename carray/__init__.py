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

Public classes
--------------

carray(...)

"""


from carray.carrayExtension import (
    carray, whichLibVersion, setBloscMaxThreads)

from carray.utils import detectNumberOfCores
from carray.version import __version__


# Initialize Blosc
ncores = detectNumberOfCores()
setBloscMaxThreads(ncores)
