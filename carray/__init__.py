########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: __init__.py  $
#
########################################################################


from carray.carrayExtension import (
    carray, whichLibVersion, setBloscMaxThreads)

from carray.utils import detectNumberOfCores
from carray.version import __version__


# Initialize Blosc
ncores = detectNumberOfCores()
setBloscMaxThreads(ncores)
