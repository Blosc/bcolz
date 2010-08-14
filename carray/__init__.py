########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: __init__,py  $
#
########################################################################


from carray.carrayExtension import (
    carray, whichLibVersion, setBloscMaxThreads)

from carray.utils import detectNumberOfCores

# Initialize Blosc
ncores = detectNumberOfCores()
setBloscMaxThreads(2)

#__version__ = getPyTablesVersion()
