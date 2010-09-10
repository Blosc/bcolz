########################################################################
#
#       License: BSD
#       Created: September 10, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################

"""Public functions.
"""

import sys, os
import itertools as it
import numpy as np
import carray as ca


def detect_number_of_cores():
    """Detect the number of cores on a system."""
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else: # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
        if ncpus > 0:
            return ncpus
    return 1 # Default


def set_num_threads(nthreads):
    """Set the number of threads to be used during carray operation.

    This affects to both Blosc and Numexpr (if available).  If you want
    to change this number only for Blosc, use `blosc_set_number_threads`
    instead.
    """
    ca.blosc_set_num_threads(nthreads)
    if ca.numexpr_here:
        ca.numexpr.set_num_threads(nthreads)


def fromiter(iterator, dtype, count=-1, **kwargs):
    """Create a carray/ctable from `iterator` object.

    `dtype` specifies the type of the outcome object.

    `count` specifies the number of items to read from iterable. The
    default is -1, which means all data is read.

    You can pass whatever additional arguments supported by
    carray/ctable constructors in `kwargs`.
    """

    if count == -1:
        # Try to guess the size of the iterator length
        if hasattr(iterator, "__length_hint__"):
            count = iterator.__length_hint__()
        else:
            # No guess
            count = sys.maxint

    # First, create the container
    obj = ca.carray(np.array([], dtype=dtype), **kwargs)
    chunklen = obj.chunklen
    nread, blen = 0, 0
    while nread < count:
        if count == sys.maxint:
            blen = -1
        elif nread + chunklen > count:
            blen = count - nread
        else:
            blen = chunklen
        chunkiter = it.islice(iterator, blen)
        chunk = np.fromiter(chunkiter, dtype=dtype, count=blen)
        obj.append(chunk)
        nread += len(chunk)
        # Check the end of the iterator
        if len(chunk) < chunklen:
            break
    return obj


class cparms(object):
    """Class to host parameters for compression and other filters.

    You can pass the `clevel` and `shuffle` params to the constructor.
    If you do not pass them, the defaults are ``5`` and ``True``
    respectively.

    It offers these read-only attributes::

      * clevel: the compression level

      * shuffle: whether the shuffle filter is active or not

    """

    @property
    def clevel(self):
        """The compression level."""
        return self._clevel

    @property
    def shuffle(self):
        """Shuffle filter is active?"""
        return self._shuffle

    def __init__(self, clevel=5, shuffle=True):
        """Create an instance with `clevel` and `shuffle` params."""
        if not isinstance(clevel, int):
            raise ValueError, "`clevel` must an int."
        if not isinstance(shuffle, (bool, int)):
            raise ValueError, "`shuffle` must a boolean."
        shuffle = bool(shuffle)
        if clevel < 0:
            raiseValueError, "clevel must be a positive integer"
        self._clevel = clevel
        self._shuffle = shuffle

    def __repr__(self):
        args = ["clevel=%d"%self._clevel, "shuffle=%s"%self._shuffle]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(args))


