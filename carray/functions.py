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
    """
    detect_number_of_cores()

    Detect the number of cores in this system.

    Returns
    -------
    out : int
        The number of cores in this system.

    """
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
    """
    set_num_threads(nthreads)

    Set the number of threads to be used during carray operation.

    This affects to both Blosc and Numexpr (if available).  If you want
    to change this number only for Blosc, use `blosc_set_number_threads`
    instead.

    Parameters
    ----------
    nthreads : int
        The number of threads to be used during carray operation.

    See also
    --------
    blosc_set_number_threads

    """
    ca.blosc_set_num_threads(nthreads)
    if ca.numexpr_here:
        ca.numexpr.set_num_threads(nthreads)


def fromiter(iterator, dtype, count=-1, **kwargs):
    """
    fromiter(iterator, dtype, count=-1, **kwargs)

    Create a carray/ctable from an `iterator` object.

    Parameters
    ----------
    dtype : numpy.dtype instance
        Specifies the type of the outcome object.

    count : int
        Specifies the number of items to read from iterable. The
        default is -1, which means all data is read.

    kwargs : list of parameters or dictionary
        Any parameter supported by the carray/ctable constructors.

    Returns
    -------
    out : a carray/ctable object

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
    """
    cparms(clevel=5, shuffle=True)

    Class to host parameters for compression and other filters.

    Parameters
    ----------
    clevel : int (0 <= clevel < 10)
        The compression level.

    shuffle : bool
        Whether the shuffle filter is active or not.

    Notes
    -----
    The shuffle filter may be automatically disable in case it is
    non-sense to use it (e.g. itemsize == 1).

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


