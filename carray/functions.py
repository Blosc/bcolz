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


def fromiter(iterable, dtype, count=-1, **kwargs):
    """
    fromiter(iterable, dtype, count=-1, **kwargs)

    Create a carray/ctable from an `iterable` object.

    Parameters
    ----------
    iterable : iterable object
        An iterable object providing data for the carray.

    dtype : numpy.dtype instance
        Specifies the type of the outcome object.

    count : int, optional
        Specifies the number of items to read from iterable. The
        default is -1, which means all data is read.

    kwargs : list of parameters or dictionary
        Any parameter supported by the carray/ctable constructors.

    Returns
    -------
    out : a carray/ctable object

    Notes
    -----
    Specify `count` to improve performance.  It allows `fromiter` to
    avoid looping the iterable twice (which is slooow).

    """

    # Check for a true iterable
    if not hasattr(iterable, "next"):
        iterable = iter(iterable)

    # Try to guess the final length
    expected = count
    if count == -1:
        # Try to guess the size of the iterable length
        if hasattr(iterable, "__length_hint__"):
            count = iterable.__length_hint__()
            expected = count
        else:
            # No guess
            count = sys.maxint
            # If we do not have a hint on the iterable length then
            # create a couple of iterables and use the second when the
            # first one is exhausted (ValueError will be raised).
            iterable, iterable2 = it.tee(iterable)
            expected = 10*1000*1000   # 10 million elements

    # First, create the container
    expectedlen = kwargs.pop("expectedlen", expected)
    obj = ca.carray(np.array([], dtype=dtype),
                    expectedlen=expectedlen,
                    **kwargs)

    # Then fill it
    chunklen = obj.chunklen
    nread, blen = 0, 0
    while nread < count:
        if nread + chunklen > count:
            blen = count - nread
        else:
            blen = chunklen
        if count != sys.maxint:
            chunk = np.fromiter(iterable, dtype=dtype, count=blen)
        else:
            try:
                chunk = np.fromiter(iterable, dtype=dtype, count=blen)
            except ValueError:
                # Positionate in second iterable
                iter2 = it.islice(iterable2, nread, None, 1)
                # We are reaching the end, use second iterable now
                chunk = np.fromiter(iter2, dtype=dtype, count=-1)
        obj.append(chunk)
        nread += len(chunk)
        # Check the end of the iterable
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


