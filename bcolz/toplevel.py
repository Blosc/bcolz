########################################################################
#
#       License: BSD
#       Created: September 10, 2010
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

"""Top level functions and classes.
"""

from __future__ import absolute_import

import sys
import os
import os.path
import glob
import itertools as it
import numpy as np
import bcolz
from bcolz.ctable import ROOTDIRS
from .py2help import xrange, _inttypes


def print_versions():
    """Print all the versions of packages that bcolz relies on."""
    print("-=" * 38)
    print("bcolz version:     %s" % bcolz.__version__)
    if bcolz.git_description:
        print("bcolz git info:    %s" % bcolz.git_description)
    print("NumPy version:     %s" % np.__version__)
    tinfo = bcolz.blosc_version()
    blosc_cnames = bcolz.blosc_compressor_list()
    print("Blosc version:     %s (%s)" % (tinfo[0], tinfo[1]))
    print("Blosc compressors: %s" % (blosc_cnames,))
    if bcolz.numexpr_here:
        print("Numexpr version:   %s" % bcolz.numexpr.__version__)
    else:
        print("Numexpr version:   not available "
              "(version >= %s not detected)" % bcolz.min_numexpr_version)
    print("Python version:    %s" % sys.version)
    if os.name == "posix":
        (sysname, nodename, release, version, machine) = os.uname()
        print("Platform:          %s-%s" % (sys.platform, machine))
    print("Byte-ordering:     %s" % sys.byteorder)
    print("Detected cores:    %s" % bcolz.detect_number_of_cores())
    print("-=" * 38)


def detect_number_of_cores():
    """
    detect_number_of_cores()

    Return the number of cores in this system.

    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default


def set_nthreads(nthreads):
    """
    set_nthreads(nthreads)

    Sets the number of threads to be used during carray operation.

    This affects to both Blosc and Numexpr (if available).  If you want to
    change this number only for Blosc, use `blosc_set_nthreads` instead.

    Parameters
    ----------
    nthreads : int
        The number of threads to be used during carray operation.

    Returns
    -------
    out : int
        The previous setting for the number of threads.

    See Also
    --------
    blosc_set_nthreads

    """
    nthreads_old = bcolz.blosc_set_nthreads(nthreads)
    if bcolz.numexpr_here:
        bcolz.numexpr.set_num_threads(nthreads)
    return nthreads_old


def open(rootdir, mode='a'):
    """
    open(rootdir, mode='a')

    Open a disk-based carray/ctable.

    Parameters
    ----------
    rootdir : pathname (string)
        The directory hosting the carray/ctable object.
    mode : the open mode (string)
        Specifies the mode in which the object is opened.  The supported
        values are:

          * 'r' for read-only
          * 'w' for emptying the previous underlying data
          * 'a' for allowing read/write on top of existing data

    Returns
    -------
    out : a carray/ctable object or IOError (if not objects are found)

    """
    # First try with a carray
    rootsfile = os.path.join(rootdir, ROOTDIRS)
    if os.path.exists(rootsfile):
        return bcolz.ctable(rootdir=rootdir, mode=mode)
    else:
        return bcolz.carray(rootdir=rootdir, mode=mode)


def fromiter(iterable, dtype, count, **kwargs):
    """
    fromiter(iterable, dtype, count, **kwargs)

    Create a carray/ctable from an `iterable` object.

    Parameters
    ----------
    iterable : iterable object
        An iterable object providing data for the carray.
    dtype : numpy.dtype instance
        Specifies the type of the outcome object.
    count : int
        The number of items to read from iterable. If set to -1, means that
        the iterable will be used until exhaustion (not recommended, see note
        below).
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray/ctable constructors.

    Returns
    -------
    out : a carray/ctable object

    Notes
    -----
    Please specify `count` to both improve performance and to save memory.  It
    allows `fromiter` to avoid looping the iterable twice (which is slooow).
    It avoids memory leaks to happen too (which can be important for large
    iterables).

    """
    _MAXINT_SIGNAL = 2**64

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
            count = _MAXINT_SIGNAL
            # If we do not have a hint on the iterable length then
            # create a couple of iterables and use the second when the
            # first one is exhausted (ValueError will be raised).
            iterable, iterable2 = it.tee(iterable)
            expected = 1000*1000   # 1 million elements

    # First, create the container
    expectedlen = kwargs.pop("expectedlen", expected)
    dtype = np.dtype(dtype)
    if dtype.kind == "V":
        # A ctable
        obj = bcolz.ctable(np.array([], dtype=dtype),
                           expectedlen=expectedlen,
                           **kwargs)
        chunklen = sum(obj.cols[name].chunklen
                       for name in obj.names) // len(obj.names)
    else:
        # A carray
        obj = bcolz.carray(np.array([], dtype=dtype),
                           expectedlen=expectedlen,
                           **kwargs)
        chunklen = obj.chunklen

    # Then fill it
    nread, blen = 0, 0
    while nread < count:
        if nread + chunklen > count:
            blen = count - nread
        else:
            blen = chunklen
        if count != _MAXINT_SIGNAL:
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
    obj.flush()
    return obj


def fill(shape, dflt=None, dtype=np.float, **kwargs):
    """
    fill(shape, dtype=float, dflt=None, **kwargs)

    Return a new carray object of given shape and type, filled with `dflt`.

    Parameters
    ----------
    shape : int
        Shape of the new array, e.g., ``(2,3)``.
    dflt : Python or NumPy scalar
        The value to be used during the filling process.  If None, values are
        filled with zeros.  Also, the resulting carray will have this value as
        its `dflt` value.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : carray
        Array filled with `dflt` values with the given shape and dtype.

    See Also
    --------
    ones, zeros

    """

    dtype = np.dtype(dtype)
    if type(shape) in _inttypes + (float,):
        shape = (int(shape),)
    else:
        shape = tuple(shape)
        if len(shape) > 1:
            # Multidimensional shape.
            # The atom will have shape[1:] dims (+ the dtype dims).
            dtype = np.dtype((dtype.base, shape[1:]+dtype.shape))
    length = shape[0]

    # Create the container
    expectedlen = kwargs.pop("expectedlen", length)
    if dtype.kind == "V" and dtype.shape == ():
        raise ValueError("fill does not support ctables objects")
    obj = bcolz.carray([], dtype=dtype, dflt=dflt, expectedlen=expectedlen,
                       **kwargs)
    chunklen = obj.chunklen

    # Then fill it
    # We need an array for the default so as to keep the atom info
    dflt = np.array(obj.dflt, dtype=dtype)
    # Making strides=(0,) below is a trick to create the array fast and
    # without memory consumption
    chunk = np.ndarray(length, dtype=dtype, buffer=dflt, strides=(0,))
    obj.append(chunk)
    obj.flush()
    return obj


def zeros(shape, dtype=np.float, **kwargs):
    """
    zeros(shape, dtype=float, **kwargs)

    Return a new carray object of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int
        Shape of the new array, e.g., ``(2,3)``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : carray
        Array of zeros with the given shape and dtype.

    See Also
    --------
    fill, ones

    """
    dtype = np.dtype(dtype)
    return fill(shape=shape, dflt=np.zeros((), dtype), dtype=dtype, **kwargs)


def ones(shape, dtype=np.float, **kwargs):
    """
    ones(shape, dtype=float, **kwargs)

    Return a new carray object of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int
        Shape of the new array, e.g., ``(2,3)``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : carray
        Array of ones with the given shape and dtype.

    See Also
    --------
    fill, zeros

    """
    dtype = np.dtype(dtype)
    return fill(shape=shape, dflt=np.ones((), dtype), dtype=dtype, **kwargs)


def arange(start=None, stop=None, step=None, dtype=None, **kwargs):
    """
    arange([start,] stop[, step,], dtype=None, **kwargs)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns a carray rather than a list.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified, `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : carray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    """

    # Check start, stop, step values
    if (start, stop) == (None, None):
        raise ValueError("You must pass a `stop` value at least.")
    elif stop is None:
        start, stop = 0, start
    elif start is None:
        start, stop = 0, stop
    if step is None:
        step = 1

    # Guess the dtype
    if dtype is None:
        if type(stop) in _inttypes:
            dtype = np.dtype(np.int_)
    dtype = np.dtype(dtype)
    stop = int(stop)

    # Create the container
    expectedlen = kwargs.pop("expectedlen", stop)
    if dtype.kind == "V":
        raise ValueError("arange does not support ctables yet.")
    else:
        obj = bcolz.carray(np.array([], dtype=dtype),
                           expectedlen=expectedlen,
                           **kwargs)
        chunklen = obj.chunklen

    # Then fill it
    incr = chunklen * step        # the increment for each chunk
    incr += step - (incr % step)  # make it match step boundary
    bstart, bstop = start, start + incr
    while bstart < stop:
        if bstop > stop:
            bstop = stop
        chunk = np.arange(bstart, bstop, step, dtype=dtype)
        obj.append(chunk)
        bstart = bstop
        bstop += incr
    obj.flush()
    return obj


def iterblocks(cobj, blen=None, start=0, stop=None):
    """iterblocks(blen=None, start=0, stop=None)

    Iterate over a `cobj` (carray/ctable) in blocks of size `blen`.

    Parameters
    ----------
    cobj : carray/ctable object
        The bcolz object to be iterated over.
    blen : int
        The length of the block that is returned.  The default is the
        chunklen, or for a ctable, the minimum of the different column
        chunklens.
    start : int
        Where the iterator starts.  The default is to start at the beginning.
    stop : int
        Where the iterator stops. The default is to stop at the end.

    Returns
    -------
    out : iterable
        This iterable returns buffers as NumPy arrays of homogeneous or
        structured types, depending on whether `cobj` is a carray or a
        ctable object.

    See Also
    --------
    whereblocks

    """

    if stop is None or stop > len(cobj):
        stop = len(cobj)
    if isinstance(cobj, bcolz.ctable):
        # A ctable object
        if blen is None:
            # Get the minimum chunklen for every column
            blen = min(cobj[col].chunklen for col in cobj.cols)
        # Create intermediate buffers for columns in a dictarray
        # (it is important that columns are contiguous)
        cbufs = {}
        for name in cobj.names:
            cbufs[name] = np.empty(blen, dtype=cobj[name].dtype)
        for i in xrange(start, stop, blen):
            buf = np.empty(blen, dtype=cobj.dtype)
            # Populate the column buffers and assign to the final buffer
            for name in cobj.names:
                cobj[name]._getrange(i, blen, cbufs[name])
                buf[name][:] = cbufs[name]
            if i + blen > stop:
                buf = buf[:stop - i]
            yield buf
    else:
        # A carray object
        if blen is None:
            blen = cobj.chunklen
        for i in xrange(start, stop, blen):
            buf = np.empty(blen, dtype=cobj.dtype)
            cobj._getrange(i, blen, buf)
            if i + blen > stop:
                buf = buf[:stop - i]
            yield buf


def walk(dir, classname=None, mode='a'):
    """walk(dir, classname=None, mode='a')

    Recursively iterate over carray/ctable objects hanging from `dir`.

    Parameters
    ----------
    dir : string
        The directory from which the listing starts.
    classname : string
        If specified, only object of this class are returned.  The values
        supported are 'carray' and 'ctable'.
    mode : string
        The mode in which the object should be opened.

    Returns
    -------
    out : iterator
        Iterator over the objects found.

    """

    # First, iterate over the carray objects in current dir
    names = os.path.join(dir, '*')
    dirs = []
    for node in glob.glob(names):
        if os.path.isdir(node):
            try:
                obj = bcolz.carray(rootdir=node, mode=mode)
            except:
                try:
                    obj = bcolz.ctable(rootdir=node, mode=mode)
                except:
                    obj = None
                    dirs.append(node)
            if obj:
                if classname:
                    if obj.__class__.__name__ == classname:
                        yield obj
                else:
                    yield obj

    # Then recurse into the true directories
    for dir_ in dirs:
        for node in walk(dir_, classname, mode):
            yield node


class cparams(object):
    """
    cparams(clevel=None, shuffle=None, cname=None)

    Class to host parameters for compression and other filters.

    Parameters
    ----------
    clevel : int (0 <= clevel < 10)
        The compression level.
    shuffle : bool
        Whether the shuffle filter is active or not.
    cname : string ('blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib')
        Select the compressor to use inside Blosc.

    In case some of the parameters are not passed, they will be
    set to a default (see `setdefaults()` method).

    See also
    --------
    cparams.setdefaults()

    """

    @property
    def clevel(self):
        """The compression level."""
        return self._clevel

    @property
    def shuffle(self):
        """Shuffle filter is active?"""
        return self._shuffle

    @property
    def cname(self):
        """The compressor name."""
        return self._cname

    @staticmethod
    def _checkparams(clevel, shuffle, cname):
        if clevel is not None:
            if not isinstance(clevel, int):
                raise ValueError("`clevel` must be an int.")
            if clevel < 0:
                raise ValueError("clevel must be a positive integer")
        if shuffle is not None:
            if not isinstance(shuffle, (bool, int)):
                raise ValueError("`shuffle` must be a boolean.")
            shuffle = bool(shuffle)
        # Store the cname as bytes object internally
        if cname is not None:
            list_cnames = bcolz.blosc_compressor_list()
            if cname not in list_cnames:
                raise ValueError(
                    "Compressor '%s' is not available in this build" % cname)
        return clevel, shuffle, cname

    @staticmethod
    def setdefaults(clevel=None, shuffle=None, cname=None):
        """Change the defaults for `clevel`, `shuffle` and `cname` params.

        Parameters
        ----------
        clevel : int (0 <= clevel < 10)
            The compression level.
        shuffle : bool
            Whether the shuffle filter is active or not.
        cname : string ('blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib')
            Select the compressor to use inside Blosc.

        If this method is not called, the defaults will be set as in
        defaults.py (``{clevel=5, shuffle=True, cname='blosclz'}``).

        """
        clevel, shuffle, cname = cparams._checkparams(clevel, shuffle, cname)
        dflts = bcolz.defaults.cparams
        if clevel is not None:
            dflts['clevel'] = clevel
        if shuffle is not None:
            dflts['shuffle'] = shuffle
        if cname is not None:
            dflts['cname'] = cname

    def __init__(self, clevel=None, shuffle=None, cname=None):
        clevel, shuffle, cname = cparams._checkparams(clevel, shuffle, cname)
        dflts = bcolz.defaults.cparams
        self._clevel = dflts['clevel'] if clevel is None else clevel
        self._shuffle = dflts['shuffle'] if shuffle is None else shuffle
        self._cname = dflts['cname'] if cname is None else cname

    def __repr__(self):
        args = ["clevel=%d" % self._clevel,
                "shuffle=%s" % self._shuffle,
                "cname='%s'" % self._cname,
                ]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(args))


# Local Variables:
# mode: python
# tab-width: 4
# fill-column: 78
# End:
