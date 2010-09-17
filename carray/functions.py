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
import math

if ca.numexpr_here:
    from numexpr.expressions import functions as numexpr_functions


# The size of the columns chunks to be used in `ctable.eval()`, in
# bytes.  For optimal performance, set this so that it will not exceed
# the size of your L2/L3 (whichever is larger) cache.
#EVAL_BLOCK_SIZE = 16            # use this for testing purposes
EVAL_BLOCK_SIZE = 3*1024*1024    # 3 MB represents a good average?


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
    dtype = np.dtype(dtype)
    if dtype.kind == "V":
        # A ctable
        obj = ca.ctable(np.array([], dtype=dtype),
                        expectedlen=expectedlen,
                        **kwargs)
        chunklen = sum(obj.cols[name].chunklen
                       for name in obj.names) // len(obj.names)
    else:
        # A carray
        obj = ca.carray(np.array([], dtype=dtype),
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


def _getvars(expression, user_dict, depth):
    """Get the variables in `expression`.

    `depth` specifies the depth of the frame in order to reach local
    or global variables.
    """

    cexpr = compile(expression, '<string>', 'eval')
    exprvars = [ var for var in cexpr.co_names
                 if var not in ['None', 'False', 'True']
                 and var not in numexpr_functions ]

    # Get the local and global variable mappings of the user frame
    user_locals, user_globals = {}, {}
    user_frame = sys._getframe(depth)
    user_locals = user_frame.f_locals
    user_globals = user_frame.f_globals

    # Look for the required variables
    reqvars = {}
    for var in exprvars:
        # Get the value.
        if var in user_dict:
            val = user_dict[var]
        elif var in user_locals:
            val = user_locals[var]
        elif var in user_globals:
            val = user_globals[var]
        else:
            raise NameError("variable name ``%s`` not found" % var)
        # Check the value.
        if hasattr(val, 'dtype') and val.dtype.str[1:] == 'u8':
            raise NotImplementedError(
                "variable ``%s`` refers to "
                "a 64-bit unsigned integer object, that is "
                "not yet supported in expressions, sorry; " % var )
        reqvars[var] = val
    return reqvars


def eval(expression, **kwargs):
    """
    eval(expression, **kwargs)

    Evaluate a Numexpr `expression` and return the result.

    Parameters
    ----------
    expression : string
        A string forming an expression, like '2*a+3*b'. The values for 'a' and
        'b' are variable names to be taken from the calling function's frame.
        These variables may be scalars, carrays or NumPy arrays.
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : carray object
        The outcome of the expression.  You can taylor the
        properties of this carray by passing additional arguments
        supported by carray constructor in `kwargs`.

    """

    if not ca.numexpr_here:
        raise ImportError(
            "You need numexpr %s or higher to use this method" % \
            ca.min_numexpr_version)

    # Get variables and column names participating in expression
    user_dict = kwargs.pop('user_dict', {})
    depth = kwargs.pop('depth', 2)
    vars = _getvars(expression, user_dict, depth)

    # Gather info about sizes and lengths
    typesize, vlen = 0, 1
    for name in vars.iterkeys():
        var = vars[name]
        if hasattr(var, "dtype"):  # numpy/carray arrays
            typesize += var.dtype.itemsize
        elif hasattr(var, "__len__"): # sequence
            arr = np.array(var[0])
            typesize += arr.dtype.itemsize
        if hasattr(var, "__len__"):
            if vlen > 1 and vlen != len(var):
                raise ValueError, "sequences must have the same length"
            vlen = len(var)

    if typesize == 0:
        # All scalars
        return ca.numexpr.evaluate(expression, local_dict=vars)

    # Compute the optimal block size (in elements)
    bsize = EVAL_BLOCK_SIZE // typesize
    # Evaluation seems more efficient if block size is a power of 2
    bsize = 2 ** (int(math.log(bsize, 2)))

    # Perform the evaluation in blocks
    vars_ = {}
    expectedlen = bsize    # a default
    for i in xrange(0, vlen, bsize):
        # Get buffers for columns
        for name in vars.iterkeys():
            var = vars[name]
            if hasattr(var, "__len__") and len(var) > bsize:
                vars_[name] = var[i:i+bsize]
                expectedlen = len(var)
            else:
                vars_[name] = var
        # Perform the evaluation for this block
        res_block = ca.numexpr.evaluate(expression, local_dict=vars_)
        if i == 0:
            # Get a decent default for expectedlen
            nrows = kwargs.pop('expectedlen', expectedlen)
            result = ca.carray(res_block, expectedlen=nrows, **kwargs)
        else:
            result.append(res_block)

    return result


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



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 78
## End:
