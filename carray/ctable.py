########################################################################
#
#       License: BSD
#       Created: September 01, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: ctable.py  $
#
########################################################################

"""The ctable module.

Public classes:

    ctable

"""

import sys, math

import numpy as np
import carray as ca

if ca.numexpr_here:
    from numexpr.expressions import functions as numexpr_functions

# The size of the columns chunks to be used in `ctable.eval()`, in
# bytes.  For optimal performance, set this so that it will not exceed
# the size of your L2/L3 (whichever is larger) cache.
#EVAL_BLOCK_SIZE = 16            # use this for testing purposes
EVAL_BLOCK_SIZE = 3*1024*1024    # 3 MB represents a good average?


class ctable(object):
    """
    This class represents a compressed, column-wise, in-memory table.

    Instance variables
    ------------------

    cols
        The column carrays (dict).

    dtype
        The data type of this ctable (numpy dtype).

    names
        The names of the columns (list).

    nrows
        The number of rows of this ctable (int).

    shape
        The shape of this ctable (tuple).

    Public methods
    --------------

    * addcol(newcol[, name][, pos])
    * delcol([name][, pos])
    * append(rows)

    Special methods
    ---------------

    * __getitem__(key)
    * __setitem__(key, value)

    """

    # Properties
    # ``````````

    @property
    def dtype(self):
        "The data type of this ctable (numpy dtype)."
        names, cols = self.names, self.cols
        l = [(name, cols[name].dtype) for name in names]
        return np.dtype(l)

    @property
    def shape(self):
        "The shape of this ctable."
        return (self.nrows,)

    @property
    def nbytes(self):
        "The original (uncompressed) size of this carray (in bytes)."
        return self.get_stats()[0]

    @property
    def cbytes(self):
        "The compressed size of this carray (in bytes)."
        return self.get_stats()[1]


    def get_stats(self):
        """Get some stats (nbytes, cbytes and ratio) about this carray."""
        nbytes, cbytes, ratio = 0, 0, 0.0
        names, cols = self.names, self.cols
        for name in names:
            column = cols[name]
            nbytes += column.nbytes
            cbytes += column.cbytes
        cratio = nbytes / float(cbytes)
        return (nbytes, cbytes, cratio)


    def __init__(self, cols, names=None, **kwargs):
        """Create a new ctable from `cols` with optional `names`.

        `cols` can be a tuple/list of carrays or NumPy arrays.  It can
        also be a NumPy structured array.

        If `names` is passed, this will be taken as the list of names
        for the columns.

        If new carrays needs to be built, you can pass whatever
        additional arguments supported by carray constructors in
        `kwargs`.
        """

        self.names = []
        """The names of the columns (list)."""
        self.cols = {}
        """The carray columns (dict)."""
        self.nrows = 0
        """The number of rows (int)."""

        # Get the names of the cols
        if names is None:
            if isinstance(cols, np.ndarray):  # ratype case
                names = list(cols.dtype.names)
            else:
                names = ["f%d"%i for i in range(len(cols))]
        else:
            if type(names) != list:
                try:
                    names = list(names)
                except:
                    raise ValueError, "cannot convert `names` into a list"
            if len(names) != len(cols):
                raise ValueError, "`cols` and `names` must have the same length"
        self.names = names

        # Guess the kind of cols input
        calist, nalist, ratype = False, False, False
        if type(cols) in (tuple, list):
            calist = [type(v) for v in cols] == [ca.carray for v in cols]
            nalist = [type(v) for v in cols] == [np.ndarray for v in cols]
        elif isinstance(cols, np.ndarray):
            ratype = hasattr(cols.dtype, "names")
        else:
            raise ValueError, "`cols` input is not supported"
        if not (calist or nalist or ratype):
            raise ValueError, "`cols` input is not supported"

        # Populate the columns
        clen = -1
        for i, name in enumerate(names):
            if calist:
                column = cols[i]
            elif nalist:
                column = cols[i]
                if column.dtype == np.void:
                    raise ValueError, "`cols` elements cannot be of type void"
                column = ca.carray(column, **kwargs)
            elif ratype:
                column = ca.carray(cols[name], **kwargs)
            self.cols[name] = column
            if clen >= 0 and clen != len(column):
                raise ValueError, "all `cols` must have the same length"
            clen = len(column)
        self.nrows += clen

        # Cache a structured array of len 1 for ctable[int] acceleration
        self._arr1 = np.empty(shape=(1,), dtype=self.dtype)


    def append(self, rows):
        """Append `rows` to ctable.

        `rows` can be a collection of scalar values, NumPy arrays or
        carrays.  It also can be a NumPy record, a NumPy recarray, or
        another ctable.
        """

        # Guess the kind of rows input
        calist, nalist, sclist, ratype = False, False, False, False
        if type(rows) in (tuple, list):
            calist = [type(v) for v in rows] == [ca.carray for v in rows]
            nalist = [type(v) for v in rows] == [np.ndarray for v in rows]
            if not (calist or nalist):
                # Try with a scalar list
                sclist = True
        elif isinstance(rows, np.ndarray):
            ratype = hasattr(rows.dtype, "names")
        elif isinstance(rows, ca.ctable):
            # Convert int a list of carrays
            rows = [rows[name] for name in self.names]
            calist = True
        else:
            raise ValueError, "`rows` input is not supported"
        if not (calist or nalist or sclist or ratype):
            raise ValueError, "`rows` input is not supported"

        # Populate the columns
        clen = -1
        for i, name in enumerate(self.names):
            if calist or sclist:
                column = rows[i]
            elif nalist:
                column = rows[i]
                if column.dtype == np.void:
                    raise ValueError, "`rows` elements cannot be of type void"
                column = column
            elif ratype:
                column = rows[name]
            self.cols[name].append(column)
            if sclist:
                clen2 = 1
            else:
                clen2 = len(column)
            if clen >= 0 and clen != clen2:
                raise ValueError, "all cols in `rows` must have the same length"
            clen = clen2
        self.nrows += clen


    def addcol(self, newcol, name=None, pos=None):
        """Add a new `newcol` carray or ndarray as column.

        If `name` is specified, the column will have this name.  If
        not, it will receive an automatic name.

        If `pos` is specified, the column will be placed in this
        position.  If not, it will be appended at the end.
        """

        # Check params
        if pos is None:
            pos = len(self.names)
        else:
            if pos and type(pos) != int:
                raise ValueError, "`pos` must be an int"
            if pos < 0 or pos > len(self.names):
                raise ValueError, "`pos` must be >= 0 and <= len(self.cols)"
        if name is None:
            name = "f%d" % pos
        else:
            if type(name) != str:
                raise ValueError, "`name` must be a string"
        if name in self.names:
            raise ValueError, "'%s' column already exists" % name
        if len(newcol) != self.nrows:
            raise ValueError, "`newcol` must have the same length than ctable"

        if isinstance(newcol, np.ndarray):
            newcol = ca.carray(newcol)

        # Insert the column
        self.names.insert(pos, name)
        self.cols[name] = newcol


    def delcol(self, name=None, pos=None):
        """Remove a column.

        If `name` is specified, the column with this name is removed.
        If `pos` is specified, the column in this position is removed.
        You must specify at least a `name` or a `pos`, and you should
        not specify both at the same time.
        """
        if name is None and pos is None:
            raise ValueError, "specify either a `name` or a `pos`"
        if name is not None and pos is not None:
            raise ValueError, "you cannot specify both a `name` and a `pos`"
        if name:
            if type(name) != str:
                raise ValueError, "`name` must be a string"
            if name not in self.names:
                raise ValueError, "`name` not found in columns"
            pos = self.names.index(name)
        elif pos is not None:
            if type(pos) != int:
                raise ValueError, "`pos` must be an int"
            if pos < 0 or pos > len(self.names):
                raise ValueError, "`pos` must be >= 0 and <= len(self.cols)"
            name = self.names[pos]

        # Remove the column
        self.names.pop(pos)
        del self.cols[name]


    def copy(self, **kwargs):
        """Return a copy of self.

        You can pass whatever additional arguments supported by
        carray/ctable constructors in `kwargs`.
        """

        # Remove possible unsupported args for columns
        names = kwargs.pop('names', self.names)
        # Copy the columns
        cols = [ self.cols[name].copy(**kwargs) for name in self.names ]
        # Remove unsupported params for ctable constructor
        kwargs.pop('cparms', None)
        # Create the ctable
        ccopy = ctable(cols, names, **kwargs)
        return ccopy


    def __len__(self):
        """Return the length of self."""
        return self.nrows


    def __sizeof__(self):
        """Return the number of bytes taken by self."""
        return self.cbytes


    def _getif(self, boolarr):
        """Return rows where `boolarr` is true as an structured array.

        This is called internally only, so we can assum that `boolarr`
        is a boolean array.
        """

        cols = []
        for name in self.names:
            cols.append(self.cols[name][boolarr])
        result = np.rec.fromarrays(cols, dtype=self.dtype).view(np.ndarray)

        return result


    def __getitem__(self, key):
        """Get a row or a range of rows.  Also a column or range of columns.

        If `key` argument is an integer, the corresponding ctable row
        is returned as a NumPy record.  If `key` is a slice, the range
        of rows determined by it is returned as a NumPy structured
        array.

        If `key` is a string, the corresponding ctable column name
        will be returned.  If `key` is not a colname, it will be
        interpreted as a boolean expression and the rows fulfilling it
        will be returned as a NumPy structured array.

        If `key` is a list of strings, the specified column names will
        be returned as a new ctable object.
        """

        # First, check for integer
        if isinstance(key, int):
            # Get a copy of the len-1 array
            ra = self._arr1.copy()
            # Fill it
            ra[0] = tuple([self.cols[name][key] for name in self.names])
            return ra[0]
        # Slices
        elif type(key) == slice:
            (start, stop, step) = key.start, key.stop, key.step
            if step and step <= 0 :
                raise NotImplementedError("step in slice can only be positive")
        # Multidimensional keys
        elif isinstance(key, tuple):
            if len(key) != 1:
                raise IndexError, "multidimensional keys are not supported"
            return self[key[0]]
        # List of integers (case of fancy indexing), or list of column names
        elif type(key) is list:
            if len(key) == 0:
                return np.empty(0, self.dtype)
            strlist = [type(v) for v in key] == [str for v in key]
            # Range of column names
            if strlist:
                cols = [self.cols[name] for name in key]
                return ctable(cols, key)
            # Try to convert to a integer array
            try:
                key = np.array(key, dtype=np.int_)
            except:
                raise IndexError, \
                      "key cannot be converted to an array of indices"
            return np.fromiter((self[i] for i in key),
                               dtype=self.dtype, count=len(key))
        # A boolean array (case of fancy indexing)
        elif hasattr(key, "dtype"):
            if key.dtype.type == np.bool_:
                return self._getif(key)
            elif np.issubsctype(key, np.int_):
                # An integer array
                return np.array([self[i] for i in key], dtype=self.dtype)
            else:
                raise IndexError, \
                      "arrays used as indices must be integer (or boolean)"
        # Column name
        elif type(key) is str:
            if key not in self.names:
                # key is not a column name, try to evaluate
                arr = self.eval(key)
                if arr.dtype.type != np.bool_:
                    raise IndexError, \
                          "`key` %s does not represent a boolean expression" %\
                          key
                return self._getif(arr)
            return self.cols[key]
        # All the rest not implemented
        else:
            raise NotImplementedError, "key not supported: %s" % repr(key)

        # From now on, will only deal with [start:stop:step] slices

        # Get the corrected values for start, stop, step
        (start, stop, step) = slice(start, stop, step).indices(self.nrows)
        # Build a numpy container
        n = ca.utils.get_len_of_range(start, stop, step)
        ra = np.empty(shape=(n,), dtype=self.dtype)
        # Fill it
        for name in self.names:
            ra[name][:] = self.cols[name][start:stop:step]

        return ra


    def __setitem__(self, key, value):
        """Set a row or a range of rows."""

        # First, convert value into a structured array
        value = ca.utils.to_ndarray(value, self.dtype)
        # Then, modify the rows
        for name in self.names:
            self.cols[name][key] = value[name]
        return


    def _getvars(self, expression, depth=2):
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
        colnames = []
        for var in exprvars:
            # Get the value.
            if var in self.cols:
                val = self.cols[var]
                colnames.append(var)
            elif var in user_locals:
                val = user_locals[var]
            elif var in user_globals:
                val = user_globals[var]
            else:
                raise NameError("name ``%s`` is not found" % var)
            # Check the value.
            if hasattr(val, 'dtype') and val.dtype.str[1:] == 'u8':
                raise NotImplementedError(
                    "variable ``%s`` refers to "
                    "a 64-bit unsigned integer object, that is "
                    "not yet supported in expressions, sorry; " % var )
            reqvars[var] = val
        return reqvars, colnames


    def eval(self, expression, **kwargs):
        """Evaluate the `expression` on columns and return the result.

        `expression` must be a string containing an expression
        supported by Numexpr.  It may contain columns or other carrays
        or NumPy arrays that can be found in the user space name.

        The output is a carray object containing the result of the
        expression.  You taylor this carray by passing additional
        arguments supported by carray constructor in `kwargs`.
        """

        if not ca.numexpr_here:
            raise ImportError(
                "You need numexpr %s or higher to use this method" % \
                ca.min_numexpr_version)

        # Get variables and column names participating in expression
        vars, colnames = self._getvars(expression)

        # Compute the optimal block size (in elements)
        typesize = sum(self.cols[name].dtype.itemsize for name in colnames)
        bsize = EVAL_BLOCK_SIZE // typesize
        # Evaluation seems more efficient if block size is a power of 2
        bsize = 2 ** (int(math.log(bsize, 2)))

        # Perform the evaluation in blocks
        vars_ = {}
        for i in xrange(0, self.nrows, bsize):
            # Get buffers for columns
            for name in vars.iterkeys():
                var = vars[name]
                if name in colnames:
                    vars_[name] = self.cols[name][i:i+bsize]
                elif hasattr(var, "__len__") and len(var) > bsize:
                    vars_[name] = var[i:i+bsize]
                else:
                    vars_[name] = var
            # Perform the evaluation for this block
            res_block = ca.numexpr.evaluate(expression, local_dict=vars_)
            if i == 0:
                # Get a decent default for expectedrows
                nrows = kwargs.pop('expectedrows', self.nrows)
                result = ca.carray(res_block, expectedrows=nrows, **kwargs)
            else:
                result.append(res_block)

        return result


    def __str__(self):
        """Represent the ctable as an string."""
        if self.nrows > 100:
            return "[%s, %s, %s, ..., %s, %s, %s]\n" % \
                   (self[0], self[1], self[2], self[-3], self[-2], self[-1])
        else:
            return str(self[:])


    def __repr__(self):
        """Represent the carray as an string, with additional info."""
        nbytes, cbytes, cratio = self.get_stats()
        snbytes = ca.utils.human_readable_size(nbytes)
        scbytes = ca.utils.human_readable_size(cbytes)
        fullrepr = """ctable(%s, %s)
  nbytes: %s; cbytes: %s; ratio: %.2f
%s""" % (self.shape, self.dtype, snbytes, scbytes, cratio, str(self))
        return fullrepr
