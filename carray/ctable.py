########################################################################
#
#       License: BSD
#       Created: September 01, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################

import sys, math

import numpy as np
import carray as ca
from carray import utils

if ca.numexpr_here:
    from numexpr.expressions import functions as numexpr_functions

# The size of the columns chunks to be used in `ctable.eval()`, in
# bytes.  For optimal performance, set this so that it will not exceed
# the size of your L2/L3 (whichever is larger) cache.
#EVAL_BLOCK_SIZE = 16            # use this for testing purposes
EVAL_BLOCK_SIZE = 3*1024*1024    # 3 MB represents a good average?


class ctable(object):
    """
    ctable(cols, names=None, **kwargs)

    This class represents a compressed, column-wise, in-memory table.

    Create a new ctable from `cols` with optional `names`.  The
    columns are carray objects.

    Parameters
    ----------
    cols : tuple or list of carray/ndarray objects, or structured ndarray
        The list of column data to build the ctable object.
        This can also be a pure NumPy structured array.

    names : list of strings
        The list of names for the columns.  If not passed, the names
        will be chosen as 'f0' for the first column, 'f1' for the
        second and so on so forth (NumPy convention).

    kwargs : list of parameters or dictionary
        Allows to pass additional arguments supported by carray
        constructors in case new carrays need to be built.

    Notes
    -----
    Columns passed as carrays are not be copied, so their settings
    will stay the same, even if you pass additional arguments (cparms,
    chunklen...).

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
        return (self.len,)

    @property
    def nbytes(self):
        "The original (uncompressed) size of this carray (in bytes)."
        return self.get_stats()[0]

    @property
    def cbytes(self):
        "The compressed size of this carray (in bytes)."
        return self.get_stats()[1]


    def get_stats(self):
        """
        get_stats()

        Get some stats (nbytes, cbytes and ratio) about this carray.

        Returns
        -------
        out : a (nbytes, cbytes, ratio) tuple
            nbytes is the number of uncompressed bytes in ctable.
            cbytes is the number of compressed bytes.  ratio is the
            compression ratio.

        """
        nbytes, cbytes, ratio = 0, 0, 0.0
        names, cols = self.names, self.cols
        for name in names:
            column = cols[name]
            nbytes += column.nbytes
            cbytes += column.cbytes
        cratio = nbytes / float(cbytes)
        return (nbytes, cbytes, cratio)


    def __init__(self, cols, names=None, **kwargs):

        self.names = []
        """The names of the columns (list)."""
        self.cols = {}
        """The carray columns (dict)."""
        self.len = 0
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
        self.len += clen

        # Cache a structured array of len 1 for ctable[int] acceleration
        self._arr1 = np.empty(shape=(1,), dtype=self.dtype)


    def append(self, rows):
        """
        append(rows)

        Append `rows` to this ctable.

        Parameters
        ----------
        rows : list/tuple of scalar values, NumPy arrays or carrays
            It also can be a NumPy record, a NumPy recarray, or
            another ctable.

        Returns
        -------
        out : int
            The number of elements appended.

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
        self.len += clen
        return clen


    def addcol(self, newcol, name=None, pos=None, **kwargs):
        """
        addcol(newcol, name=None, pos=None)

        Add a new `newcol` carray or ndarray as column.

        Parameters
        ----------
        newcol : carray or ndarray
            If a carray is passed, no conversion will be carried out.
            If conversion to a carray has to be done, `kwargs` will
            apply.

        name : string, optional
            The name for the new column.  If not passed, it will
            receive an automatic name.

        pos : int, optional
            The column position.  If not passed, it will be appended
            at the end.

        kwargs : list of parameters or dictionary
            Any parameter supported by the carray constructor.

        Notes
        -----
        You should not specificy both `name` and `pos` arguments,
        unless they are compatible.

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
        if len(newcol) != self.len:
            raise ValueError, "`newcol` must have the same length than ctable"

        if isinstance(newcol, np.ndarray):
            newcol = ca.carray(newcol, **kwargs)

        # Insert the column
        self.names.insert(pos, name)
        self.cols[name] = newcol


    def delcol(self, name=None, pos=None):
        """
        delcol(name=None, pos=None)

        Remove the column named `name` or in position `pos`.

        Parameters
        ----------
        name: string, optional
            The name of the column to remove.

        pos: int, optional
            The position of the column to remove.

        Notes
        -----
        You must specify at least a `name` or a `pos`.  You should not
        specificy both `name` and `pos` arguments, unless they are
        compatible.

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
        """
        copy(**kwargs)

        Return a copy of this ctable.

        Parameters
        ----------
        kwargs : list of parameters or dictionary
            Any parameter supported by the carray/ctable constructor.

        Returns
        -------
        out : ctable object
            The copy of this ctable.

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
        return self.len


    def __sizeof__(self):
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
        """
        x.__getitem__(y) <==> x[y]

        Returns values based on `key`.  Many different actions are to
        be carried out depending on the value of `key`.

        Parameters
        ----------
        key : int
            The corresponding ctable row is returned as a NumPy record.

        key : slice
            The range of rows determined by it is returned as a NumPy
            structured array.

        key : string
            The corresponding ctable column name will be returned.  If
            not a colname, it will be interpreted as a string
            ``expression`` returning boolean values, and the rows
            where these values are true will be returned as a NumPy
            structured array.

        key : list of strings
            The specified column names will be returned as a new
            ctable object.

        key : boolean carray/ndarray
            Return a ctable object filled with rows where `key` is true.

        key : list of ints, array of ints
            Return a ctable object filled with rows in this list/array.

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
                arr = self.eval(key, depth=3)
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
        (start, stop, step) = slice(start, stop, step).indices(self.len)
        # Build a numpy container
        n = utils.get_len_of_range(start, stop, step)
        ra = np.empty(shape=(n,), dtype=self.dtype)
        # Fill it
        for name in self.names:
            ra[name][:] = self.cols[name][start:stop:step]

        return ra


    def __setitem__(self, key, value):
        """
        x.__setitem__(i, y) <==> x[i]=y

        Sets `value` in `key` positions.  Many different actions are
        to be carried out depending on the value of `key`.

        Parameters
        ----------
        key : int
            The corresponding ctable row is returned as a NumPy record.

        key : slice
            The range of rows determined by it is returned as a NumPy
            structured array.

        key : boolean carray/ndarray
            Set ctable rows with `value` where `key` is true.

        key : list of ints, array of ints
            Set ctable rows with `value` in rows in this list/array.

        """

        # First, convert value into a structured array
        value = utils.to_ndarray(value, self.dtype)
        # Then, modify the rows
        for name in self.names:
            self.cols[name][key] = value[name]
        return


    def _getvars(self, expression, depth):
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
        """
        eval(expression, **kwargs)

        Evaluate the `expression` on columns and return the result.

        Parameters
        ----------
        expression : string
            Must be a string containing an expression supported by
            Numexpr.  It may contain columns or other carrays or NumPy
            arrays that can be found in the user space name.

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
        depth = kwargs.pop('depth', 2)
        vars, colnames = self._getvars(expression, depth=depth)

        # Compute the optimal block size (in elements)
        typesize = 0
        for name in vars.iterkeys():
            var = vars[name]
            if name in colnames:
                typesize += self.cols[name].dtype.itemsize
            elif hasattr(var, "dtype"):  # numpy arrays
                typesize += var.dtype.itemsize
            elif hasattr(var, "__len__"): # collections
                arr = np.array(var[0])
                typesize += arr.dtype.itemsize
        bsize = EVAL_BLOCK_SIZE // typesize
        # Evaluation seems more efficient if block size is a power of 2
        bsize = 2 ** (int(math.log(bsize, 2)))

        # Perform the evaluation in blocks
        vars_ = {}
        for i in xrange(0, self.len, bsize):
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
                # Get a decent default for expectedlen
                nrows = kwargs.pop('expectedlen', self.len)
                result = ca.carray(res_block, expectedlen=nrows, **kwargs)
            else:
                result.append(res_block)

        return result


    def __str__(self):
        if self.len > 100:
            return "[%s, %s, %s, ..., %s, %s, %s]\n" % \
                   (self[0], self[1], self[2], self[-3], self[-2], self[-1])
        else:
            return str(self[:])


    def __repr__(self):
        nbytes, cbytes, cratio = self.get_stats()
        snbytes = utils.human_readable_size(nbytes)
        scbytes = utils.human_readable_size(cbytes)
        fullrepr = """ctable(%s, %s)
  nbytes: %s; cbytes: %s; ratio: %.2f
%s""" % (self.shape, self.dtype, snbytes, scbytes, cratio, str(self))
        return fullrepr
