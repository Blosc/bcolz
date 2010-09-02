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

import numpy as np
import carray as ca


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


    def __init__(self, cols, names=None):
        """Create a new ctable from `cols` with optional `names`.

        `cols` can be a tuple/list of carrays or NumPy arrays.  It can
        also be a NumPy structured array.

        If `names` is passed, this will be taken as the list of names
        for the columns.
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
                column = ca.carray(column)
            elif ratype:
                column = ca.carray(cols[name])
            self.cols[name] = column
            if clen >= 0 and clen != len(column):
                raise ValueError, "all `cols` must have the same length"
            clen = len(column)
        self.nrows += clen


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


    def _get_len_of_range(self, start, stop, step):
        """Get the length of a (start, stop, step) range."""
        n = 0
        if start < stop:
            n = ((stop - start - 1) // step + 1);
        return n


    def __len__(self):
        """Return the length of self."""
        return self.nrows


    def __sizeof__(self):
        """Return the number of bytes taken by self."""
        return self.cbytes


    def __getitem__(self, key):
        """Get a row or a range of rows.  Also a column or range of columns.

        If `key` argument is an integer, the corresponding ctable row
        is returned as a NumPy record.  If `key` is a slice, the range
        of rows determined by it is returned as a NumPy structured
        array.

        If `key` is a string, the corresponding ctable column name
        will be returned.  If `key` is a list of names, the specified
        columns will be returned as a new ctable object.
        """

        # First check for a column name or range of names
        if type(key) is str:
            if key not in self.names:
                raise KeyError, "key is not a column name."
            return self.cols[key]
        elif type(key) is list:
            strlist = [type(v) for v in key] == [str for v in key]
            if strlist:
                cols = [self.cols[name] for name in key]
                return ctable(cols, key)
            else:
                raise KeyError, "key is not a list of names"

        # First check for an int or range of ints
        # Get rid of multidimensional keys
        if type(key) == tuple:
            if len(key) != 1:
                raise KeyError, "multidimensional keys are not supported"
            key = key[0]

        if type(key) == int:
            if key >= self.nrows:
                raise IndexError, "index out of range"
            if key < 0:
                # To support negative values
                key += self.nrows
            (start, stop, step) = key, key+1, 1
            scalar = True
        elif type(key) == slice:
            (start, stop, step) = key.start, key.stop, key.step
        else:
            raise NotImplementedError, "key not supported: %s" % repr(key)

        if step and step <= 0 :
            raise NotImplementedError("step in slice can only be positive")

        # Get the corrected values for start, stop, step
        (start, stop, step) = slice(start, stop, step).indices(self.nrows)
        # Build a numpy container
        n = self._get_len_of_range(start, stop, step)
        ra = np.empty(shape=(n,), dtype=self.dtype)
        # Fill it
        for name in self.names:
            ra[name][:] = self.cols[name][start:stop:step]

        return ra


    def __setitem__(self, key, value):
        """Set a row or a range of rows."""
        raise NotImplementedError


    def __str__(self):
        """Represent the ctable as an string."""
        if self.nrows > 100:
            return "[%s, %s, %s... %s, %s, %s]\n" % \
                   (self[0], self[1], self[2], self[-3], self[-2], self[-1])
        else:
            return str(self[:])


    def __repr__(self):
        """Represent the carray as an string, with additional info."""
        nbytes, cbytes, cratio = self.get_stats()
        fullrepr = "ctable(%s, %s)  nbytes: %d; cbytes: %d; ratio: %.2f\n%s" % \
                   (self.shape, self.dtype, nbytes, cbytes, cratio, str(self))
        return fullrepr
