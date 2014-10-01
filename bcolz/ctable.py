########################################################################
#
#       License: BSD
#       Created: September 01, 2010
#       Author:  Francesc Alted - francesc@blosc.io
#
########################################################################

from __future__ import absolute_import

import numpy as np
import bcolz
from bcolz import utils, attrs, array2string
import itertools
from collections import namedtuple
import json
import os
import os.path
import shutil
from .py2help import _inttypes, imap, xrange
from bcolz.carray_ext import carray, groupby_cython, carray_is_in

_inttypes += (np.integer,)
islice = itertools.islice

ROOTDIRS = '__rootdirs__'


class cols(object):
    """Class for accessing the columns on the ctable object."""

    def __init__(self, rootdir, mode):
        self.rootdir = rootdir
        self.mode = mode
        self.names = []
        self._cols = {}

    def read_meta_and_open(self):
        """Read the meta-information and initialize structures."""
        # Get the directories of the columns
        rootsfile = os.path.join(self.rootdir, ROOTDIRS)
        with open(rootsfile, 'rb') as rfile:
            data = json.loads(rfile.read().decode('ascii'))
        # JSON returns unicode, but we want plain bytes for Python 2.x
        self.names = [str(name) for name in data['names']]
        # Initialize the cols by instantiating the carrays
        for name in self.names:
            dir_ = os.path.join(self.rootdir, name)
            self._cols[name] = bcolz.carray(rootdir=dir_, mode=self.mode)

    def update_meta(self):
        """Update metainfo about directories on-disk."""
        if not self.rootdir:
            return
        data = {'names': self.names}
        rootsfile = os.path.join(self.rootdir, ROOTDIRS)
        with open(rootsfile, 'wb') as rfile:
            rfile.write(json.dumps(data).encode('ascii'))
            rfile.write(b"\n")

    def __getitem__(self, name):
        return self._cols[name]

    def __setitem__(self, name, carray):
        self.names.append(name)
        self._cols[name] = carray
        self.update_meta()

    def __iter__(self):
        return iter(self._cols)

    def __len__(self):
        return len(self.names)

    def insert(self, name, pos, carray):
        """Insert carray in the specified pos and name."""
        self.names.insert(pos, name)
        self._cols[name] = carray
        self.update_meta()

    def pop(self, name):
        """Return the named column and remove it."""
        pos = self.names.index(name)
        name = self.names.pop(pos)
        col = self._cols[name]
        self.update_meta()
        return col

    def __str__(self):
        fullrepr = ""
        for name in self.names:
            fullrepr += "%s : %s" % (name, str(self._cols[name]))
        return fullrepr

    def __repr__(self):
        fullrepr = ""
        for name in self.names:
            fullrepr += "%s : %s\n" % (name, repr(self._cols[name]))
        return fullrepr


class ctable(object):
    """
    ctable(cols, names=None, **kwargs)

    This class represents a compressed, column-wise, in-memory table.

    Create a new ctable from `cols` with optional `names`.

    Parameters
    ----------
    columns : tuple or list of column objects
        The list of column data to build the ctable object.  This can also be
        a pure NumPy structured array.  A list of lists or tuples is valid
        too, as long as they can be converted into carray objects.
    names : list of strings or string
        The list of names for the columns.  The names in this list must be
        valid Python identifiers, must not start with an underscore, and has
        to be specified in the same order as the `cols`.  If not passed, the
        names will be chosen as 'f0' for the first column, 'f1' for the second
        and so on so forth (NumPy convention).
    kwargs : list of parameters or dictionary
        Allows to pass additional arguments supported by carray
        constructors in case new carrays need to be built.

    Notes
    -----
    Columns passed as carrays are not be copied, so their settings
    will stay the same, even if you pass additional arguments (cparams,
    chunklen...).

    """

    # Properties
    # ``````````

    @property
    def cbytes(self):
        "The compressed size of this object (in bytes)."
        return self._get_stats()[1]

    @property
    def cparams(self):
        "The compression parameters for this object."
        return self._cparams

    @property
    def dtype(self):
        "The data type of this object (numpy dtype)."
        names, cols = self.names, self.cols
        l = [(name, cols[name].dtype) for name in names]
        return np.dtype(l)

    @property
    def names(self):
        "The names of the object (list)."
        return self.cols.names

    @property
    def ndim(self):
        "The number of dimensions of this object."
        return len(self.shape)

    @property
    def nbytes(self):
        "The original (uncompressed) size of this object (in bytes)."
        return self._get_stats()[0]

    @property
    def shape(self):
        "The shape of this object."
        return (self.len,)

    @property
    def size(self):
        "The size of this object."
        return np.prod(self.shape)

    def __init__(self, columns=None, names=None, **kwargs):

        # Important optional params
        self._cparams = kwargs.get('cparams', bcolz.cparams())
        self.rootdir = kwargs.get('rootdir', None)
        "The directory where this object is saved."
        if self.rootdir is None and columns is None:
            raise ValueError(
                "You should pass either a `columns` or a `rootdir` param"
                " at very least")
        # The mode in which the object is created/opened
        if self.rootdir is not None and os.path.exists(self.rootdir):
            self.mode = kwargs.setdefault('mode', 'a')
            if columns is not None and self.mode == 'a':
                raise ValueError(
                    "You cannot pass a `columns` param in 'a'ppend mode")
        else:
            self.mode = kwargs.setdefault('mode', 'w')

        # Setup the columns accessor
        self.cols = cols(self.rootdir, self.mode)
        "The ctable columns accessor."

        # The length counter of this array
        self.len = 0

        # Create a new ctable or open it from disk
        _new = False
        if self.mode in ('r', 'a'):
            self.open_ctable()
        elif columns is not None:
            self.create_ctable(columns, names, **kwargs)
            _new = True
        else:
            raise ValueError(
                "You cannot open a ctable in 'w'rite mode"
                " without a `columns` param")

        # Attach the attrs to this object
        self.attrs = attrs.attrs(self.rootdir, self.mode, _new=_new)

        # Cache a structured array of len 1 for ctable[int] acceleration
        self._arr1 = np.empty(shape=(1,), dtype=self.dtype)

    def create_ctable(self, columns, names, **kwargs):
        """Create a ctable anew."""

        # Create the rootdir if necessary
        if self.rootdir:
            self.mkdir_rootdir(self.rootdir, self.mode)

        # Get the names of the columns
        if names is None:
            if isinstance(columns, np.ndarray):  # ratype case
                names = list(columns.dtype.names)
            else:
                names = ["f%d" % i for i in range(len(columns))]
        else:
            if type(names) == tuple:
                names = list(names)
            if type(names) != list:
                raise ValueError(
                    "`names` can only be a list or tuple")
            if len(names) != len(columns):
                raise ValueError(
                    "`columns` and `names` must have the same length")
        # Check names validity
        nt = namedtuple('_nt', list(names), verbose=False)
        names = list(nt._fields)

        # Guess the kind of columns input
        calist, nalist, ratype = False, False, False
        if type(columns) in (tuple, list):
            calist = [type(v) for v in columns] == \
                     [bcolz.carray for v in columns]
            nalist = [type(v) for v in columns] == \
                     [np.ndarray for v in columns]
        elif isinstance(columns, np.ndarray):
            ratype = hasattr(columns.dtype, "names")
            if ratype:
                if len(columns.shape) != 1:
                    raise ValueError("only unidimensional shapes supported")
        else:
            raise ValueError("`columns` input is not supported")

        # Populate the columns
        clen = -1
        for i, name in enumerate(names):
            if self.rootdir:
                # Put every carray under each own `name` subdirectory
                kwargs['rootdir'] = os.path.join(self.rootdir, name)
            if calist:
                column = columns[i]
                if self.rootdir:
                    # Store this in destination
                    column = column.copy(**kwargs)
            elif nalist:
                column = columns[i]
                if column.dtype == np.void:
                    raise ValueError(
                        "`columns` elements cannot be of type void")
                column = bcolz.carray(column, **kwargs)
            elif ratype:
                column = bcolz.carray(columns[name], **kwargs)
            else:
                # Try to convert from a sequence of columns
                column = bcolz.carray(columns[i], **kwargs)
            self.cols[name] = column
            if clen >= 0 and clen != len(column):
                if self.rootdir:
                    shutil.rmtree(self.rootdir)
                raise ValueError("all `columns` must have the same length")
            clen = len(column)

        self.len = clen

    def open_ctable(self):
        """Open an existing ctable on-disk."""

        # Open the ctable by reading the metadata
        self.cols.read_meta_and_open()

        # Get the length out of the first column
        self.len = len(self.cols[self.names[0]])

    def mkdir_rootdir(self, rootdir, mode):
        """Create the `self.rootdir` directory safely."""
        if os.path.exists(rootdir):
            if mode != "w":
                raise IOError(
                    "specified rootdir path '%s' already exists "
                    "and creation mode is '%s'" % (rootdir, mode))
            if os.path.isdir(rootdir):
                shutil.rmtree(rootdir)
            else:
                os.remove(rootdir)
        os.mkdir(rootdir)

    def append(self, cols):
        """
        append(cols)

        Append `cols` to this ctable.

        Parameters
        ----------
        cols : list/tuple of scalar values, NumPy arrays or carrays
            It also can be a NumPy record, a NumPy recarray, or
            another ctable.

        """

        # Guess the kind of cols input
        calist, nalist, sclist, ratype = False, False, False, False
        if type(cols) in (tuple, list):
            calist = [type(v) for v in cols] == [bcolz.carray for v in cols]
            nalist = [type(v) for v in cols] == [np.ndarray for v in cols]
            if not (calist or nalist):
                # Try with a scalar list
                sclist = True
        elif isinstance(cols, np.ndarray):
            ratype = hasattr(cols.dtype, "names")
        elif isinstance(cols, bcolz.ctable):
            # Convert int a list of carrays
            cols = [cols[name] for name in self.names]
            calist = True
        else:
            raise ValueError("`cols` input is not supported")
        if not (calist or nalist or sclist or ratype):
            raise ValueError("`cols` input is not supported")

        # Populate the columns
        clen = -1
        for i, name in enumerate(self.names):
            if calist or sclist:
                column = cols[i]
            elif nalist:
                column = cols[i]
                if column.dtype == np.void:
                    raise ValueError("`cols` elements cannot be of type void")
                column = column
            elif ratype:
                column = cols[name]
            # Append the values to column
            self.cols[name].append(column)
            if sclist and not hasattr(column, '__len__'):
                clen2 = 1
            else:
                clen2 = len(column)
            if clen >= 0 and clen != clen2:
                raise ValueError(
                    "all cols in `cols` must have the same length")
            clen = clen2
        self.len += clen

    def trim(self, nitems):
        """
        trim(nitems)

        Remove the trailing `nitems` from this instance.

        Parameters
        ----------
        nitems : int
            The number of trailing items to be trimmed.

        """

        for name in self.names:
            self.cols[name].trim(nitems)
        self.len -= nitems

    def resize(self, nitems):
        """
        resize(nitems)

        Resize the instance to have `nitems`.

        Parameters
        ----------
        nitems : int
            The final length of the instance.  If `nitems` is larger than the
            actual length, new items will appended using `self.dflt` as
            filling values.

        """

        for name in self.names:
            self.cols[name].resize(nitems)
        self.len = nitems

    def addcol(self, newcol, name=None, pos=None, **kwargs):
        """
        addcol(newcol, name=None, pos=None, **kwargs)

        Add a new `newcol` object as column.

        Parameters
        ----------
        newcol : carray, ndarray, list or tuple
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

        See Also
        --------
        delcol

        """

        # Check params
        if pos is None:
            pos = len(self.names)
        else:
            if pos and type(pos) != int:
                raise ValueError("`pos` must be an int")
            if pos < 0 or pos > len(self.names):
                raise ValueError("`pos` must be >= 0 and <= len(self.cols)")
        if name is None:
            name = "f%d" % pos
        else:
            if type(name) != str:
                raise ValueError("`name` must be a string")
        if name in self.names:
            raise ValueError("'%s' column already exists" % name)
        if len(newcol) != self.len:
            raise ValueError("`newcol` must have the same length than ctable")

        if isinstance(newcol, np.ndarray):
            if 'cparams' not in kwargs:
                kwargs['cparams'] = self.cparams
            newcol = bcolz.carray(newcol, **kwargs)
        elif type(newcol) in (list, tuple):
            if 'cparams' not in kwargs:
                kwargs['cparams'] = self.cparams
            newcol = bcolz.carray(newcol, **kwargs)
        elif type(newcol) != bcolz.carray:
            raise ValueError(
                """`newcol` type not supported""")

        # Insert the column
        self.cols.insert(name, pos, newcol)
        # Update _arr1
        self._arr1 = np.empty(shape=(1,), dtype=self.dtype)

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
        specify both `name` and `pos` arguments, unless they are
        compatible.

        See Also
        --------
        addcol

        """

        if name is None and pos is None:
            raise ValueError("specify either a `name` or a `pos`")
        if name is not None and pos is not None:
            raise ValueError("you cannot specify both a `name` and a `pos`")
        if name:
            if type(name) != str:
                raise ValueError("`name` must be a string")
            if name not in self.names:
                raise ValueError("`name` not found in columns")
            pos = self.names.index(name)
        elif pos is not None:
            if type(pos) != int:
                raise ValueError("`pos` must be an int")
            if pos < 0 or pos > len(self.names):
                raise ValueError("`pos` must be >= 0 and <= len(self.cols)")
            name = self.names[pos]

        # Remove the column
        self.cols.pop(name)
        # Update _arr1
        self._arr1 = np.empty(shape=(1,), dtype=self.dtype)

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

        # Check that origin and destination do not overlap
        rootdir = kwargs.get('rootdir', None)
        if rootdir and self.rootdir and rootdir == self.rootdir:
                raise IOError("rootdir cannot be the same during copies")

        # Remove possible unsupported args for columns
        names = kwargs.pop('names', self.names)

        # Copy the columns
        if rootdir:
            # A copy is always made during creation with a rootdir
            cols = [self.cols[name] for name in self.names]
        else:
            cols = [self.cols[name].copy(**kwargs) for name in self.names]
        # Create the ctable
        ccopy = ctable(cols, names, **kwargs)
        return ccopy

    @staticmethod
    def fromdataframe(df, **kwargs):
        """
        fromdataframe(df, **kwargs)

        Return a ctable object out of a pandas dataframe.

        Parameters
        ----------
        df : DataFrame
            A pandas dataframe.
        kwargs : list of parameters or dictionary
            Any parameter supported by the ctable constructor.

        Returns
        -------
        out : ctable object
            A ctable filled with values from `df`.

        Note
        ----
        The 'object' dtype will be converted into a 'S'tring type, if possible.
        This allows for much better storage savings in bcolz.

        See Also
        --------
        ctable.todataframe

        """
        if bcolz.pandas_here:
            import pandas as pd
        else:
            raise ValueError("you need pandas to use this functionality")

        # Use the names in kwargs, or if not there, the names in dataframe
        if 'names' in kwargs:
            names = kwargs.pop('names')
        else:
            names = list(df.columns.values)

        # Build the list of columns as in-memory numpy arrays and carrays
        # (when doing the conversion object -> string)
        cols = []
        # Remove a possible rootdir argument to prevent copies going to disk
        ckwargs = kwargs.copy()
        if 'rootdir' in ckwargs:
            del ckwargs['rootdir']
        for key in names:
            vals = df[key].values  # just a view as a numpy array
            if vals.dtype == np.object:
                inferred_type = pd.lib.infer_dtype(vals)
                # Next code could be made to work if
                # pd.lib.max_len_string_array(vals) below would work
                # with unicode in Python 2
                # if inferred_type == 'unicode':
                #     maxitemsize = pd.lib.max_len_string_array(vals)
                #     print "maxitemsize:", maxitesize
                #     # Convert the view into a carray of Unicode strings
                #     col = bcolz.carray(vals,
                #                        dtype='U%d' % maxitemsize, **ckwargs)
                # elif inferred_type == 'string':
                if inferred_type == 'string':
                    maxitemsize = pd.lib.max_len_string_array(vals)
                    # Convert the view into a carray of regular strings
                    col = bcolz.carray(vals, dtype='S%d' %
                                       maxitemsize, **ckwargs)
                else:
                    col = vals
                cols.append(col)
            else:
                cols.append(vals)

        # Create the ctable
        ct = ctable(cols, names, **kwargs)
        return ct

    @staticmethod
    def fromhdf5(filepath, nodepath='/ctable', **kwargs):
        """
        fromhdf5(filepath, nodepath='/ctable', **kwargs)

        Return a ctable object out of a compound HDF5 dataset (PyTables Table).

        Parameters
        ----------
        filepath : string
            The path of the HDF5 file.
        nodepath : string
            The path of the node inside the HDF5 file.
        kwargs : list of parameters or dictionary
            Any parameter supported by the ctable constructor.

        Returns
        -------
        out : ctable object
            A ctable filled with values from the HDF5 node.

        See Also
        --------
        ctable.tohdf5

        """
        if bcolz.tables_here:
            import tables as tb
        else:
            raise ValueError("you need PyTables to use this functionality")

        # Read the Table on file
        f = tb.open_file(filepath)
        try:
            t = f.get_node(nodepath)
        except:
            f.close()
            raise

        # Use the names in kwargs, or if not there, the names in Table
        if 'names' in kwargs:
            names = kwargs.pop('names')
        else:
            names = t.colnames
        # Collect metadata
        dtypes = [t.dtype.fields[name][0] for name in names]
        cols = [np.zeros(0, dtype=dt) for dt in dtypes]
        # Create an empty ctable
        ct = ctable(cols, names, **kwargs)
        # Fill it chunk by chunk
        bs = t._v_chunkshape[0]
        for i in xrange(0, len(t), bs):
            ct.append(t[i:i+bs])
        # Get the attributes
        for key in t.attrs._f_list():
            ct.attrs[key] = t.attrs[key]
        f.close()
        return ct

    def todataframe(self, columns=None, orient='columns'):
        """
        todataframe(columns=None, orient='columns')

        Return a pandas dataframe out of this object.

        Parameters
        ----------
        columns : sequence of column labels, optional
            Must be passed if orient='index'.
        orient : {'columns', 'index'}, default 'columns'
            The "orientation" of the data. If the keys of the input correspond
            to column labels, pass 'columns' (default). Otherwise if the keys
            correspond to the index, pass 'index'.

        Returns
        -------
        out : DataFrame
            A pandas DataFrame filled with values from this object.

        See Also
        --------
        ctable.fromdataframe

        """
        if bcolz.pandas_here:
            import pandas as pd
        else:
            raise ValueError("you need pandas to use this functionality")

        # Use a generator here to minimize the number of column copies
        # existing simultaneously in-memory
        df = pd.DataFrame.from_items(
            ((key, self[key][:]) for key in self.names),
            columns=columns, orient=orient)
        return df

    def tohdf5(self, filepath, nodepath='/ctable', mode='w',
               cparams=None, cname=None):
        """
        tohdf5(filepath, nodepath='/ctable', mode='w',
               cparams=None, cname=None)

        Write this object into an HDF5 file.

        Parameters
        ----------
        filepath : string
            The path of the HDF5 file.
        nodepath : string
            The path of the node inside the HDF5 file.
        mode : string
            The mode to open the PyTables file.  Default is 'w'rite mode.
        cparams : cparams object
            The compression parameters.  The defaults are the same than for
            the current bcolz environment.
        cname : string
            Any of the compressors supported by PyTables (e.g. 'zlib').  The
            default is to use 'blosc' as meta-compressor in combination with
            one of its compressors (see `cparams` parameter above).

        See Also
        --------
        ctable.fromhdf5

        """
        if bcolz.tables_here:
            import tables as tb
        else:
            raise ValueError("you need PyTables to use this functionality")

        if os.path.exists(filepath):
            raise IOError("path '%s' already exists" % filepath)

        f = tb.open_file(filepath, mode=mode)
        cparams = cparams if cparams is not None else bcolz.defaults.cparams
        cname = cname if cname is not None else "blosc:"+cparams['cname']
        filters = tb.Filters(complevel=cparams['clevel'],
                             shuffle=cparams['clevel'],
                             complib=cname)
        t = f.create_table(f.root, nodepath[1:], self.dtype, filters=filters)
        # Set the attributes
        for key, val in self.attrs:
            t.attrs[key] = val
        # Copy the data
        for block in bcolz.iterblocks(self):
            t.append(block)
        f.close()

    def __len__(self):
        return self.len

    def __sizeof__(self):
        return self.cbytes

    def where(self, expression, outcols=None, limit=None, skip=0):
        """
        where(expression, outcols=None, limit=None, skip=0)

        Iterate over rows where `expression` is true.

        Parameters
        ----------
        expression : string or carray
            A boolean Numexpr expression or a boolean carray.
        outcols : list of strings or string
            The list of column names that you want to get back in results.
            Alternatively, it can be specified as a string such as 'f0 f1' or
            'f0, f1'.  If None, all the columns are returned.  If the special
            name 'nrow__' is present, the number of row will be included in
            output.
        limit : int
            A maximum number of elements to return.  The default is return
            everything.
        skip : int
            An initial number of elements to skip.  The default is 0.

        Returns
        -------
        out : iterable
            This iterable returns rows as NumPy structured types (i.e. they
            support being mapped either by position or by name).

        See Also
        --------
        iter

        """

        # Check input
        if type(expression) is str:
            # That must be an expression
            boolarr = self.eval(expression)
        elif hasattr(expression, "dtype") and expression.dtype.kind == 'b':
            boolarr = expression
        else:
            raise ValueError(
                "only boolean expressions or arrays are supported")

        # Check outcols
        if outcols is None:
            outcols = self.names
        else:
            if type(outcols) not in (list, tuple, str):
                raise ValueError("only list/str is supported for outcols")
            # Check name validity
            nt = namedtuple('_nt', outcols, verbose=False)
            outcols = list(nt._fields)
            if set(outcols) - set(self.names+['nrow__']) != set():
                raise ValueError("not all outcols are real column names")

        # Get iterators for selected columns
        icols, dtypes = [], []
        for name in outcols:
            if name == "nrow__":
                icols.append(boolarr.wheretrue(limit=limit, skip=skip))
                dtypes.append((name, np.int_))
            else:
                col = self.cols[name]
                icols.append(col.where(boolarr, limit=limit, skip=skip))
                dtypes.append((name, col.dtype))
        dtype = np.dtype(dtypes)
        return self._iter(icols, dtype)

    def whereblocks(self, expression, blen=None, outfields=None, limit=None,
                    skip=0):
        """whereblocks(expression, blen=None, outfields=None, limit=None, skip=0)

        Iterate over the rows that fullfill the `expression` condition on
        this ctable, in blocks of size `blen`.

        Parameters
        ----------
        expression : string or carray
            A boolean Numexpr expression or a boolean carray.
        blen : int
            The length of the block that is returned.  The default is the
            chunklen, or for a ctable, the minimum of the different column
            chunklens.
        outfields : list of strings or string
            The list of column names that you want to get back in results.
            Alternatively, it can be specified as a string such as 'f0 f1' or
            'f0, f1'.
        limit : int
            A maximum number of elements to return.  The default is return
            everything.
        skip : int
            An initial number of elements to skip.  The default is 0.

        Returns
        -------
        out : iterable
            This iterable returns buffers as NumPy arrays made of
            structured types (or homogeneous ones in case `outfields` is a
            single field.

        See Also
        --------
        iterblocks

        """

        if blen is None:
            # Get the minimum chunklen for every field
            blen = min(self[col].chunklen for col in self.cols)
        if outfields is None:
            dtype = self.dtype
        else:
            if not isinstance(outfields, (list, tuple)):
                raise ValueError("only a sequence is supported for outfields")
            # Get the dtype for the outfields set
            try:
                dtype = [(name, self[name].dtype) for name in outfields]
            except IndexError:
                raise ValueError(
                    "Some names in `outfields` are not real fields")

        buf = np.empty(blen, dtype=dtype)
        nrow = 0
        for row in self.where(expression, outfields, limit, skip):
            buf[nrow] = row
            nrow += 1
            if nrow == blen:
                yield buf
                buf = np.empty(blen, dtype=dtype)
                nrow = 0
        yield buf[:nrow]

    def where_terms(self, term_list, outcols=None, limit=None, skip=0):
        """
        where_terms(term_list, outcols=None, limit=None, skip=0)

        Iterate over rows where `term_list` is true.
        A terms list has a [(col, operator, value), ..] construction.
        Eg. [('sales', '>', 2), ('state', 'in', ['IL', 'AR'])]

        :param term_list:
        :param outcols:
        :param limit:
        :param skip:
        :return: :raise ValueError:
        """

        if type(term_list) not in [list, set, tuple]:
            raise ValueError("Only term lists are supported")

        eval_string = ''
        eval_list = []

        for term in term_list:
            filter_col = term[0]
            filter_operator = term[1].lower()
            filter_value = term[2]

            if filter_operator not in ['in', 'not in']:
                # direct filters should be added to the eval_string

                # add and logic if not the first term
                if eval_string:
                    eval_string += ' & '

                eval_string += filter_col + ' ' \
                               + filter_operator + ' ' \
                               + str(filter_value)

            elif filter_operator in ['in', 'not in']:
                # Check input
                if type(filter_value) not in [list, set, tuple]:
                    raise ValueError("In selections need lists, sets or tuples")

                if len(filter_value) < 1:
                    raise ValueError("A value list needs to have values")

                elif len(filter_value) == 1:
                    # handle as eval
                    # add and logic if not the first term
                    if eval_string:
                        eval_string += ' & '

                    if filter_operator == 'not in':
                        filter_operator = '!='
                    else:
                        filter_operator = '=='

                    eval_string += filter_col + ' ' + \
                                   filter_operator + ' '

                    filter_value = filter_value[0]

                    if type(filter_value) == str:
                        filter_value = '"' + filter_value + '"'
                    else:
                        filter_value = str(filter_value)

                    eval_string += filter_value

                else:

                    if type(filter_value) in [list, tuple]:
                        filter_value = set(filter_value)

                    eval_list.append(
                        (filter_col, filter_operator, filter_value)
                    )
            else:
                raise ValueError(
                    "Input not correctly formatted for eval or list filtering"
                )

        # (1) Evaluate terms in eval
        # return eval_string, eval_list
        if eval_string:
            boolarr = self.eval(eval_string)
            if eval_list:
                # convert to numpy array for array_is_in
                boolarr = boolarr[:]
        else:
            boolarr = np.ones(self.size, dtype=bool)

        # (2) Evaluate other terms like 'in' or 'not in' ...
        for term in eval_list:

            name = term[0]
            col = self.cols[name]

            operator = term[1]
            if operator.lower() == 'not in':
                reverse = True
            elif operator.lower() == 'in':
                reverse = False
            else:
                raise ValueError(
                    "Input not correctly formatted for list filtering"
                )

            value_set = set(term[2])

            carray_is_in(col, value_set, boolarr, reverse)

        if eval_list:
            # convert boolarr back to carray
            boolarr = carray(boolarr)

        if outcols is None:
            outcols = self.names

        # Check outcols
        if outcols is None:
            outcols = self.names
        else:
            if type(outcols) not in (list, tuple, str):
                raise ValueError("only list/str is supported for outcols")
            # Check name validity
            nt = namedtuple('_nt', outcols, verbose=False)
            outcols = list(nt._fields)
            if set(outcols) - set(self.names + ['nrow__']) != set():
                raise ValueError("not all outcols are real column names")

        # Get iterators for selected columns
        icols, dtypes = [], []
        for name in outcols:
            if name == "nrow__":
                icols.append(boolarr.wheretrue(limit=limit, skip=skip))
                dtypes.append((name, np.int_))
            else:
                col = self.cols[name]
                icols.append(col.where(boolarr, limit=limit, skip=skip))
                dtypes.append((name, col.dtype))
        dtype = np.dtype(dtypes)

        return self._iter(icols, dtype)

    def groupby(self, groupby_cols, measure_cols, where=None, where_terms=None):
        """
        Groups the measure_cols over the groupby_cols. Currently only sums are supported.
        Also supports where and where_terms filtering

        :param groupby_cols: A list of groupby columns
        :param measure_cols: A list of measure columns (sum only atm)
        :param where: A where filter, if it should be applied pre-grouping; default: None
        :param where_terms: A where_terms filter, if it should be applied pre-grouping; default: None
        :return:
        """

        outcols = groupby_cols + measure_cols
        col_dtype_set = {col: self.dtype[col] for col in outcols}

        if where is not None:
            iter_gen = self.where(where, outcols=outcols)
        elif where_terms is not None:
            iter_gen = self.where_terms(where_terms, outcols=outcols)
        else:
            iter_gen = self.iter(outcols=outcols)

        return groupby_cython(iter_gen, groupby_cols, measure_cols, col_dtype_set)

    def __iter__(self):
        return self.iter(0, self.len, 1)

    def iter(self, start=0, stop=None, step=1, outcols=None,
             limit=None, skip=0):
        """
        iter(start=0, stop=None, step=1, outcols=None, limit=None, skip=0)

        Iterator with `start`, `stop` and `step` bounds.

        Parameters
        ----------
        start : int
            The starting item.
        stop : int
            The item after which the iterator stops.
        step : int
            The number of items incremented during each iteration.  Cannot be
            negative.
        outcols : list of strings or string
            The list of column names that you want to get back in results.
            Alternatively, it can be specified as a string such as 'f0 f1' or
            'f0, f1'.  If None, all the columns are returned.  If the special
            name 'nrow__' is present, the number of row will be included in
            output.
        limit : int
            A maximum number of elements to return.  The default is return
            everything.
        skip : int
            An initial number of elements to skip.  The default is 0.

        Returns
        -------
        out : iterable

        See Also
        --------
        where

        """

        # Check outcols
        if outcols is None:
            outcols = self.names
        else:
            if type(outcols) not in (list, tuple, str):
                raise ValueError("only list/str is supported for outcols")
            # Check name validity
            nt = namedtuple('_nt', outcols, verbose=False)
            outcols = list(nt._fields)
            if set(outcols) - set(self.names+['nrow__']) != set():
                raise ValueError("not all outcols are real column names")

        # Check limits
        if step <= 0:
            raise NotImplementedError("step param can only be positive")
        start, stop, step = slice(start, stop, step).indices(self.len)

        # Get iterators for selected columns
        icols, dtypes = [], []
        for name in outcols:
            if name == "nrow__":
                istop = None
                if limit is not None:
                    istop = limit + skip
                icols.append(islice(xrange(start, stop, step), skip, istop))
                dtypes.append((name, np.int_))
            else:
                col = self.cols[name]
                icols.append(
                    col.iter(start, stop, step, limit=limit, skip=skip))
                dtypes.append((name, col.dtype))
        dtype = np.dtype(dtypes)
        return self._iter(icols, dtype)

    def _iter(self, icols, dtype):
        """Return a list of `icols` iterators with `dtype` names."""

        icols = tuple(icols)
        namedt = namedtuple('row', dtype.names)
        iterable = imap(namedt, *icols)
        return iterable

    def _where(self, boolarr, colnames=None):
        """Return rows where `boolarr` is true as an structured array.

        This is called internally only, so we can assum that `boolarr`
        is a boolean array.
        """

        if colnames is None:
            colnames = self.names
        cols = [self.cols[name][boolarr] for name in colnames]
        dtype = np.dtype([(name, self.cols[name].dtype) for name in colnames])
        result = np.rec.fromarrays(cols, dtype=dtype).view(np.ndarray)

        return result

    def __getitem__(self, key):
        """
        x.__getitem__(key) <==> x[key]

        Returns values based on `key`.  All the functionality of
        ``ndarray.__getitem__()`` is supported (including fancy
        indexing), plus a special support for expressions:

        Parameters
        ----------
        key : string
            The corresponding ctable column name will be returned.  If
            not a column name, it will be interpret as a boolean
            expression (computed via `ctable.eval`) and the rows where
            these values are true will be returned as a NumPy
            structured array.

        See Also
        --------
        ctable.eval

        """

        # First, check for integer
        if isinstance(key, _inttypes):
            # Get a copy of the len-1 array
            ra = self._arr1.copy()
            # Fill it
            ra[0] = tuple([self.cols[name][key] for name in self.names])
            return ra[0]
        # Slices
        elif type(key) == slice:
            (start, stop, step) = key.start, key.stop, key.step
            if step and step <= 0:
                raise NotImplementedError("step in slice can only be positive")
        # Multidimensional keys
        elif isinstance(key, tuple):
            if len(key) != 1:
                raise IndexError("multidimensional keys are not supported")
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
                raise IndexError(
                    "key cannot be converted to an array of indices")
            return np.fromiter((self[i] for i in key),
                               dtype=self.dtype, count=len(key))
        # A boolean array (case of fancy indexing)
        elif hasattr(key, "dtype"):
            if key.dtype.type == np.bool_:
                return self._where(key)
            elif np.issubsctype(key, np.int_):
                # An integer array
                return np.array([self[i] for i in key], dtype=self.dtype)
            else:
                raise IndexError(
                    "arrays used as indices must be integer (or boolean)")
        # Column name or expression
        elif type(key) is str:
            if key not in self.names:
                # key is not a column name, try to evaluate
                arr = self.eval(key, depth=4)
                if arr.dtype.type != np.bool_:
                    raise IndexError(
                        "`key` %s does not represent a boolean "
                        "expression" % key)
                return self._where(arr)
            return self.cols[key]
        # All the rest not implemented
        else:
            raise NotImplementedError("key not supported: %s" % repr(key))

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
        x.__setitem__(key, value) <==> x[key] = value

        Sets values based on `key`.  All the functionality of
        ``ndarray.__setitem__()`` is supported (including fancy
        indexing), plus a special support for expressions:

        Parameters
        ----------
        key : string
            The corresponding ctable column name will be set to `value`.  If
            not a column name, it will be interpret as a boolean expression
            (computed via `ctable.eval`) and the rows where these values are
            true will be set to `value`.

        See Also
        --------
        ctable.eval

        """

        # First, convert value into a structured array
        value = utils.to_ndarray(value, self.dtype)
        # Check if key is a condition actually
        if type(key) is bytes:
            # Convert key into a boolean array
            # key = self.eval(key)
            # The method below is faster (specially for large ctables)
            rowval = 0
            for nrow in self.where(key, outcols=["nrow__"]):
                nrow = nrow[0]
                if len(value) == 1:
                    for name in self.names:
                        self.cols[name][nrow] = value[name]
                else:
                    for name in self.names:
                        self.cols[name][nrow] = value[name][rowval]
                    rowval += 1
            return
        # Then, modify the rows
        for name in self.names:
            self.cols[name][key] = value[name]
        return

    def eval(self, expression, **kwargs):
        """
        eval(expression, **kwargs)

        Evaluate the `expression` on columns and return the result.

        Parameters
        ----------
        expression : string
            A string forming an expression, like '2*a+3*b'. The values
            for 'a' and 'b' are variable names to be taken from the
            calling function's frame.  These variables may be column
            names in this table, scalars, carrays or NumPy arrays.
        kwargs : list of parameters or dictionary
            Any parameter supported by the `eval()` first level function.

        Returns
        -------
        out : carray object
            The outcome of the expression.  You can tailor the
            properties of this carray by passing additional arguments
            supported by carray constructor in `kwargs`.

        See Also
        --------
        eval (first level function)

        """

        # Get the desired frame depth
        depth = kwargs.pop('depth', 3)
        # Call top-level eval with cols as user_dict
        return bcolz.eval(expression, user_dict=self.cols, depth=depth,
                          **kwargs)

    def flush(self):
        """Flush data in internal buffers to disk.

        This call should typically be done after performing modifications
        (__settitem__(), append()) in persistence mode.  If you don't do this,
        you risk loosing part of your modifications.

        """
        for name in self.names:
            self.cols[name].flush()

    def free_cachemem(self):
        """Get rid of internal caches to free memory.

        This call can typically be made after reading from a
        carray/ctable so as to free the memory used internally to
        cache data blocks/chunks.

        """
        for name in self.names:
            self.cols[name].free_cachemem()

    def _get_stats(self):
        """
        _get_stats()

        Get some stats (nbytes, cbytes and ratio) about this object.

        Returns
        -------
        out : a (nbytes, cbytes, ratio) tuple
            nbytes is the number of uncompressed bytes in ctable.
            cbytes is the number of compressed bytes.  ratio is the
            compression ratio.

        """

        nbytes, cbytes = 0, 0
        names, cols = self.names, self.cols
        for name in names:
            column = cols[name]
            nbytes += column.nbytes
            cbytes += column.cbytes
        cratio = nbytes / float(cbytes)
        return (nbytes, cbytes, cratio)

    def __str__(self):
        return array2string(self)

    def __repr__(self):
        nbytes, cbytes, cratio = self._get_stats()
        snbytes = utils.human_readable_size(nbytes)
        scbytes = utils.human_readable_size(cbytes)
        header = "ctable(%s, %s)\n" % (self.shape, self.dtype)
        header += "  nbytes: %s; cbytes: %s; ratio: %.2f\n" % (
            snbytes, scbytes, cratio)
        header += "  cparams := %r\n" % self.cparams
        if self.rootdir:
            header += "  rootdir := '%s'\n" % self.rootdir
        fullrepr = header + str(self)
        return fullrepr


# Local Variables:
# mode: python
# tab-width: 4
# fill-column: 78
# End:
